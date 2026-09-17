"""
Phase 4, file 1: a read-only JSON API over lab.db.

Standard library only - http.server and sqlite3. No Flask, no FastAPI. The
point is that you can read this file and know exactly what SQL produces every
number the UI shows; a framework would hide the interesting half.

Design decisions worth understanding:

GROUPS ARE THE COMPARISON UNIT. We measured a 1-2% within-session noise floor
and ~4% between sessions, so a single trial is not a result. /api/groups
returns one entry per run_group with median and spread; /api/runs exists for
drill-down only. Encoding that here means the UI cannot accidentally compare
two single rows and call it a finding.

NULL STAYS NULL. sqlite3 gives None, json.dumps writes null, and the UI must
render that as "unavailable" rather than 0. A zero in gpu_percent would average
into a chart and quietly lie.

MEDIANS ARE COMPUTED IN PYTHON. SQLite has no median function, and AVG is the
wrong statistic for latency - one GPU hiccup drags it around while the median
shrugs.

Usage:
    python api.py                # serves on http://127.0.0.1:8010
    python api.py --port 8011
"""

import argparse
import json
import os
import sqlite3
import statistics
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# So `import servers` works regardless of the directory this is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import servers  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "lab.db")


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def spread(values):
    """Median plus enough context to judge whether a difference is real."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    out = {"median": statistics.median(values),
           "min": min(values), "max": max(values), "n": len(values)}
    if len(values) > 1:
        sd = statistics.stdev(values)
        mean = statistics.mean(values)
        out["sd"] = sd
        # Coefficient of variation: the honest noise figure. Careful reading it
        # on small absolute values - TTFT of 12 ms swings 17% over a 6 ms range.
        out["cv"] = (sd / mean * 100) if mean else None
    return out


def groups(conn, limit=100):
    """
    One entry per run_group. This is what the dashboard and comparison views
    read; it is the level at which the measurements are meaningful.
    """
    rows = conn.execute("""
        SELECT * FROM inference_runs ORDER BY timestamp_utc DESC
    """).fetchall()

    # Keyed on (run_group, concurrency), NOT run_group alone. A concurrency
    # sweep stores every level under one run_group and distinguishes them by
    # the concurrency column - grouping on run_group alone silently averages
    # level 1 together with level 8 and reports the result as one measurement.
    by_group = {}
    for r in rows:
        by_group.setdefault((r["run_group"], r["concurrency"]), []).append(r)

    out = []
    for (name, _level), trials in by_group.items():
        first = trials[0]
        vram_peak = max((t["gpu_vram_used_mb"] for t in trials
                         if t["gpu_vram_used_mb"] is not None), default=None)
        baseline = first["gpu_vram_baseline_mb"]
        out.append({
            "run_group": name,
            "timestamp_utc": min(t["timestamp_utc"] for t in trials),
            "backend": first["backend"],
            "backend_version": first["backend_version"],
            "model": first["model"],
            "model_key": first["model_key"],
            "quantization": first["quantization"],
            "context_length": first["context_length"],
            "concurrency": first["concurrency"],
            "trials": len(trials),
            "notes": first["notes"],
            "decode_tok_s": spread([t["decode_tokens_per_s"] for t in trials]),
            "ttft_ms": spread([t["ttft_ms"] for t in trials]),
            "latency_ms": spread([t["client_total_ms"] for t in trials]),
            "output_tokens": spread([t["output_tokens"] for t in trials]),
            # Aggregate throughput only exists for concurrency runs, where every
            # row of a batch carries the same wall time.
            "aggregate_tok_s": (
                sum(t["output_tokens"] or 0 for t in trials)
                / (first["batch_wall_ms"] / 1000.0)
                if first["batch_wall_ms"] else None),
            "gpu_percent": spread([t["gpu_percent"] for t in trials]),
            "gpu_clock_mhz": spread([t["gpu_clock_mhz"] for t in trials]),
            "gpu_power_w": spread([t["gpu_power_w"] for t in trials]),
            "gpu_temp_c": spread([t["gpu_temp_c"] for t in trials]),
            "cpu_percent": spread([t["cpu_percent"] for t in trials]),
            # Board-wide peak, the baseline it was measured against, and the
            # difference. The UI should prefer the delta and say so - the raw
            # peak includes the desktop and browser.
            "vram_peak_mb": vram_peak,
            "vram_baseline_mb": baseline,
            "vram_delta_mb": (vram_peak - baseline)
                             if (vram_peak is not None and baseline is not None)
                             else None,
        })
    # Sort before limiting, or "limit" would return an arbitrary subset rather
    # than the most recent ones.
    out.sort(key=lambda g: (g["timestamp_utc"], g["concurrency"] or 0),
             reverse=True)
    return out[:limit]


def summary(conn):
    """Dashboard headline numbers."""
    g = groups(conn)
    single = [x for x in g if not x["concurrency"]]
    decodes = [x["decode_tok_s"]["median"] for x in single if x["decode_tok_s"]]
    ttfts = [x["ttft_ms"]["median"] for x in single if x["ttft_ms"]]
    row = conn.execute("""
        SELECT COUNT(*) AS runs,
               COUNT(DISTINCT model_key)    AS models,
               COUNT(DISTINCT backend)      AS backends,
               COUNT(DISTINCT quantization) AS quantizations,
               MAX(timestamp_utc)           AS latest
        FROM inference_runs
    """).fetchone()
    return {
        "total_runs": row["runs"],
        "run_groups": len(g),
        "models": row["models"],
        "backends": row["backends"],
        "quantizations": row["quantizations"],
        "latest": row["latest"],
        "median_decode_tok_s": statistics.median(decodes) if decodes else None,
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        # Named so nobody mistakes it for a live reading - it is the highest
        # board-wide figure ever recorded, not what the GPU is doing now.
        "max_recorded_vram_mb": conn.execute(
            "SELECT MAX(gpu_vram_used_mb) FROM inference_runs").fetchone()[0],
    }


def runs(conn, limit=200, run_group=None):
    sql = "SELECT * FROM inference_runs"
    params = []
    if run_group:
        sql += " WHERE run_group = ?"
        params.append(run_group)
    sql += " ORDER BY timestamp_utc DESC, trial_index ASC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params)]


ROUTES = {
    "/api/summary": lambda conn, q: summary(conn),
    "/api/groups": lambda conn, q: groups(conn, int(q.get("limit", ["100"])[0])),
    "/api/runs": lambda conn, q: runs(
        conn, int(q.get("limit", ["200"])[0]), q.get("run_group", [None])[0]),
    # Cheap enough to poll every couple of seconds: two HTTP HEADs and a
    # process check. Readiness is asked, never inferred from a log line.
    "/api/servers": lambda conn, q: servers.status_all(),
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        handler = ROUTES.get(parsed.path)
        if handler is None:
            self.send_json({"error": "not found", "routes": sorted(ROUTES)}, 404)
            return
        try:
            conn = connect()
            body = handler(conn, parse_qs(parsed.query))
            conn.close()
            self.send_json(body)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def do_POST(self):
        """
        Process control: /api/servers/<name>/<start|stop>.

        POST rather than GET because these change machine state, and a GET that
        stops a server would be triggered by anything that prefetches links.
        """
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) != 4 or parts[0] != "api" or parts[1] != "servers":
            self.send_json({"error": "expected /api/servers/<name>/<action>"}, 404)
            return

        _, _, name, action = parts
        if name not in servers.SERVERS or action not in ("start", "stop"):
            self.send_json({"error": f"unknown server or action: {name}/{action}",
                            "servers": sorted(servers.SERVERS)}, 400)
            return

        try:
            if action == "start":
                # Returns as soon as the process is spawned. vLLM takes ~1 min
                # to load, so the UI polls /api/servers for readiness rather
                # than holding this request open.
                body = servers.start(name)
            else:
                body = servers.stop(name)
            self.send_json(body)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.cors()
        self.end_headers()

    def cors(self):
        # Vite's dev server runs on a different port, so the browser treats
        # every request as cross-origin and blocks it without these.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def send_json(self, payload, status=200):
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"  {self.command} {self.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"no database at {os.path.abspath(DB_PATH)}")
        print("run a benchmark first, e.g. phase1-ollama-baseline/05_benchmark.py")
        sys.exit(1)

    conn = connect()
    s = summary(conn)
    conn.close()
    print(f"lab.db: {s['total_runs']} runs in {s['run_groups']} groups, "
          f"{s['models']} models, {s['backends']} backends")
    print(f"serving on http://127.0.0.1:{args.port}")
    for route in sorted(ROUTES):
        print(f"  {route}")
    # ThreadingHTTPServer so a slow query cannot block the whole UI.
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

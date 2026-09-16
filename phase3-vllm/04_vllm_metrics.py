"""
Phase 3, file 4: asking vLLM what it was doing.

The concurrency sweep left a question open: vLLM's advantage over Ollama was
1.61x at concurrency 4 but only 1.27x at concurrency 8, and per-request rate
halved. Candidate causes were scheduler budget, KV cache exhaustion, or
preemption - all guesses.

External hardware telemetry cannot answer it, and on this machine it cannot see
WSL-hosted work at all (LibreHardwareMonitor read 0% GPU and a 2 MHz clock
while vLLM was saturating the card). But vLLM instruments itself and exposes
/metrics in Prometheus text format, which is better data anyway:

    num_requests_running     how many were ACTUALLY in the batch
    num_requests_waiting     how many were queued instead
    gpu_cache_usage_perc     how full the KV cache was
    num_preemptions_total    whether the scheduler evicted and recomputed

Prometheus text format needs no library - it is one metric per line:

    vllm:num_requests_running{model_name="qwen..."} 3.0

Gauges (running, waiting, cache usage) are sampled for their PEAK during a
level. Counters (preemptions) only ever increase, so what matters is the DELTA
across the level.

Usage:
    python 04_vllm_metrics.py --list            # what this build exposes
    python 04_vllm_metrics.py --levels 4,8      # sweep and diagnose
"""

import argparse
import importlib.util
import os
import statistics
import sys
import threading
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = _load("client", "01_openai_client.py")
conc_mod = _load("conc", "03_concurrency_compare.py")

METRICS_URL = "http://127.0.0.1:8000/metrics"

# Substrings of the metric names we care about. Matching loosely on purpose -
# vLLM renames metrics between versions, and --list shows what this build has.
INTERESTING = ["num_requests_running", "num_requests_waiting",
               "gpu_cache_usage", "kv_cache_usage", "num_preemption",
               "num_requests_swapped"]


def read_metrics(url=METRICS_URL, timeout=3):
    """
    Parse Prometheus text format into {metric_name: value}.

    Labels are stripped: this server holds one model, so the label set adds
    nothing. Histograms expose _sum/_count/_bucket lines which we keep as-is.
    """
    try:
        raw = urllib.request.urlopen(url, timeout=timeout).read()
    except Exception:
        return None

    out = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name_part, value = line.rsplit(" ", 1)
        except ValueError:
            continue
        name = name_part.split("{", 1)[0]
        try:
            out[name] = float(value)
        except ValueError:
            continue          # NaN or similar
    return out


class MetricsSampler:
    """Same pattern as HardwareSampler, but asking the server about itself."""

    def __init__(self, interval=0.05, url=METRICS_URL):
        self.interval = interval
        self.url = url
        self.samples = []
        self.available = read_metrics(url) is not None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False

    def _poll(self):
        while not self._stop.is_set():
            reading = read_metrics(self.url)
            if reading:
                self.samples.append((time.perf_counter(), reading))
            time.sleep(self.interval)

    def window(self, start, end):
        return [r for ts, r in self.samples if start <= ts <= end]


def summarise(rows, key):
    values = [r[key] for r in rows if key in r]
    if not values:
        return None, None, None
    return max(values), min(values), statistics.mean(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="4,8")
    parser.add_argument("--list", action="store_true",
                        help="print every metric this build exposes, then exit")
    args = parser.parse_args()

    probe = read_metrics()
    if probe is None:
        print(f"vLLM /metrics unreachable at {METRICS_URL}")
        print("Start the server first (see docs/phase3-vllm-setup.md).")
        sys.exit(1)

    if args.list:
        print(f"{len(probe)} metrics exposed:\n")
        for name in sorted(probe):
            mark = " <-" if any(k in name for k in INTERESTING) else ""
            print(f"  {name:<58} {probe[name]}{mark}")
        sys.exit(0)

    cfg = client.BACKENDS["vllm"]
    keys = sorted({k for k in probe if any(i in k for i in INTERESTING)})
    if not keys:
        print("none of the expected scheduler metrics are exposed by this build.")
        print("run with --list to see what is available.")
        sys.exit(1)

    print(f"tracking: {', '.join(keys)}\n")
    print("warm-up...", end="", flush=True)
    client.run_once(cfg["base_url"], cfg["model"], max_tokens=conc_mod.MAX_TOKENS)
    print(" ok\n")

    for level in [int(x) for x in args.levels.split(",")]:
        sampler = MetricsSampler()
        with sampler:
            t0 = time.perf_counter()
            done, failed, wall = conc_mod.run_level(
                cfg["base_url"], cfg["model"], level)
            t1 = time.perf_counter()

        rows = sampler.window(t0, t1)
        total_out = sum(r["output_tokens"] for r in done)

        print("=" * 76)
        print(f"concurrency {level}   wall {wall:.2f}s   "
              f"{total_out / wall:.1f} tok/s aggregate   "
              f"per-req median {statistics.median(r['decode_tok_s'] for r in done):.1f} tok/s")
        print(f"{len(rows)} metric samples during the batch")
        print("-" * 76)

        for key in keys:
            peak, low, mean = summarise(rows, key)
            if peak is None:
                continue
            if "total" in key:
                # Counter: only the change across this level is meaningful.
                print(f"  {key:<46} delta {peak - low:>10.1f}")
            else:
                print(f"  {key:<46} peak {peak:>7.2f}  mean {mean:>7.2f}")
        print()

    print("How to read this:")
    print("  num_requests_running peaking BELOW the concurrency level means the")
    print("    scheduler never had them all in one batch - they queued.")
    print("  gpu_cache_usage near 1.0 means KV blocks ran out, which forces")
    print("    preemption: evicted sequences must recompute their prefill.")
    print("  preemptions delta > 0 is the smoking gun for wasted work.")

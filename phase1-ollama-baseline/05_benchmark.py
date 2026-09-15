"""
Phase 1, file 5: the harness. Benchmark a configuration and persist it.

This is where the loose scripts become a tool. One command runs a configuration
and stores every trial, so Phases 6, 9 and 10 - which are dozens of
configurations - become feasible to run honestly.

Usage:
    python 05_benchmark.py
    python 05_benchmark.py --model qwen2.5:14b --trials 5 --notes "size ladder"
    python 05_benchmark.py --history
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA = "http://127.0.0.1:11434"


def _load(name, filename):
    """
    Python identifiers cannot start with a digit, so `import 03_repeated_trials`
    is a syntax error. We load by file path instead. The numbered filenames are
    worth keeping - they make the learning order obvious - so we pay this small
    cost rather than renaming everything.
    """
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trials_mod = _load("trials", "03_repeated_trials.py")
storage = _load("storage", "04_storage.py")
sampler_mod = _load("sampler", "06_hardware_sampler.py")


def get_json(path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        OLLAMA + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def describe_model(model):
    """
    Read the model's real metadata from the server instead of typing it in.

    This matters more than it looks: in Phase 6 the same model runs at four
    quantization levels, and a row mislabelled by hand makes the whole
    comparison wrong in a way that is very hard to spot afterwards.
    """
    show = get_json("/api/show", {"model": model})
    details = show.get("details", {})

    # /api/show has no digest, so we look the tag up in /api/tags. The digest
    # pins the exact weights - tags like ':latest' get repointed over time.
    digest = None
    for entry in get_json("/api/tags").get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            digest = entry.get("digest")
            break

    return {
        "quantization": details.get("quantization_level"),
        "parameter_size": details.get("parameter_size"),
        "model_digest": digest,
        "backend_version": get_json("/api/version").get("version"),
    }


def show_history(conn):
    """Group-level view. We compare medians of groups, never single rows."""
    rows = conn.execute("""
        SELECT run_group, model, quantization, context_length,
               COUNT(*)            AS trials,
               ROUND(AVG(decode_tokens_per_s), 1) AS avg_decode,
               ROUND(MIN(ttft_ms), 1)             AS best_ttft,
               ROUND(AVG(gpu_percent), 0)         AS gpu_pct,
               ROUND(AVG(gpu_clock_mhz), 0)       AS gpu_mhz,
               ROUND(MAX(gpu_vram_used_mb), 0)    AS vram_mb,
               MIN(timestamp_utc)                 AS started
        FROM inference_runs
        GROUP BY run_group
        ORDER BY started DESC
    """).fetchall()

    if not rows:
        print("no runs recorded yet")
        return

    print(f"{'model':<15} {'quant':<8} {'ctx':>5} {'n':>3} "
          f"{'dec t/s':>8} {'ttft':>7} {'gpu%':>6} {'MHz':>6} {'vramMB':>8}  started")
    print("-" * 95)
    for r in rows:
        dash = lambda v: "-" if v is None else v
        print(f"{r['model']:<15} {str(r['quantization']):<8} "
              f"{r['context_length']:>5} {r['trials']:>3} "
              f"{r['avg_decode']:>8} {r['best_ttft']:>7} "
              f"{dash(r['gpu_pct']):>6} {dash(r['gpu_mhz']):>6} "
              f"{dash(r['vram_mb']):>8}  {r['started'][:19]}")


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen2.5:1.5b")
parser.add_argument("--trials", type=int, default=7)
parser.add_argument("--num-ctx", type=int, default=4096)
parser.add_argument("--notes", default=None)
parser.add_argument("--history", action="store_true", help="list past runs and exit")
args = parser.parse_args()

conn = storage.connect()

if args.history:
    show_history(conn)
    sys.exit(0)

meta = describe_model(args.model)
run_group = f"{args.model}|ctx{args.num_ctx}|{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"

print(f"model   : {args.model}  ({meta['parameter_size']}, {meta['quantization']})")
print(f"digest  : {meta['model_digest']}")
print(f"backend : ollama {meta['backend_version']}")
print(f"group   : {run_group}")

print("\nwarm-up (discarded)...", end="", flush=True)
warm = trials_mod.run_once(args.model, args.num_ctx)
print(f" {warm['client_total_ms']:.0f} ms")

# One sampler for the whole loop, sliced per trial. Starting and stopping a
# thread around each request would cost more than it measures, and a 350 ms
# trial would catch barely one sample.
hw = sampler_mod.HardwareSampler(interval=0.15)
if not hw.available:
    print()
    print("WARNING: LibreHardwareMonitor unreachable - every hardware column")
    print("  will be NULL. Start it ELEVATED, then Options > Remote Web Server.")

print("trials: ", end="", flush=True)
results = []
with hw:
    for i in range(args.trials):
        t0 = time.perf_counter()
        metrics = trials_mod.run_once(args.model, args.num_ctx)
        t1 = time.perf_counter()
        results.append(metrics)

        # Only samples taken DURING this request. Without the window, means are
        # diluted by the idle gaps between trials - which is how the 1.5b came
        # out looking like 11% GPU load while it was actually saturating it.
        hw_stats = hw.summary(t0, t1)
        ram_gb = hw_stats.get("system_ram_used_gb_peak")

        # Mean for rates (how hard was it working), peak for anything that
        # constrains us (VRAM decides whether the model fits at all).
        storage.record_run(
            conn,
            run_group=run_group,
            trial_index=i,
            backend="ollama",
            backend_version=meta["backend_version"],
            model=args.model,
            model_digest=meta["model_digest"],
            quantization=meta["quantization"],
            context_length=args.num_ctx,
            temperature=0.0,
            seed=42,
            streaming=1,
            prompt=trials_mod.PROMPT,
            response=metrics["response"],
            input_tokens=metrics["input_tokens"],
            cached_input_tokens=metrics["cached_input_tokens"],
            output_tokens=metrics["output_tokens"],
            total_tokens=metrics["total_tokens"],
            ttft_ms=metrics["ttft_ms"],
            client_total_ms=metrics["client_total_ms"],
            load_ms=metrics["load_ms"],
            prefill_ms=metrics["prefill_ms"],
            decode_ms=metrics["decode_ms"],
            server_total_ms=metrics["server_total_ms"],
            decode_tokens_per_s=metrics["decode_tok_s"],
            prefill_tokens_per_s=metrics["prefill_tok_s"],
            gpu_percent=hw_stats.get("gpu_percent_mean"),
            gpu_clock_mhz=hw_stats.get("gpu_clock_mhz_mean"),
            gpu_power_w=hw_stats.get("gpu_power_w_mean"),
            gpu_temp_c=hw_stats.get("gpu_temp_c_peak"),
            gpu_vram_used_mb=hw_stats.get("gpu_vram_used_mb_peak"),
            cpu_percent=hw_stats.get("cpu_percent_mean"),
            cpu_temp_c=hw_stats.get("cpu_temp_c_peak"),
            system_ram_used_mb=(ram_gb * 1024) if ram_gb else None,
            hardware_id="Ryzen 5600X / RX 7800 XT / 32GB DDR4",
            notes=args.notes,
        )
        print(f"{i + 1} ", end="", flush=True)
print()

print("\n" + "=" * 78)
trials_mod.summarize("time to first token", [r["ttft_ms"] for r in results], "ms")
trials_mod.summarize("decode rate", [r["decode_tok_s"] for r in results], "tok/s")
trials_mod.summarize("total wall clock", [r["client_total_ms"] for r in results], "ms")
print("=" * 78)
print(f"stored {len(results)} rows in group {run_group}\n")

show_history(conn)

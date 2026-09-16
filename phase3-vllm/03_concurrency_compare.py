"""
Phase 3, file 3: the comparison that actually matters.

Everything measured so far is single-stream decode - the workload vLLM's design
cares LEAST about. Its architecture exists for many sequences in flight:
continuous batching (new requests join a running batch instead of waiting for
it to drain), PagedAttention (KV cache in small on-demand blocks instead of
full-context slots reserved per sequence), and attention kernels written for
the batched case.

The number to beat, measured in Phase 2 on Ollama:

    concurrency 4  ->  1.90x aggregate throughput (14B), 1.75x (1.5B)

Sublinear because only the weight-streaming part of each token is shared
between sequences; per-sequence overhead barely parallelises in llama.cpp.
Whether vLLM does better on THIS card is the open question.

Method notes:

- threading.Barrier so every request starts at the same instant. Without it,
  thread startup staggers them and short requests finish before later ones
  begin - sequential requests measured as concurrency.
- Distinct prompts, so we measure batching rather than KV prefix caching.
- Fixed max_tokens, so every request does identical decode work.
- Each request is stored with its concurrency level and the wall time of the
  whole batch, so aggregate throughput is recoverable from the database.

IMPORTANT - a shared GPU invalidates this test. vLLM reserves its VRAM at
startup; if both servers are up, whichever loses the race gets paged over PCIe
and reads ~8 tok/s instead of ~120. Run one backend at a time with the card to
itself, and keep both runs in the same session.

Usage:
    python 03_concurrency_compare.py --backend vllm --levels 1,2,4,8
    python 03_concurrency_compare.py --backend ollama --levels 1,2,4,8
    python 03_concurrency_compare.py --report
"""

import argparse
import importlib.util
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = _load("client", "01_openai_client.py")
bench_mod = _load("bench", "02_benchmark_backends.py")
storage = _load("storage", "../phase1-ollama-baseline/04_storage.py")
sampler_mod = _load("sampler", "../phase1-ollama-baseline/06_hardware_sampler.py")

HARDWARE_ID = "Ryzen 5600X / RX 7800 XT / 32GB DDR4"
# 300 tokens, while the prompts ask for 500 WORDS (~600+ tokens) - so the cap
# BINDS and every request generates exactly this many. Identical decode work
# per request is what makes aggregate throughput comparable across levels.
#
# The first version used "Write one sentence about X" with a 100-token cap that
# never bound: requests generated ~25 tokens each, fixed per-request costs
# (prefill, scheduling, HTTP) dominated, and batching had almost no decode to
# amortise. vLLM's own /metrics proved the scheduler was fine - 8 of 8 running,
# nothing queued, no preemption - so the apparent "cliff" at concurrency 8 was
# a benchmark artefact, not a serving limit.
MAX_TOKENS = 300

# Distinct on purpose. Identical prompts share a KV cache prefix and we would be
# measuring prefix caching instead of batching.
PROMPTS = [
    "Write a 500-word description of the ocean.",
    "Write a 500-word description of a mountain range.",
    "Write a 500-word description of a desert.",
    "Write a 500-word description of an old-growth forest.",
    "Write a 500-word description of a river delta.",
    "Write a 500-word description of a glacier.",
    "Write a 500-word description of an active volcano.",
    "Write a 500-word description of a tropical island.",
    "Write a 500-word description of a deep canyon.",
    "Write a 500-word description of a coral reef.",
    "Write a 500-word description of the arctic tundra.",
    "Write a 500-word description of an african savannah.",
    "Write a 500-word description of a norwegian fjord.",
    "Write a 500-word description of a limestone cave.",
    "Write a 500-word description of a coastal lagoon.",
    "Write a 500-word description of a grass prairie.",
]


def run_level(base_url, model, concurrency):
    """Fire `concurrency` requests simultaneously; return them plus wall time."""
    barrier = threading.Barrier(concurrency)
    results = [None] * concurrency
    errors = [None] * concurrency

    def worker(index):
        try:
            barrier.wait()                  # everyone starts together
            results[index] = client.run_once(
                base_url, model,
                prompt=PROMPTS[index % len(PROMPTS)],
                max_tokens=MAX_TOKENS,
            )
        except Exception as exc:            # one failure must not hang the rest
            errors[index] = exc
            try:
                barrier.abort()
            except Exception:
                pass

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(concurrency)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start

    done = [r for r in results if r]
    failed = [e for e in errors if e]
    return done, failed, wall


def sweep(name, cfg, levels, notes, conn):
    print(f"\n{'=' * 80}")
    print(f"{name}  ({cfg['model']} @ {cfg['base_url']})")
    print("=" * 80)

    try:
        client.run_once(cfg["base_url"], cfg["model"], max_tokens=MAX_TOKENS)
    except Exception as exc:
        print(f"  UNREACHABLE: {exc}")
        return None
    print("warm-up ok")

    meta = bench_mod.describe(name, cfg, "FP16")
    run_group = (f"conc|{name}|{cfg['model']}|"
                 f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}")

    hw = sampler_mod.HardwareSampler(interval=0.15)
    if not hw.available:
        print("WARNING: LibreHardwareMonitor unreachable - hardware columns NULL")

    rows = []
    with hw:
        for level in levels:
            print(f"  concurrency {level:>2} ...", end="", flush=True)
            t0 = time.perf_counter()
            done, failed, wall = run_level(cfg["base_url"], cfg["model"], level)
            t1 = time.perf_counter()

            if failed:
                print(f" {len(failed)} request(s) FAILED: {failed[0]}")
            if not done:
                continue

            s = hw.summary(t0, t1)
            ram_gb = s.get("system_ram_used_gb_peak")
            total_out = sum(r["output_tokens"] for r in done)

            for i, m in enumerate(done):
                storage.record_run(
                    conn,
                    run_group=run_group, trial_index=i,
                    backend=name, backend_version=meta["backend_version"],
                    model=cfg["model"], model_digest=meta["model_digest"],
                    quantization=meta["quantization"],
                    context_length=meta["context_length"],
                    temperature=0.0, seed=42, streaming=1,
                    prompt=PROMPTS[i % len(PROMPTS)], response=m["response"],
                    input_tokens=m["input_tokens"],
                    cached_input_tokens=m["cached_input_tokens"],
                    output_tokens=m["output_tokens"],
                    total_tokens=m["total_tokens"],
                    ttft_ms=m["ttft_ms"],
                    client_total_ms=m["client_total_ms"],
                    decode_tokens_per_s=m["decode_tok_s"],
                    concurrency=level,
                    batch_wall_ms=wall * 1000,
                    gpu_percent=s.get("gpu_percent_mean"),
                    gpu_clock_mhz=s.get("gpu_clock_mhz_mean"),
                    gpu_power_w=s.get("gpu_power_w_mean"),
                    gpu_temp_c=s.get("gpu_temp_c_peak"),
                    gpu_vram_used_mb=s.get("gpu_vram_used_mb_peak"),
                    cpu_percent=s.get("cpu_percent_mean"),
                    system_ram_used_mb=(ram_gb * 1024) if ram_gb else None,
                    hardware_id=HARDWARE_ID, notes=notes,
                )

            rows.append({
                "concurrency": level,
                "wall_s": wall,
                # What the SERVER delivered in total - the capacity number.
                "aggregate": total_out / wall,
                # What ONE user experienced - what batching trades away.
                "per_req": statistics.median(r["decode_tok_s"] for r in done),
                "ttft": statistics.median(r["ttft_ms"] for r in done),
                "latency": statistics.median(r["client_total_ms"] for r in done),
                "gpu": s.get("gpu_percent_mean"),
            })
            print(f" {wall:.2f}s  {total_out / wall:.1f} tok/s aggregate")

    if not rows:
        return None

    base = rows[0]["aggregate"]
    print(f"\n{'conc':>4} {'aggregate':>11} {'per-req':>10} {'ttft ms':>9} "
          f"{'latency ms':>11} {'gpu%':>6} {'speedup':>8}")
    print("-" * 80)
    for r in rows:
        gpu = "-" if r["gpu"] is None else f"{r['gpu']:.0f}"
        print(f"{r['concurrency']:>4} {r['aggregate']:>9.1f} t/s "
              f"{r['per_req']:>8.1f} t/s {r['ttft']:>9.1f} {r['latency']:>11.1f} "
              f"{gpu:>6} {r['aggregate'] / base:>7.2f}x")
    print(f"stored in {run_group}")
    return {"backend": name, "rows": rows}


def report(conn):
    """Aggregate throughput recomputed from stored rows, not from a printout."""
    rows = conn.execute("""
        SELECT backend, concurrency,
               ROUND(SUM(output_tokens) / (MAX(batch_wall_ms) / 1000.0), 1) AS aggregate,
               ROUND(AVG(decode_tokens_per_s), 1) AS per_req,
               ROUND(AVG(ttft_ms), 1)             AS ttft,
               ROUND(AVG(gpu_percent), 0)         AS gpu,
               COUNT(*)                           AS n,
               MIN(timestamp_utc)                 AS started
        FROM inference_runs
        WHERE concurrency IS NOT NULL
        GROUP BY run_group, concurrency
        ORDER BY started, concurrency
    """).fetchall()
    if not rows:
        print("no concurrency runs recorded yet")
        return
    print(f"{'backend':<9}{'conc':>5}{'aggregate':>12}{'per-req':>10}"
          f"{'ttft':>8}{'gpu%':>6}{'n':>4}  started")
    print("-" * 80)
    for r in rows:
        gpu = "-" if r["gpu"] is None else int(r["gpu"])
        print(f"{r['backend']:<9}{r['concurrency']:>5}{r['aggregate']:>12}"
              f"{r['per_req']:>10}{r['ttft']:>8}{gpu:>6}{r['n']:>4}  "
              f"{r['started'][:19]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="vllm",
                        choices=["ollama", "vllm", "both"])
    parser.add_argument("--levels", default="1,2,4,8")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--report", action="store_true",
                        help="print stored concurrency results and exit")
    args = parser.parse_args()

    conn = storage.connect()

    if args.report:
        report(conn)
        sys.exit(0)

    levels = [int(x) for x in args.levels.split(",")]
    names = list(client.BACKENDS) if args.backend == "both" else [args.backend]

    if args.backend == "both":
        print("NOTE: running both back to back means they share the card only if")
        print("both servers are up. vLLM reserves VRAM at startup - if Ollama")
        print("cannot fit alongside it, Ollama pages over PCIe and reads ~8 t/s.")
        print("Prefer one backend at a time with the card to itself.\n")

    results = [sweep(n, client.BACKENDS[n], levels, args.notes, conn)
               for n in names]
    results = [r for r in results if r]

    if len(results) > 1:
        print(f"\n{'=' * 80}")
        print("AGGREGATE THROUGHPUT (tok/s) BY CONCURRENCY")
        print("-" * 80)
        header = "".join(f"{r['backend']:>14}" for r in results)
        print(f"{'conc':>5}{header}")
        for i, level in enumerate(levels):
            line = f"{level:>5}"
            for r in results:
                match = next((x for x in r["rows"]
                              if x["concurrency"] == level), None)
                line += f"{match['aggregate']:>13.1f}" if match else f"{'-':>14}"
            print(line)

    print()
    report(conn)

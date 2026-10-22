"""
Phase 7, file 3: the same model on the GPU and on the CPU.

Three placements of identical weights, each on a side Ollama (side_server.py):

    gpu    ROCm build, every layer on the GPU (num_gpu 999)
    ngl0   ROCm build, zero layers on the GPU (num_gpu 0). The GPU library is
           still loaded - and llama.cpp may still ship big prompt batches to
           the GPU ("op offload"). GPU load during prefill shows whether it does.
    cpu    CPU build only, no GPU library at all. The real CPU number.

Predictions (docs/phase7-plan.md, written before running):
- decode tracks memory bandwidth, so the GPU/CPU decode ratio should be near
  the bandwidth ratio. Measured ceilings (file 2): 602 GB/s VRAM read vs
  46.6 GB/s RAM read = 13x. The plan guessed 8-10x from datasheets.
- prefill is compute-bound, so its ratio should be far larger: 50-100x.

Then a thread sweep on the CPU build (1.5B): decode should stop improving once
RAM is saturated, prefill should keep scaling with cores.

Usage:
    python 03_cpu_vs_gpu.py                         # 1.5b, 7b, 14b; 2 rounds
    python 03_cpu_vs_gpu.py --models qwen2.5:1.5b --rounds 1 --trials 1
    python 03_cpu_vs_gpu.py --threads-only
"""

import argparse
import random
import sys
from datetime import datetime, timezone

import harness
import side_server

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELS = ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:14b"]
# (label, server library, num_gpu)
MODES = [("gpu", "rocm", 999), ("ngl0", "rocm", 0), ("cpu", "cpu", None)]
THREADS = [1, 2, 3, 4, 6, 12]

# 128 decode tokens: enough to time reliably, and the 14B on the CPU (~4 tok/s
# expected) stays near half a minute. 1,200 prompt words, as in Phase 6 file 01,
# so prefill is comparable with it - a 400-word smoke test left the GPU at
# ~10 TFLOPS (3,415 tok/s on the 1.5B against Phase 6's 5,950), which
# understated the GPU's prefill lead.
DECODE_TOKENS = 128
PREFILL_WORDS = 1200


def run_comparison(conn, models, rounds, n_trials, seed, stamp, notes):
    rng = random.Random(seed)
    results = {}
    for r in range(rounds):
        # Group by server so each round starts each library once, but shuffle
        # both the library order and the configs within it.
        libraries = ["rocm", "cpu"]
        rng.shuffle(libraries)
        for lib in libraries:
            configs = [(m, mode) for m in models for mode in MODES if mode[1] == lib]
            rng.shuffle(configs)
            print(f"\nround {r + 1}/{rounds}, {lib} build")
            with side_server.SideServer(lib) as server:
                version = harness.trials._ollama("/api/version").get("version")
                for model, (label, _, num_gpu) in configs:
                    res = harness.measure(
                        conn, server, model, label=label,
                        run_group=f"cpuvsgpu|{model}|{label}|{stamp}",
                        round_no=r, n_trials=n_trials,
                        decode_tokens=DECODE_TOKENS, prefill_words=PREFILL_WORDS,
                        seed_base=1000 * r + 10 * models.index(model),
                        num_gpu=num_gpu, version=version, notes=notes)
                    results.setdefault((model, label), []).append(res)
                    print(f"  {model:<14} {label:<5} decode {res['decode'] or 0:7.1f}  "
                          f"prefill {res['prefill'] or 0:8.1f} tok/s  "
                          f"GPU% {pct(res['gpu_load_decode']):>3}/{pct(res['gpu_load_prefill']):>3}  "
                          f"layers {res['layers']}"
                          + (f"  ({res['short']} SHORT)" if res["short"] else ""))
    return results


def run_threads(conn, n_trials, stamp, notes):
    model = "qwen2.5:1.5b"
    out = {}
    print(f"\nthread sweep, {model}, CPU build")
    with side_server.SideServer("cpu") as server:
        version = harness.trials._ollama("/api/version").get("version")
        for i, n in enumerate(THREADS):
            res = harness.measure(
                conn, server, model, label=f"threads{n}",
                run_group=f"cputhreads|{model}|t{n}|{stamp}", round_no=0,
                n_trials=n_trials, decode_tokens=DECODE_TOKENS,
                prefill_words=PREFILL_WORDS, seed_base=5000 + 10 * i,
                num_thread=n, version=version, notes=notes)
            out[n] = res
            print(f"  {n:>2} threads  decode {res['decode'] or 0:6.1f}  "
                  f"prefill {res['prefill'] or 0:7.1f} tok/s  CPU% {pct(res['cpu_load_decode'])}")
    return out


def med(rs, key):
    xs = sorted(x[key] for x in rs if x[key] is not None)
    return xs[len(xs) // 2] if xs else None


def pct(x):
    return "-" if x is None else f"{x:.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads-only", action="store_true")
    parser.add_argument("--skip-threads", action="store_true")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    if harness.sampler_mod.read_sensors() is None:
        sys.exit("LibreHardwareMonitor is not reachable - GPU load would be NULL, "
                 "and GPU load is how ngl0's hidden GPU work is detected.")
    harness.evict_main()
    conn = harness.storage.connect()
    ceilings = harness.latest_ceilings(conn)
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if not args.threads_only:
        results = run_comparison(conn, models, args.rounds, args.trials, args.seed,
                                 stamp, args.notes)
        harness.use_server(side_server.BASE)
        print(f"\n{'model':<14} {'mode':<5} {'decode':>8} {'prefill':>9} "
              f"{'GPU% d/p':>9} {'GB/s':>6} {'of ceil':>8}")
        print("-" * 66)
        sizes = {}
        with side_server.SideServer("cpu"):
            sizes = {m: harness.weights_gb(m) for m in models}
        for m in models:
            for label, _, _ in MODES:
                rs = results.get((m, label), [])
                dec, pre = med(rs, "decode"), med(rs, "prefill")
                gd, gp = med(rs, "gpu_load_decode"), med(rs, "gpu_load_prefill")
                bw = sizes[m] * dec if sizes.get(m) and dec else None
                ceil = ceilings["vram"] if label == "gpu" else ceilings["ram"]
                print(f"{m:<14} {label:<5} {dec or 0:>8.1f} {pre or 0:>9.1f} "
                      f"{pct(gd):>4}/{pct(gp):<4} {bw or 0:>6.1f} "
                      f"{(bw / ceil) if bw and ceil else 0:>7.0%}")
            g, c = results.get((m, "gpu"), []), results.get((m, "cpu"), [])
            if g and c and med(c, "decode") and med(c, "prefill"):
                print(f"{'':<14} GPU/CPU: decode {med(g, 'decode') / med(c, 'decode'):.1f}x, "
                      f"prefill {med(g, 'prefill') / med(c, 'prefill'):.1f}x")
        print("\nGB/s = text-weights size x decode rate; 'of ceil' is against the "
              "MEASURED ceiling\n(file 2: VRAM read for gpu, RAM read for ngl0/cpu), "
              "not the datasheet.")

    if not args.skip_threads:
        run_threads(conn, args.trials, stamp, args.notes)

    print(f"\nstored under cpuvsgpu|... / cputhreads|... {stamp}")


if __name__ == "__main__":
    main()

"""
Phase 7, file 5: same runtime, same card, same weights - two GPU libraries.

    Ollama -> llama.cpp -> ggml-hip.dll    -> ROCm (HIP, rocBLAS)  -> RX 7800 XT
    Ollama -> llama.cpp -> ggml-vulkan.dll -> Vulkan compute       -> RX 7800 XT

Only the "GPU libraries" layer of the stack differs. Ollama ships both and
prefers ROCm when both see the card; side_server.py hides ROCm to get Vulkan.

ROCm is AMD's CUDA-equivalent: vendor-specific, with hand-tuned math libraries
and kernels compiled per GPU architecture (977 MB in Ollama's install). Vulkan
is a cross-vendor graphics-and-compute API; llama.cpp's Vulkan backend is its
own shaders, compiled by the driver at load time (40 MB). Which one is faster
on a given card is an empirical question - no prediction is made here.

Measured per library and model, interleaved: decode, prefill, VRAM footprint.
Plus one partial-offload point. A smoke test had Vulkan at 13 tok/s against
ROCm's 120 with 10 of 29 layers on the GPU; both were 3-token answers, too
short to time (see harness.py). Measured properly the two are level.

Result (2026-10-22): ROCm prefills 2.3-3.7x faster (tuned GEMM libraries);
decode is mixed - Vulkan +8% on the 1.5B, ROCm ahead on 7B (1.48x) and 14B
(1.30x). Vulkan uses 300-800 MB less VRAM. See docs/phase7.md.

Usage:
    python 05_backend_swap.py
    python 05_backend_swap.py --models qwen2.5:1.5b --rounds 1 --trials 1
"""

import argparse
import random
import sys
from datetime import datetime, timezone

import harness
import side_server

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELS = ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:14b"]
LIBRARIES = ["rocm", "vulkan"]
PARTIAL = ("qwen2.5:1.5b", 10)       # (model, layers on GPU)
DECODE_TOKENS = 256                  # as Phase 6: all-GPU runs are fast
PREFILL_WORDS = 1200


def med(rs, key):
    xs = sorted(x[key] for x in rs if x.get(key) is not None)
    return xs[len(xs) // 2] if xs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--no-partial", action="store_true")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    if harness.sampler_mod.read_sensors() is None:
        sys.exit("LibreHardwareMonitor is not reachable - footprints would be NULL.")
    harness.evict_main()
    conn = harness.storage.connect()
    ceilings = harness.latest_ceilings(conn)
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    rng = random.Random(args.seed)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    configs = [(m, 999) for m in models]
    if not args.no_partial:
        configs.append(PARTIAL)

    results = {}
    for r in range(args.rounds):
        # Alternate which library goes first, so neither always runs on a
        # freshly cooled card.
        libs = LIBRARIES[:] if r % 2 == 0 else LIBRARIES[::-1]
        for lib in libs:
            order = configs[:]
            rng.shuffle(order)
            print(f"\nround {r + 1}/{args.rounds}, {lib}")
            with side_server.SideServer(lib) as server:
                version = harness.trials._ollama("/api/version").get("version")
                for model, ngl in order:
                    label = "all" if ngl == 999 else f"ngl{ngl}"
                    res = harness.measure(
                        conn, server, model, label=f"{lib} {label}",
                        run_group=f"backend|{lib}|{model}|{label}|{stamp}",
                        round_no=r, n_trials=args.trials,
                        decode_tokens=DECODE_TOKENS, prefill_words=PREFILL_WORDS,
                        seed_base=7000 + 1000 * r + 10 * configs.index((model, ngl)),
                        num_gpu=ngl, version=version, notes=args.notes)
                    results.setdefault((lib, model, ngl), []).append(res)
                    print(f"  {model:<14} {label:<5} decode {res['decode'] or 0:6.1f}  "
                          f"prefill {res['prefill'] or 0:7.1f}  "
                          f"footprint {res['footprint'] or 0:6,.0f} MB  "
                          f"layers {res['layers']}"
                          + (f"  ({res['short']} SHORT)" if res["short"] else ""))

    harness.use_server(side_server.BASE)
    with side_server.SideServer("cpu"):
        sizes = {m: harness.weights_gb(m) for m in models}

    print(f"\n{'model':<14} {'layers':<6} {'library':<7} {'decode':>8} {'prefill':>9} "
          f"{'footprint':>10} {'GB/s':>6} {'of ceil':>8}")
    print("-" * 74)
    for model, ngl in configs:
        rows = {}
        for lib in LIBRARIES:
            rs = results.get((lib, model, ngl), [])
            dec, pre, fp = med(rs, "decode"), med(rs, "prefill"), med(rs, "footprint")
            rows[lib] = (dec, pre)
            bw = sizes.get(model) * dec if ngl == 999 and sizes.get(model) and dec else None
            print(f"{model:<14} {'all' if ngl == 999 else ngl:<6} {lib:<7} {dec or 0:>8.1f} "
                  f"{pre or 0:>9.1f} {fp or 0:>9,.0f}M {bw or 0:>6.0f} "
                  f"{(bw / ceilings['vram']) if bw and ceilings['vram'] else 0:>7.0%}")
        (rd, rp), (vd, vp) = rows["rocm"], rows["vulkan"]
        if rd and vd and rp and vp:
            print(f"{'':<14} Vulkan/ROCm: decode {vd / rd:.2f}x, prefill {vp / rp:.2f}x")
    print(f"\nstored under backend|<library>|<model>|<layers>|{stamp}")


if __name__ == "__main__":
    main()

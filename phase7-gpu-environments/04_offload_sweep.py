"""
Phase 7, file 4: what happens between "fits" and "does not fit".

Part A - qwen2.5:14b, layer sweep. The 14B (9 GB) fits, so every split can be
measured: num_gpu = 0 .. all layers. llama.cpp places whole layers; the rest
run on the CPU from system RAM, and the hidden state crosses PCIe at each
boundary (small: a few KB per token at decode).

The prediction has no free parameters. Decode time per token is the sum of
each device's share:

    t(f) = f / r_gpu + (1 - f) / r_cpu        f = fraction of layers on GPU

with r_gpu and r_cpu the all-GPU and all-CPU decode rates measured in this same
run. It is harmonic, not linear: the slow device dominates, so moving 10% of
the layers off the GPU costs far more than 10% of the speed.

Part B - qwen2.5:32b (19.9 GB on a 16 GB card), the model in the user's agent
log. First Ollama's own choice (num_gpu unset), then forced placements around
it, then ALL layers forced onto the GPU. That last one is the third way "does
not fit" can go: Phase 4 found WDDM lets an oversubscribed allocation succeed
and pages the excess over PCIe, while Ollama still reports "100% GPU".

Usage:
    python 04_offload_sweep.py                  # both parts
    python 04_offload_sweep.py --part a --rounds 1 --trials 1
"""

import argparse
import random
import sys
from datetime import datetime, timezone

import harness
import side_server

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SWEEP_MODEL = "qwen2.5:14b"
BIG_MODEL = "qwen2.5:32b"
DECODE_TOKENS = 128
PREFILL_WORDS = 1200        # as file 3 and Phase 6, so prefill compares
ALL = 999                   # "every layer" - llama.cpp clamps to the real count

# A typical agent turn, for the per-turn cost: a fresh 2,000-token chunk of
# prompt (tool output, new messages) and a 200-token reply.
TURN_PROMPT, TURN_REPLY = 2000, 200


def turn_seconds(res):
    if not res["decode"] or not res["prefill"]:
        return None
    return TURN_PROMPT / res["prefill"] + TURN_REPLY / res["decode"]


def run_points(conn, server, model, points, rounds, n_trials, rng, stamp, tag,
               version, notes):
    """points: list of num_gpu values (None = let Ollama decide)."""
    results = {p: [] for p in points}
    for r in range(rounds):
        order = points[:]
        rng.shuffle(order)
        print(f"\n{model}, round {r + 1}/{rounds}")
        for p in order:
            label = "default" if p is None else f"ngl{p}"
            res = harness.measure(
                conn, server, model, label=f"{tag} {label}",
                run_group=f"offload|{model}|{label}|{stamp}", round_no=r,
                n_trials=n_trials, decode_tokens=DECODE_TOKENS,
                prefill_words=PREFILL_WORDS, seed_base=3000 + 1000 * r + points.index(p),
                num_gpu=p, version=version, notes=notes)
            results[p].append(res)
            placed = res["layers"]
            print(f"  {label:<8} layers {placed[0] if placed else '?':>3}/"
                  f"{placed[1] if placed and placed[1] else '?':<3} "
                  f"VRAM {res['vram_mb'] or 0:>6,.0f}/{res['size_mb'] or 0:<6,.0f} MB  "
                  f"decode {res['decode'] or 0:6.1f}  prefill {res['prefill'] or 0:7.1f}  "
                  f"load {res['load_s']:.1f}s"
                  + (f"  ({res['short']} SHORT)" if res["short"] else ""))
    return results


def med(rs, key):
    xs = sorted(x[key] for x in rs if x.get(key) is not None)
    return xs[len(xs) // 2] if xs else None


def layers_of(rs):
    for x in rs:
        if x["layers"] and x["layers"][1]:
            return x["layers"]
    return rs[0]["layers"] if rs else None


def part_a(conn, server, rounds, n_trials, rng, stamp, version, notes):
    points = [0, 8, 16, 24, 32, 40, 44, 46, 48, ALL]
    res = run_points(conn, server, SWEEP_MODEL, points, rounds, n_trials, rng,
                     stamp, "sweep", version, notes)
    total = next((layers_of(res[p])[1] for p in points
                  if layers_of(res[p]) and layers_of(res[p])[1]), None)
    r_gpu, r_cpu = med(res[ALL], "decode"), med(res[0], "decode")

    print(f"\n{SWEEP_MODEL}: decode vs layers on GPU "
          f"(prediction from the two endpoints, no fitting)")
    print(f"{'layers':>8} {'f':>5} {'decode':>8} {'predicted':>10} {'error':>7} "
          f"{'prefill':>9} {'VRAM MB':>8}")
    print("-" * 62)
    for p in points:
        placed = layers_of(res[p])
        n = placed[0] if placed else p
        f = n / total if total else None
        dec = med(res[p], "decode")
        pred = 1 / (f / r_gpu + (1 - f) / r_cpu) if f is not None and r_gpu and r_cpu else None
        err = (dec - pred) / pred if dec and pred else None
        print(f"{n:>4}/{total or '?':<3} {f or 0:>5.2f} {dec or 0:>8.1f} {pred or 0:>10.1f} "
              f"{err or 0:>+7.0%} {med(res[p], 'prefill') or 0:>9.1f} "
              f"{med(res[p], 'vram_mb') or 0:>8,.0f}")
    return res


def part_b(conn, server, rounds, n_trials, rng, stamp, version, notes, sweep):
    # Ollama's own choice first, alone, to learn where it puts the split.
    first = run_points(conn, server, BIG_MODEL, [None], 1, n_trials, rng, stamp,
                       "big", version, notes)
    placed = layers_of(first[None])
    if not placed or not placed[1]:
        print("could not read Ollama's default placement; stopping part B")
        return first
    chosen, total = placed
    print(f"\nOllama placed {chosen}/{total} layers of {BIG_MODEL} on the GPU")

    points = sorted({max(0, chosen - 16), max(0, chosen - 8), chosen + 4, ALL})
    rest = run_points(conn, server, BIG_MODEL, points, rounds, n_trials, rng, stamp,
                      "big", version, notes)
    res = {**first, **rest}
    if rounds > 1:
        more = run_points(conn, server, BIG_MODEL, [None], rounds - 1, n_trials, rng,
                          stamp, "big", version, notes)
        res[None] += more[None]

    print(f"\n{BIG_MODEL}: {'placement':<10} {'layers':>7} {'decode':>8} {'prefill':>9} "
          f"{'VRAM MB':>8} {'agent turn':>11}")
    print("-" * 66)
    for p in [None] + points:
        pl = layers_of(res[p])
        t = turn_seconds({"decode": med(res[p], "decode"), "prefill": med(res[p], "prefill")})
        print(f"{'':<14}{'default' if p is None else 'ngl ' + str(p):<10} "
              f"{pl[0] if pl else '?':>3}/{total:<3} {med(res[p], 'decode') or 0:>8.1f} "
              f"{med(res[p], 'prefill') or 0:>9.1f} {med(res[p], 'vram_mb') or 0:>8,.0f} "
              f"{t or 0:>10.1f}s")
    if sweep:
        t14 = turn_seconds({"decode": med(sweep[ALL], "decode"),
                            "prefill": med(sweep[ALL], "prefill")})
        t32 = turn_seconds({"decode": med(res[None], "decode"),
                            "prefill": med(res[None], "prefill")})
        if t14 and t32:
            print(f"\nagent turn ({TURN_PROMPT} new prompt tokens + {TURN_REPLY} reply): "
                  f"14B on GPU {t14:.1f}s, 32B split {t32:.1f}s = {t32 / t14:.1f}x slower")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["a", "b", "both"], default="both")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    if harness.sampler_mod.read_sensors() is None:
        sys.exit("LibreHardwareMonitor is not reachable - VRAM figures would be NULL.")
    harness.evict_main()
    conn = harness.storage.connect()
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    rng = random.Random(args.seed)

    with side_server.SideServer("rocm") as server:
        version = harness.trials._ollama("/api/version").get("version")
        sweep = None
        if args.part in ("a", "both"):
            sweep = part_a(conn, server, args.rounds, args.trials, rng, stamp,
                           version, args.notes)
        if args.part in ("b", "both"):
            part_b(conn, server, args.rounds, args.trials, rng, stamp, version,
                   args.notes, sweep)
    print(f"\nstored under offload|<model>|<placement>|{stamp}")


if __name__ == "__main__":
    main()

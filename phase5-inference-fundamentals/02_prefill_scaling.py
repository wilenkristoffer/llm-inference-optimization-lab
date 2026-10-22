"""
Phase 5, file 2: what TTFT is actually made of.

Phase 1 measured prefill once, at a 41-token prompt, and got 967 tokens/s -
flagged at the time as unreliable, because 42 ms at that size is mostly fixed
overhead rather than throughput. This measures it properly by sweeping prompt
length.

The model being tested:

    TTFT = intercept + slope x input_tokens

    intercept   fixed cost per request - connection, HTTP, scheduling, the
                first forward pass. Independent of prompt length.
    1 / slope   real prefill throughput, in tokens/sec.

Splitting those apart is what turns "TTFT" from a single reading into a number
you can reason about. It answers practical questions directly: how much does a
longer system prompt cost, and at what length does prefill start to dominate.

Three things the method has to get right:

1. EVERY PROMPT MUST BE UNIQUE. Repeating one would hit the KV prefix cache and
   prefill would collapse to near zero - we would be measuring caching, not
   prefill. Each trial uses a different seed.
2. max_tokens=1, so almost nothing is spent on decode and TTFT is the signal.
3. STAY UNDER num_ctx. Ollama defaults to 4096 and silently truncates past it,
   which would flatten the curve and look like the model got faster.

Usage:
    python 02_prefill_scaling.py --backend vllm
    python 02_prefill_scaling.py --backend ollama --trials 3
"""

import argparse
import importlib.util
import os
import random
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = _load("client", "../phase3-vllm/01_openai_client.py")
storage = _load("storage", "../phase1-ollama-baseline/04_storage.py")
sampler_mod = _load("sampler", "../phase1-ollama-baseline/06_hardware_sampler.py")

HARDWARE_ID = "Ryzen 5600X / RX 7800 XT / 32GB DDR4"

# Ordinary prose, because tokenization rate depends on the text (file 01: prose
# ~3.3 chars/token, numbers ~1.1). Filler that tokenized unusually would make
# the x-axis mean something different from a real prompt.
VOCAB = ("the quick brown fox jumps over a lazy dog while morning light falls "
         "across quiet fields and rivers carry water toward distant hills where "
         "old stone walls divide green pasture from darker woodland thick with "
         "birds that call between tall trees during long summer evenings when "
         "wind moves slowly through open country under wide clear skies").split()

# Word counts, not token counts: the server reports the real token count and we
# plot against that. Top end stays well under Ollama's 4096 default.
WORD_COUNTS = [40, 150, 400, 800, 1400, 2000]


def make_prompt(n_words, seed):
    """Unique filler of a given length. A fresh seed per trial defeats the
    prefix cache, which would otherwise make prefill look free."""
    rng = random.Random(seed)
    words = [rng.choice(VOCAB) for _ in range(n_words)]
    # A trailing instruction so the model has something to answer, keeping the
    # request shaped like a real one rather than a wall of noise.
    return " ".join(words) + "\n\nReply with the single word: ok"


def fit_line(xs, ys):
    """Least-squares fit plus R^2, so we can see whether it is actually linear."""
    n = len(xs)
    if n < 2:
        return None, None, None
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None, None, None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else None
    return slope, intercept, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="vllm", choices=["ollama", "vllm"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--notes", default="prefill sweep")
    args = parser.parse_args()

    cfg = client.BACKENDS[args.backend]
    conn = storage.connect()

    hw = sampler_mod.HardwareSampler(interval=0.15)
    baseline_vram = hw.baseline_vram_mb()

    print(f"{args.backend}  ({cfg['model']} @ {cfg['base_url']})")
    try:
        client.run_once(cfg["base_url"], cfg["model"], prompt="warm up",
                        max_tokens=1)
    except Exception as exc:
        print(f"  UNREACHABLE: {exc}")
        print("  Start it from the Backends panel in the UI.")
        sys.exit(1)
    print("warm-up ok\n")

    run_group = (f"prefill|{args.backend}|"
                 f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}")

    print(f"{'words':>7}{'tokens':>8}{'ttft ms':>10}{'trials':>8}")
    print("-" * 34)

    xs, ys = [], []
    with hw:
        for i, n_words in enumerate(WORD_COUNTS):
            ttfts, tokens = [], []
            for t in range(args.trials):
                # Seed varies with BOTH length and trial, so no two requests in
                # the whole sweep share a prefix.
                prompt = make_prompt(n_words, seed=n_words * 1000 + t)
                t0 = time.perf_counter()
                m = client.run_once(cfg["base_url"], cfg["model"],
                                    prompt=prompt, max_tokens=1)
                t1 = time.perf_counter()
                if m["ttft_ms"] is None:
                    continue
                ttfts.append(m["ttft_ms"])
                tokens.append(m["input_tokens"] or 0)

                s = hw.summary(t0, t1)
                storage.record_run(
                    conn,
                    run_group=run_group, trial_index=i * args.trials + t,
                    backend=args.backend, model=cfg["model"],
                    quantization="FP16", temperature=0.0, seed=42, streaming=1,
                    # The prompt is generated, so the recipe is stored instead
                    # of 8 KB of filler - fully reproducible and far smaller.
                    prompt=f"[generated filler: {n_words} words, "
                           f"seed {n_words * 1000 + t}]",
                    response=m["response"],
                    input_tokens=m["input_tokens"],
                    output_tokens=m["output_tokens"],
                    total_tokens=m["total_tokens"],
                    ttft_ms=m["ttft_ms"],
                    client_total_ms=m["client_total_ms"],
                    decode_tokens_per_s=None,
                    gpu_percent=s.get("gpu_percent_mean"),
                    gpu_clock_mhz=s.get("gpu_clock_mhz_mean"),
                    gpu_power_w=s.get("gpu_power_w_mean"),
                    gpu_vram_used_mb=s.get("gpu_vram_used_mb_peak"),
                    gpu_vram_baseline_mb=baseline_vram,
                    cpu_percent=s.get("cpu_percent_mean"),
                    hardware_id=HARDWARE_ID, notes=args.notes,
                )

            if not ttfts:
                continue
            ttfts.sort()
            median_ttft = ttfts[len(ttfts) // 2]
            median_tok = sorted(tokens)[len(tokens) // 2]
            xs.append(median_tok)
            ys.append(median_ttft)
            print(f"{n_words:>7}{median_tok:>8}{median_ttft:>10.1f}{len(ttfts):>8}")

    slope, intercept, r2 = fit_line(xs, ys)
    print("\n" + "=" * 62)
    if slope is None or slope <= 0:
        print("could not fit a line - not enough points, or TTFT did not grow")
        return

    print("TTFT = intercept + slope x input_tokens")
    print(f"  intercept  {intercept:8.1f} ms     fixed per-request cost")
    print(f"  slope      {slope:8.4f} ms/token")
    print(f"  prefill    {1000 / slope:8.0f} tokens/s")
    if r2 is not None:
        print(f"  R^2        {r2:8.4f}        1.0 would be perfectly linear")
    print("=" * 62)

    # Where the fixed cost stops dominating - the practical reading.
    crossover = intercept / slope
    print(f"\nPrefill overtakes the fixed cost at ~{crossover:.0f} input tokens.")
    print("Below that, TTFT is mostly overhead and barely depends on prompt")
    print("length. Above it, every extra token of system prompt costs real time.")
    print(f"\nFor comparison, Phase 1 measured 'prefill' at a 41-token prompt")
    print("and got 967 tokens/s. That number was dominated by the intercept.")
    print(f"\nstored {len(xs) * args.trials} rows in {run_group}")


if __name__ == "__main__":
    main()

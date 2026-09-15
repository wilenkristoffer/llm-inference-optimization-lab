"""
Phase 1, file 3: measuring the noise before trusting the signal.

Every number before this was n=1. We saw decode read ~224 then ~204 tok/s and
could not say whether that was a regression or normal variation, because we had
never measured what "normal variation" is on this machine.

This file runs the same request N times and reports median and spread.

Usage:
    python 03_repeated_trials.py
    python 03_repeated_trials.py --model qwen2.5:14b --trials 10
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # 127.0.0.1, not localhost
PROMPT = "Explain what a KV cache is in one short paragraph."


def run_once(model, num_ctx, prompt=PROMPT, temperature=0, seed=42,
             num_predict=None):
    """
    One streaming request. Returns every metric we can observe, from both
    clocks: ours (client) and the server's.

    File 5 imports this, so it returns the full set rather than only what
    this file prints.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx,
                    # num_predict caps output length. Pinning it makes every
                    # request do identical decode work, which is what lets us
                    # compare aggregate throughput across concurrency levels.
                    **({"num_predict": num_predict} if num_predict else {})},
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )

    t_sent = time.perf_counter()
    t_first = t_last = None
    pieces = []
    final = {}

    with urllib.request.urlopen(request) as response:
        for line in response:
            if not line.strip():
                continue
            now = time.perf_counter()
            chunk = json.loads(line)
            text = chunk.get("response")
            if text:
                if t_first is None:
                    t_first = now
                t_last = now
                pieces.append(text)
            if chunk.get("done"):
                final = chunk

    t_done = time.perf_counter()

    def ms(ns):
        return None if ns is None else ns / 1_000_000

    in_tokens = final.get("prompt_eval_count") or 0
    cached = final.get("prompt_eval_cached_count") or 0
    out_tokens = final.get("eval_count") or 0
    uncached = in_tokens - cached
    generation = t_last - t_first
    prefill_ms = ms(final.get("prompt_eval_duration"))

    return {
        # client clock
        "ttft_ms": (t_first - t_sent) * 1000,
        "client_total_ms": (t_done - t_sent) * 1000,
        # server clock
        "load_ms": ms(final.get("load_duration")),
        "prefill_ms": prefill_ms,
        "decode_ms": ms(final.get("eval_duration")),
        "server_total_ms": ms(final.get("total_duration")),
        # tokens
        "input_tokens": in_tokens,
        "cached_input_tokens": cached,
        "output_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        # derived. prefill rate counts only tokens actually processed.
        "decode_tok_s": (out_tokens - 1) / generation if generation > 0 else 0,
        "prefill_tok_s": (uncached / (prefill_ms / 1000))
                         if prefill_ms and uncached > 0 else None,
        "response": "".join(pieces),
    }


def summarize(label, values, unit):
    """
    Median, not mean: a single GPU hiccup drags a mean around, the median
    shrugs it off.

    Coefficient of variation (stdev as a % of the mean) is the honest noise
    figure. Under ~2% is tight. But check the absolute range too - cv inflates
    badly on small numbers, as it does for TTFT here.
    """
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    if len(values) > 1:
        sd = statistics.stdev(values)
        cv = (sd / statistics.mean(values)) * 100
        spread = f"  sd {sd:6.1f}  cv {cv:4.1f}%"
    else:
        spread = ""
    print(f"{label:22s} median {med:8.1f} {unit}   "
          f"min {lo:8.1f}  max {hi:8.1f}{spread}")


# Guarded, so file 5 can import run_once without triggering a benchmark.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--num-ctx", type=int, default=4096)
    args = parser.parse_args()

    print(f"model={args.model}  trials={args.trials}  num_ctx={args.num_ctx}")

    # The warm-up absorbs model loading and cold caches. We throw it away:
    # including it mixes a one-off 2-second cost into a steady-state figure.
    print("\nwarm-up (discarded)...", end="", flush=True)
    warm = run_once(args.model, args.num_ctx)
    print(f" {warm['client_total_ms']:.0f} ms")

    print("\ntrials: ", end="", flush=True)
    results = []
    for i in range(args.trials):
        results.append(run_once(args.model, args.num_ctx))
        print(f"{i + 1} ", end="", flush=True)
    print()

    print("\n" + "=" * 78)
    summarize("time to first token", [r["ttft_ms"] for r in results], "ms")
    summarize("decode rate", [r["decode_tok_s"] for r in results], "tok/s")
    summarize("total wall clock", [r["client_total_ms"] for r in results], "ms")
    print("=" * 78)

    # With temperature=0 every trial should generate an identical token count.
    # If not, decoding is not deterministic and every comparison in this
    # project is shakier than we assumed.
    counts = {r["output_tokens"] for r in results}
    if len(counts) == 1:
        print(f"output: {counts.pop()} tokens every trial - deterministic, comparable")
    else:
        print(f"output: token counts VARIED {sorted(counts)} - not deterministic!")

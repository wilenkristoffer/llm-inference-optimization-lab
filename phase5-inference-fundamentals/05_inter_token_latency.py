"""
Phase 5, file 5: is decode steady, or does it stutter?

Every throughput number in this project is an average: total tokens divided by
total time. An average cannot tell the difference between a metronome and a
machine that emits ten tokens instantly then stalls for 80 ms - and a reader
notices the difference immediately.

So this records the arrival time of EVERY token and looks at the gaps between
them. The gap between consecutive tokens is the inter-token latency; its
reciprocal is the decode rate we have been quoting all along.

What the numbers mean:

    p50    the typical gap - what the average approximates
    p99    the gap 1 in 100 tokens waits. Stalls live here.
    max    the worst single pause

TTFT is excluded from the gaps on purpose: it is prefill plus connection
setup, a different quantity measured in file 02. Mixing it in would show up as
one huge outlier and distort every percentile.

Usage:
    python 05_inter_token_latency.py --backend ollama
    python 05_inter_token_latency.py --backend vllm --tokens 500
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = _load("client", "../phase3-vllm/01_openai_client.py")

PROMPT = "Write a 600-word description of a coastal harbour at dawn."


def stream_arrivals(base_url, model, prompt, max_tokens):
    """
    Same SSE parsing as the Phase 3 client, but keeping the arrival time of
    every chunk rather than only the first and last. Written out here rather
    than reused because the per-token timing IS the experiment.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, "temperature": 0, "seed": 42,
        # vLLM defaults to 1.1 from generation_config.json; see Phase 3 client.
        "repetition_penalty": 1.0,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    request = urllib.request.Request(
        base_url + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer not-used-locally"}, method="POST")

    sent = time.perf_counter()
    arrivals = []
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            for choice in chunk.get("choices", []):
                if (choice.get("delta") or {}).get("content"):
                    arrivals.append(time.perf_counter())
    return sent, arrivals


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def histogram(gaps, buckets=12, width=44):
    """A text histogram, because the shape is the point and a table hides it."""
    ordered = sorted(gaps)
    lo = ordered[0]
    # Clip the axis at p99 so one 80 ms stall does not squash every real bar
    # into the first bucket. Anything beyond gets its own overflow row.
    hi = percentile(ordered, 99)
    if hi <= lo:
        hi = lo + 1
    step = (hi - lo) / buckets
    counts = [0] * buckets
    overflow = 0
    for g in gaps:
        if g > hi:
            overflow += 1
            continue
        idx = min(int((g - lo) / step), buckets - 1)
        counts[idx] += 1
    peak = max(counts + [overflow]) or 1
    for i, c in enumerate(counts):
        left = lo + i * step
        bar = "#" * max(int(c / peak * width), 1 if c else 0)
        print(f"  {left:7.2f} - {left + step:6.2f} ms  {bar} {c}")
    if overflow:
        bar = "#" * max(int(overflow / peak * width), 1)
        print(f"  {hi:7.2f} +          ms  {bar} {overflow}   <- the tail")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama", choices=["ollama", "vllm"])
    parser.add_argument("--tokens", type=int, default=400)
    args = parser.parse_args()

    cfg = client.BACKENDS[args.backend]
    print(f"{args.backend}  ({cfg['model']})   target {args.tokens} tokens")

    try:
        # Warm up so the first measured run is not paying a model load.
        client.run_once(cfg["base_url"], cfg["model"], prompt="hi", max_tokens=1)
    except Exception as exc:
        print(f"  UNREACHABLE: {exc}")
        print("  Start it from the Backends panel in the UI.")
        sys.exit(1)

    sent, arrivals = stream_arrivals(cfg["base_url"], cfg["model"],
                                     PROMPT, args.tokens)
    if len(arrivals) < 10:
        print(f"only {len(arrivals)} chunks arrived - not enough to analyse")
        return

    ttft_ms = (arrivals[0] - sent) * 1000
    gaps = [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])]
    ordered = sorted(gaps)

    print(f"\n{len(arrivals)} tokens, TTFT {ttft_ms:.1f} ms "
          f"(excluded from the gaps below)\n")

    print("INTER-TOKEN LATENCY")
    print("-" * 46)
    print(f"  mean       {statistics.mean(gaps):7.2f} ms   "
          f"-> {1000 / statistics.mean(gaps):6.1f} tok/s")
    print(f"  p50        {percentile(ordered, 50):7.2f} ms")
    print(f"  p90        {percentile(ordered, 90):7.2f} ms")
    print(f"  p99        {percentile(ordered, 99):7.2f} ms")
    print(f"  min        {ordered[0]:7.2f} ms")
    print(f"  max        {ordered[-1]:7.2f} ms")
    print(f"  stdev      {statistics.stdev(gaps):7.2f} ms")

    print("\nDISTRIBUTION")
    print("-" * 46)
    histogram(gaps)

    # How much of the total generation time the worst 1% of gaps accounts for.
    # If a handful of stalls own a large share, the average is hiding them.
    total = sum(gaps)
    worst = ordered[int(len(ordered) * 0.99):]
    share = sum(worst) / total * 100 if total else 0

    print("\nWHAT THIS SAYS")
    print("-" * 46)
    p50 = percentile(ordered, 50)
    ratio = ordered[-1] / p50 if p50 else 0
    print(f"  the slowest 1% of gaps account for {share:.1f}% of total "
          f"generation time")
    print(f"  the worst single gap is {ratio:.1f}x the median")
    if ratio < 3:
        print("\n  Decode is steady. The mean is a fair summary, and the")
        print("  tokens/sec figures quoted elsewhere in this project describe")
        print("  what a reader actually experiences.")
    else:
        print("\n  Decode STUTTERS. The mean understates what a reader sees:")
        print("  most tokens arrive on time and a few arrive late enough to")
        print("  notice. Worth finding out what the stalls line up with -")
        print("  scheduler steps, batch boundaries, or GPU clock changes.")


if __name__ == "__main__":
    main()

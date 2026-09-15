"""
Phase 2, file 2: what happens when requests arrive at the same time.

Everything so far sent one request and waited. That is not what a server does.
This sends N requests simultaneously and measures what it costs.

The concept being tested is BATCHING, and it follows from something we already
measured: decode is memory-bandwidth-bound. Generating one token means
streaming every weight in the model out of VRAM. The arithmetic units are
mostly idle while that happens.

So if two requests decode at the same time, the server can read each weight
ONCE and apply it to both sequences. The expensive part - moving weights - is
shared. In theory that means:

    2 concurrent requests  ->  ~2x aggregate throughput
                           ->  each individual request barely slower

That is the single most important reason dedicated serving runtimes exist, and
it is what Phase 3's vLLM comparison is really about.

Whether Ollama actually does this depends on OLLAMA_NUM_PARALLEL. If it is 1,
requests queue instead of batching and we will see the opposite: throughput
flat, latency rising linearly. Either result is worth knowing.

Usage:
    python 02_concurrency.py
    python 02_concurrency.py --model qwen2.5:7b --levels 1,2,4,8
"""

import argparse
import importlib.util
import os
import statistics
import sys
import threading
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trials = _load("trials", "../phase1-ollama-baseline/03_repeated_trials.py")

# Distinct prompts on purpose. Identical prompts would share a KV cache prefix,
# and we would be measuring prefix caching rather than batching.
PROMPTS = [
    "Write one sentence about oceans.",
    "Write one sentence about mountains.",
    "Write one sentence about deserts.",
    "Write one sentence about forests.",
    "Write one sentence about rivers.",
    "Write one sentence about glaciers.",
    "Write one sentence about volcanoes.",
    "Write one sentence about islands.",
]

NUM_PREDICT = 100      # every request generates exactly this many tokens


def run_level(model, num_ctx, concurrency):
    """
    Fire `concurrency` requests at the same instant and wait for all of them.

    The Barrier matters: without it, threads start staggered by however long
    thread creation takes, and short requests would finish before later ones
    began. We would be measuring sequential requests and calling it concurrency.
    """
    barrier = threading.Barrier(concurrency)
    results = [None] * concurrency

    def worker(index):
        barrier.wait()                      # everyone starts together
        results[index] = trials.run_once(
            model, num_ctx,
            prompt=PROMPTS[index % len(PROMPTS)],
            num_predict=NUM_PREDICT,
        )

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(concurrency)]

    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start

    done = [r for r in results if r]
    total_out = sum(r["output_tokens"] for r in done)

    return {
        "concurrency": concurrency,
        "wall_s": wall,
        # Aggregate throughput: what the SERVER delivered, all requests summed.
        # This is the number a capacity plan is built on.
        "aggregate_tok_s": total_out / wall,
        # Per-request decode rate: what ONE user experiences. Batching trades
        # this against the number above, and the trade is the whole point.
        "per_req_tok_s": statistics.median(r["decode_tok_s"] for r in done),
        "ttft_ms": statistics.median(r["ttft_ms"] for r in done),
        "latency_ms": statistics.median(r["client_total_ms"] for r in done),
        "out_tokens": total_out,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen2.5:1.5b")
parser.add_argument("--num-ctx", type=int, default=4096)
parser.add_argument("--levels", default="1,2,4,8")
args = parser.parse_args()

levels = [int(x) for x in args.levels.split(",")]

print(f"model={args.model}  num_predict={NUM_PREDICT}  levels={levels}")
parallel = os.environ.get("OLLAMA_NUM_PARALLEL")
if parallel is None:
    parallel = "(unset - Ollama picks its own default)"
print(f"OLLAMA_NUM_PARALLEL={parallel}")

print("\nwarm-up...", end="", flush=True)
trials.run_once(args.model, args.num_ctx, num_predict=NUM_PREDICT)
print(" done")

rows = []
for level in levels:
    print(f"running concurrency={level}...", end="", flush=True)
    row = run_level(args.model, args.num_ctx, level)
    rows.append(row)
    print(f" {row['wall_s']:.2f}s")

print("\n" + "=" * 78)
print(f"{'conc':>4} {'wall s':>8} {'aggregate':>11} {'per-req':>9} "
      f"{'ttft ms':>9} {'latency ms':>11} {'speedup':>8}")
print("-" * 78)
base = rows[0]["aggregate_tok_s"]
for r in rows:
    print(f"{r['concurrency']:>4} {r['wall_s']:>8.2f} "
          f"{r['aggregate_tok_s']:>9.1f} t/s {r['per_req_tok_s']:>7.1f} t/s "
          f"{r['ttft_ms']:>9.1f} {r['latency_ms']:>11.1f} "
          f"{r['aggregate_tok_s'] / base:>7.2f}x")
print("=" * 78)

print("""
How to read this:

  speedup near Nx        the server is BATCHING. weights are read once and
                         applied to every sequence in flight. this is close to
                         free throughput, and it is what vLLM optimizes hard.

  speedup near 1.0x      the server is QUEUEING. requests run one after
                         another, so latency grows linearly and total
                         throughput does not improve at all.

  per-request rate falls while aggregate rises
                         the expected trade. each user waits slightly longer,
                         the server serves many more users. tuning that balance
                         is most of what production LLM serving is.
""")

"""
Phase 5, file 4: measuring the KV cache prefix reuse.

We have leaned on this repeatedly without measuring it. The advice given for
the agent - "stable content first, varying content last" - rests entirely on
it, and file 02 revealed by accident that the Phase 3 TTFT figures were
warm-cache numbers rather than real prefill.

WHAT PREFIX CACHING IS

Token i's Key and Value vectors depend only on tokens 0..i - causal masking
guarantees future tokens cannot change them. So if a new prompt starts with the
same tokens as the last one, their K/V are still correct and can be reused
instead of recomputed. Prefill then only has to process the tokens that differ.

The consequence is asymmetric, and that asymmetry is the whole practical point:

    change the END of a prompt    -> everything before it still cached
    change the START of a prompt  -> nothing after it is valid, all recomputed

Four cases, in order, against the same base prompt:

    1 cold     novel text                     expect ~0 cached
    2 warm     byte-identical repeat          expect nearly all cached
    3 suffix   same start, different ending   expect nearly all cached
    4 prefix   different start, same ending   expect ~0 cached

Uses Ollama's native API because it reports `prompt_eval_cached_count` per
request - a direct instrument. vLLM caches too but does not expose per-request
hits in the response body.

Usage:
    python 04_prefix_cache.py
    python 04_prefix_cache.py --model qwen2.5:7b --filler-words 1200
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import urllib.request
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA = "http://127.0.0.1:11434"


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


storage = _load("storage", "../phase1-ollama-baseline/04_storage.py")

VOCAB = ("meadow lantern gravel harbour timber ribbon copper thunder willow "
         "anchor marble pebble cavern orchard shadow blanket compass feather "
         "granite hollow juniper kettle mineral nutmeg quarry saffron").split()


def call(payload):
    request = urllib.request.Request(
        OLLAMA + "/api/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def ask(model, prompt, num_ctx):
    return call({"model": model, "prompt": prompt, "stream": False,
                 "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx,
                             "num_predict": 1}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--filler-words", type=int, default=800)
    parser.add_argument("--num-ctx", type=int, default=4096)
    args = parser.parse_args()

    # A fresh seed every run, so case 1 is genuinely cold rather than hitting
    # whatever the last run left in the cache.
    rng = random.Random()
    body = " ".join(rng.choice(VOCAB) for _ in range(args.filler_words))

    base = f"Document.\n\n{body}\n\nSummarise in one word."
    # Only the last few tokens differ.
    suffix_changed = f"Document.\n\n{body}\n\nSummarise in three words."
    # Only the first few tokens differ - everything after is byte-identical.
    prefix_changed = f"Report.\n\n{body}\n\nSummarise in one word."

    cases = [
        ("1 cold    ", base, "novel text, nothing to reuse"),
        ("2 warm    ", base, "byte-identical repeat"),
        ("3 suffix  ", suffix_changed, "same start, different ending"),
        ("4 prefix  ", prefix_changed, "different start, same ending"),
    ]

    try:
        ask(args.model, "ping", args.num_ctx)
    except Exception as exc:
        print(f"Ollama not reachable at {OLLAMA}: {exc}")
        print("Start it from the Backends panel in the UI.")
        sys.exit(1)

    conn = storage.connect()
    run_group = f"prefixcache|{args.model}|{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"

    print(f"model {args.model}   num_ctx {args.num_ctx}\n")
    print(f"{'case':<11}{'input':>7}{'cached':>8}{'hit %':>7}"
          f"{'prefill ms':>12}  what changed")
    print("-" * 74)

    cold_prefill = None
    for i, (label, prompt, description) in enumerate(cases):
        r = ask(args.model, prompt, args.num_ctx)
        total = r.get("prompt_eval_count") or 0
        cached = r.get("prompt_eval_cached_count") or 0
        prefill_ms = (r.get("prompt_eval_duration") or 0) / 1e6
        hit = (cached / total * 100) if total else 0
        if cold_prefill is None:
            cold_prefill = prefill_ms
        print(f"{label}{total:>7}{cached:>8}{hit:>6.0f}%{prefill_ms:>12.1f}"
              f"  {description}")

        storage.record_run(
            conn, run_group=run_group, trial_index=i,
            backend="ollama", model=args.model, quantization="Q4_K_M",
            context_length=args.num_ctx, temperature=0.0, seed=42, streaming=0,
            prompt=f"[prefix-cache case: {label.strip()} - {description}]",
            response=r.get("response", "")[:200],
            input_tokens=total, cached_input_tokens=cached,
            output_tokens=r.get("eval_count"),
            total_tokens=total + (r.get("eval_count") or 0),
            prefill_ms=prefill_ms,
            decode_ms=(r.get("eval_duration") or 0) / 1e6,
            server_total_ms=(r.get("total_duration") or 0) / 1e6,
            hardware_id="Ryzen 5600X / RX 7800 XT / 32GB DDR4",
            notes="prefix cache",
        )

    print("\n" + "=" * 74)
    print("WHAT THIS MEANS")
    print("=" * 74)
    print("""
  Cases 2 and 3 should be nearly fully cached; case 4 should not be, even
  though it differs from case 3 by about the same number of tokens. The
  position of the change is what matters, not its size.

  That is the whole basis for "stable content first, varying content last":

    system prompt + tool definitions FIRST   cached on every call
    conversation and user input LAST         only the new part is prefilled

  Put a timestamp, session id or re-ordered tool list near the start and every
  request pays full prefill - the cache is invalidated from that token onward.

  The agent log analysed in Phase 2 showed cacheHit true on only 86 of 381
  calls (23%). In a loop where the system prompt is identical every turn that
  should be close to 100%, so something varies near the start of its prompts.
""")
    print(f"stored 4 rows in {run_group}")


if __name__ == "__main__":
    main()

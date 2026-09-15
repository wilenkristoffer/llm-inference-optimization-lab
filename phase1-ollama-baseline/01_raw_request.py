"""
Phase 1, file 1: what an LLM inference request actually is.

No dependencies. No abstractions. One HTTP POST to Ollama, then we look at
every field the server hands back.

The point of this file is NOT to measure things ourselves yet. It is to see
that Ollama already reports its own timings, and that those timings separate
prefill (prompt processing) from decode (token generation).
"""

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"
PROMPT = "Explain what a KV cache is in one short paragraph."


def ns_to_s(nanoseconds):
    """Ollama reports every duration in nanoseconds. 1e9 ns = 1 second."""
    if nanoseconds is None:
        return None
    return nanoseconds / 1_000_000_000


# ---------------------------------------------------------------------------
# 1. The request itself.
#
# This is the whole of "calling an LLM": a JSON body posted over HTTP.
# stream=False means the server does all the work and replies once, at the end.
# ---------------------------------------------------------------------------
body = json.dumps({
    "model": MODEL,
    "prompt": PROMPT,
    "stream": False,
}).encode("utf-8")

request = urllib.request.Request(
    OLLAMA_URL,
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)

print(f"POST {OLLAMA_URL}")
print(f"model={MODEL}  stream=False")
print(f"prompt={PROMPT!r}")
print("\nwaiting for the server (nothing streams back, so this just blocks)...\n")

with urllib.request.urlopen(request) as response:
    payload = json.loads(response.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# 2. The generated text.
# ---------------------------------------------------------------------------
print("=" * 70)
print("RESPONSE TEXT")
print("=" * 70)
print(payload["response"].strip())


# ---------------------------------------------------------------------------
# 3. The raw payload, minus the two noisy fields.
#
# "context" is the conversation state as token IDs; it can be thousands of
# numbers long, so we replace it with its length rather than printing it.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RAW JSON FROM THE SERVER")
print("=" * 70)
readable = dict(payload)
if "context" in readable:
    readable["context"] = f"<{len(readable['context'])} token ids, omitted>"
readable["response"] = "<printed above, omitted>"
print(json.dumps(readable, indent=2))


# ---------------------------------------------------------------------------
# 4. What those timing fields actually mean.
#
# total_duration       everything, end to end, as the server saw it
# load_duration        getting the model into memory (near zero once warm)
# prompt_eval_count    input tokens  -> how many tokens your prompt became
# prompt_eval_duration PREFILL time  -> processing all input tokens
# eval_count           output tokens -> how many tokens were generated
# eval_duration        DECODE time   -> generating those output tokens
# ---------------------------------------------------------------------------
total = ns_to_s(payload.get("total_duration"))
load = ns_to_s(payload.get("load_duration"))
prefill = ns_to_s(payload.get("prompt_eval_duration"))
decode = ns_to_s(payload.get("eval_duration"))
in_tokens = payload.get("prompt_eval_count")
out_tokens = payload.get("eval_count")

# How many prompt tokens were served from the KV cache instead of being
# recomputed. Anything cached did NOT cost prefill work this request.
cached_tokens = payload.get("prompt_eval_cached_count") or 0
uncached_tokens = (in_tokens or 0) - cached_tokens

print("\n" + "=" * 70)
print("DERIVED METRICS")
print("=" * 70)
print(f"input tokens (prompt) : {in_tokens}")
print(f"  reused from KV cache: {cached_tokens}")
print(f"  actually processed  : {uncached_tokens}")
print(f"output tokens (gen)   : {out_tokens}")
print(f"total tokens          : {(in_tokens or 0) + (out_tokens or 0)}")
print()
print(f"total duration        : {total:.3f} s")
print(f"  model load          : {load:.3f} s")
print(f"  prefill  (prompt)   : {prefill:.3f} s")
print(f"  decode   (generate) : {decode:.3f} s")
print(f"  unaccounted         : {total - load - prefill - decode:.3f} s")
print()

# Two DIFFERENT speeds. This distinction is the whole lesson.
#
# Prefill speed must be computed over the tokens actually PROCESSED, not the
# tokens submitted. Dividing by prompt_eval_count when most of the prompt came
# from cache credits the GPU with work it never did, and inflates the number.
if prefill and uncached_tokens > 0:
    note = ""
    if uncached_tokens < 32:
        note = "  [too few tokens to be meaningful - mostly fixed overhead]"
    print(f"prefill speed         : {uncached_tokens / prefill:,.1f} tokens/s"
          f"   (compute-bound, parallel){note}")
elif prefill:
    print("prefill speed         : n/a (entire prompt served from KV cache)")

if decode and out_tokens:
    print(f"decode speed          : {out_tokens / decode:,.1f} tokens/s"
          "   (memory-bandwidth-bound, sequential)")

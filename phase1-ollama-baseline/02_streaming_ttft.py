"""
Phase 1, file 2: streaming, and measuring time to first token ourselves.

File 1 asked the server how long things took. Here we run our own clock,
because TTFT cannot be measured server-side - the server has no idea when its
bytes actually reached us.

Also pins temperature=0 and a fixed seed, so the same prompt finally produces
the same output twice. Without that, nothing we measure is comparable.
"""

import json
import sys
import time
import urllib.request

# Windows consoles default to cp1252 and will crash on a stray unicode token.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # NOT "localhost": on Windows that
                                                  # resolves to ::1 first, and Ollama
                                                  # listens on IPv4 only -> ~2s stall
MODEL = "qwen2.5:1.5b"
PROMPT = "Explain what a KV cache is in one short paragraph."


def ns_to_s(nanoseconds):
    return None if nanoseconds is None else nanoseconds / 1_000_000_000


body = json.dumps({
    "model": MODEL,
    "prompt": PROMPT,
    "stream": True,          # <- the change: send each token as it is produced
    "options": {
        "temperature": 0,    # greedy decoding: always take the likeliest token
        "seed": 42,          # belt and braces; matters if temperature > 0
        "num_ctx": 4096,     # pin it, so VRAM and truncation are not a variable
    },
}).encode("utf-8")

request = urllib.request.Request(
    OLLAMA_URL,
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)

print(f"{MODEL}  stream=True  temperature=0  seed=42")
print("=" * 70)

# ---------------------------------------------------------------------------
# The measurement.
#
# perf_counter() is a monotonic high-resolution clock. Never use time.time()
# for durations - it can jump backwards when the system clock is adjusted.
# ---------------------------------------------------------------------------
t_request_sent = time.perf_counter()
t_first_token = None
t_last_token = None
chunk_count = 0
final = {}

with urllib.request.urlopen(request) as response:
    # A streaming Ollama response is newline-delimited JSON: one object per
    # token, then a final object carrying done=true and all the timing fields.
    for line in response:
        if not line.strip():
            continue
        now = time.perf_counter()
        chunk = json.loads(line)

        text = chunk.get("response", "")
        if text:
            if t_first_token is None:
                t_first_token = now          # <- TTFT lands here
            t_last_token = now
            chunk_count += 1
            print(text, end="", flush=True)  # flush, or we buffer and see nothing

        if chunk.get("done"):
            final = chunk

t_done = time.perf_counter()

# ---------------------------------------------------------------------------
# Client-side metrics: what a user actually experiences.
# ---------------------------------------------------------------------------
ttft = t_first_token - t_request_sent
total_wall = t_done - t_request_sent
generation_wall = t_last_token - t_first_token

out_tokens = final.get("eval_count") or 0
in_tokens = final.get("prompt_eval_count") or 0
cached = final.get("prompt_eval_cached_count") or 0

print("\n" + "=" * 70)
print("CLIENT-SIDE (what the user feels)")
print("=" * 70)
print(f"time to first token   : {ttft * 1000:8.1f} ms   <- responsiveness")
print(f"generation (1st->last): {generation_wall * 1000:8.1f} ms")
print(f"total wall clock      : {total_wall * 1000:8.1f} ms")
print(f"streamed chunks       : {chunk_count}  (vs {out_tokens} eval_count)")

if out_tokens > 1 and generation_wall > 0:
    # Inter-token latency: the gap between consecutive tokens. Its reciprocal
    # is decode tokens/sec. Humans read at roughly 5-8 tokens/sec, so anything
    # under ~150 ms per token already feels faster than you can read.
    itl = generation_wall / (out_tokens - 1)
    print(f"inter-token latency   : {itl * 1000:8.1f} ms/token")
    print(f"decode rate (client)  : {(out_tokens - 1) / generation_wall:8.1f} tok/s")

# ---------------------------------------------------------------------------
# Server-side metrics, for comparison. The difference is our overhead:
# HTTP, JSON parsing, and Python itself.
# ---------------------------------------------------------------------------
server_total = ns_to_s(final.get("total_duration")) or 0
server_decode = ns_to_s(final.get("eval_duration")) or 0
server_prefill = ns_to_s(final.get("prompt_eval_duration")) or 0
server_load = ns_to_s(final.get("load_duration")) or 0

print("\n" + "=" * 70)
print("SERVER-SIDE (what Ollama reports)")
print("=" * 70)
print(f"load                  : {server_load * 1000:8.1f} ms")
print(f"prefill               : {server_prefill * 1000:8.1f} ms"
      f"   ({in_tokens} tokens, {cached} from cache)")
print(f"decode                : {server_decode * 1000:8.1f} ms   ({out_tokens} tokens)")
print(f"total                 : {server_total * 1000:8.1f} ms")
if server_decode:
    print(f"decode rate (server)  : {out_tokens / server_decode:8.1f} tok/s")

print("\n" + "=" * 70)
print("THE GAP")
print("=" * 70)
print(f"client total - server total = {(total_wall - server_total) * 1000:.1f} ms")
print("That gap is HTTP, JSON decoding and Python - none of it inference.")
print("Tiny over loopback. Not tiny over a real network.")

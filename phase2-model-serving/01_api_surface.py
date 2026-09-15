"""
Phase 2, file 1: the three ways to ask Ollama the same question.

Ollama exposes the same model through three different HTTP interfaces:

    /api/generate            you send one prompt string
    /api/chat                you send a list of role/content messages
    /v1/chat/completions     the same as /api/chat, in OpenAI's request and
                             response shape

The trap: ALL THREE apply the model's chat template by default. Despite the
name, /api/generate is NOT a raw completion endpoint - it drops your string
into the template as well. Only "raw": true opts out, and then the markup is
your problem.

Measured on qwen2.5:1.5b: the same question is 12 tokens raw and 41 tokens
templated. That 29-token difference is the system prompt and role markers,
paid in prefill on every request.

The third one matters for Phase 3: vLLM, TGI, llama.cpp and every hosted
provider speak the OpenAI shape. If our client targets that, swapping backends
is a base-URL change rather than a rewrite - which is what makes an
Ollama-vs-vLLM comparison honest rather than a comparison of two client paths.

Usage:
    python 01_api_surface.py
"""

import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:1.5b"
QUESTION = "What is the capital of France? Answer in one word."


def call(path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        OLLAMA + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# 0. Server introspection. A model server is not just an inference endpoint -
#    it has to tell you what it holds and what state it is in.
# ---------------------------------------------------------------------------
print("=" * 74)
print("SERVER STATE")
print("=" * 74)
print(f"version   : {call('/api/version')['version']}")
loaded = call("/api/ps").get("models", [])
if loaded:
    for m in loaded:
        print(f"loaded    : {m['name']}  {m['size'] / 1e9:.1f} GB  "
              f"ctx={m.get('context_length')}  until={m.get('expires_at', '')[:19]}")
else:
    print("loaded    : nothing resident (first request will pay a cold start)")
print(f"available : {len(call('/api/tags')['models'])} models on disk")


# ---------------------------------------------------------------------------
# 1. The chat template.
#
# A base model just continues text. Instruction-tuned models were trained on a
# specific markup that marks who is speaking, and that markup also supplies the
# token that ENDS a turn. Ollama applies it on every endpoint unless you say
# "raw": true.
# ---------------------------------------------------------------------------
template = call("/api/show", {"model": MODEL}).get("template", "")
print("\n" + "=" * 74)
print("THE CHAT TEMPLATE (what /api/chat wraps your message in)")
print("=" * 74)
print(template.strip()[:400])


# ---------------------------------------------------------------------------
# 2. Same question, four ways.
# ---------------------------------------------------------------------------
options = {"temperature": 0, "seed": 42, "num_ctx": 4096}
results = {}

# -- a single prompt string. Ollama STILL applies the chat template here.
generate = call("/api/generate", {
    "model": MODEL, "prompt": QUESTION, "stream": False, "options": options,
})
results["/api/generate"] = {
    "text": generate["response"].strip(),
    "in": generate.get("prompt_eval_count"),
    "out": generate.get("eval_count"),
    "stop": generate.get("done_reason"),
}

# -- messages instead of a string. Same template, same token count. The only
#    real difference from /api/generate is the shape of the request.
chat = call("/api/chat", {
    "model": MODEL,
    "messages": [{"role": "user", "content": QUESTION}],
    "stream": False, "options": options,
})
results["/api/chat"] = {
    "text": chat["message"]["content"].strip(),
    "in": chat.get("prompt_eval_count"),
    "out": chat.get("eval_count"),
    "stop": chat.get("done_reason"),
}

# -- OpenAI-compatible. Same work as /api/chat, different field names.
#    Note what is MISSING from the response: no load/prefill/decode durations.
#    OpenAI's schema reports token usage and nothing about timing, so on this
#    interface every latency number must be measured client-side.
openai = call("/v1/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": QUESTION}],
    "temperature": 0, "seed": 42,
})
usage = openai.get("usage", {})
results["/v1/chat/completions"] = {
    "text": openai["choices"][0]["message"]["content"].strip(),
    "in": usage.get("prompt_tokens"),
    "out": usage.get("completion_tokens"),
    "stop": openai["choices"][0].get("finish_reason"),
}

# -- raw=True is the ONLY way to bypass the template. num_predict caps it
#    because without the template there is no <|im_end|>, so the model has no
#    learned stopping point and continues text until it hits a limit.
raw = call("/api/generate", {
    "model": MODEL, "prompt": QUESTION, "raw": True, "stream": False,
    "options": dict(options, num_predict=60),
})
results["/api/generate raw=True"] = {
    "text": raw["response"].strip(),
    "in": raw.get("prompt_eval_count"),
    "out": raw.get("eval_count"),
    "stop": raw.get("done_reason"),
}

print("\n" + "=" * 74)
print("SAME QUESTION, FOUR WAYS")
print("=" * 74)
print(f"{'endpoint':<26} {'in':>4} {'out':>4} {'stop':>7}  answer")
print("-" * 74)
for endpoint, r in results.items():
    print(f"{endpoint:<26} {str(r['in']):>4} {str(r['out']):>4} "
          f"{str(r['stop']):>7}  {r['text'][:26]!r}")

overhead = results["/api/chat"]["in"] - results["/api/generate raw=True"]["in"]
print()
print(f"template overhead: {overhead} extra input tokens for identical text.")
print("Those are the system prompt and role markers the model was trained on,")
print("and you pay for them in prefill on every single request.")
print()
print("Note the stop column. Templated calls end with 'stop' - the model")
print("emitted <|im_end|>, which it learned to do. The raw call ends with")
print("'length': it had no stop token to emit and just kept going.")


# ---------------------------------------------------------------------------
# 3. Field-name mapping, for Phase 3.
#
# Ollama's native shape and OpenAI's shape carry the same facts under different
# names. A backend-agnostic client has to normalize them.
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("FIELD MAPPING (native -> OpenAI)")
print("=" * 74)
for native, oai in [
    ("prompt_eval_count", "usage.prompt_tokens"),
    ("eval_count", "usage.completion_tokens"),
    ("response / message.content", "choices[0].message.content"),
    ("prompt_eval_duration", "(absent - measure client-side)"),
    ("eval_duration", "(absent - measure client-side)"),
    ("total_duration", "(absent - measure client-side)"),
    ("done_reason", "choices[0].finish_reason"),
]:
    print(f"  {native:<30} -> {oai}")

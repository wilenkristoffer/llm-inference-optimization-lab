"""
Phase 3, file 1: one client, two backends.

Every client so far spoke Ollama's native API. To compare serving runtimes we
need one that speaks the OpenAI shape, so the SAME code drives both and the
only thing that changes is a base URL. Otherwise we would be comparing two
client paths and calling it a backend difference.

Two things differ from the Phase 1 client:

1. The stream format. Ollama's native API sends newline-delimited JSON. The
   OpenAI API sends Server-Sent Events: lines prefixed "data: ", terminated by
   a literal "data: [DONE]".

2. There are NO server-side timings. OpenAI's schema reports token usage and
   nothing about duration. So every latency number here comes from our clock -
   which is why the Phase 1 work on client-side measurement mattered.

Usage:
    python 01_openai_client.py --backend vllm
    python 01_openai_client.py --backend both --trials 5
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Both speak /v1/chat/completions. Same weights, same precision, same template -
# verified by both reporting 41 prompt tokens for the same message.
BACKENDS = {
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:1.5b-instruct-fp16",
    },
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "qwen2.5-1.5b-instruct",
    },
}

PROMPT = "Explain what a KV cache is in one short paragraph."
MAX_TOKENS = 100          # fixed, so every request does identical decode work


def run_once(base_url, model, prompt=PROMPT, max_tokens=MAX_TOKENS,
             temperature=0, seed=42):
    """
    One streaming chat request. Returns the same keys as the Phase 1
    run_once(), so the existing concurrency harness and SQLite storage work
    against this unchanged.

    Server-side fields are None: the OpenAI schema does not carry them.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        # Without this the final chunk carries no token counts. vLLM honours
        # it; if a backend ignores it we fall back to counting chunks.
        "stream_options": {"include_usage": True},
    }).encode("utf-8")

    request = urllib.request.Request(
        base_url + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 # No real key needed - neither backend validates it. Some
                 # clients refuse to send a request without the header at all.
                 "Authorization": "Bearer not-used-locally"},
        method="POST",
    )

    t_sent = time.perf_counter()
    t_first = t_last = None
    pieces = []
    usage = {}
    chunks = 0

    with urllib.request.urlopen(request) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break

            now = time.perf_counter()
            chunk = json.loads(payload)

            # The usage chunk arrives at the end and has an empty choices list.
            if chunk.get("usage"):
                usage = chunk["usage"]

            for choice in chunk.get("choices", []):
                text = (choice.get("delta") or {}).get("content")
                if text:
                    if t_first is None:
                        t_first = now       # <- TTFT, client-side only
                    t_last = now
                    chunks += 1
                    pieces.append(text)

    t_done = time.perf_counter()

    out_tokens = usage.get("completion_tokens") or chunks
    in_tokens = usage.get("prompt_tokens")
    generation = (t_last - t_first) if (t_first and t_last) else 0

    return {
        "ttft_ms": (t_first - t_sent) * 1000 if t_first else None,
        "client_total_ms": (t_done - t_sent) * 1000,
        # Not available over the OpenAI interface. NULL, never zero.
        "load_ms": None,
        "prefill_ms": None,
        "decode_ms": None,
        "server_total_ms": None,
        "cached_input_tokens": None,
        "prefill_tok_s": None,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": (in_tokens or 0) + out_tokens,
        "decode_tok_s": (out_tokens - 1) / generation if generation > 0 else 0,
        "response": "".join(pieces),
    }


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="both",
                        choices=["ollama", "vllm", "both"])
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    trials_mod = _load("trials", "../phase1-ollama-baseline/03_repeated_trials.py")
    names = list(BACKENDS) if args.backend == "both" else [args.backend]

    for name in names:
        cfg = BACKENDS[name]
        print(f"\n{'=' * 74}")
        print(f"{name}  ({cfg['model']} @ {cfg['base_url']})")
        print("=" * 74)

        try:
            warm = run_once(cfg["base_url"], cfg["model"])
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        print(f"warm-up ok: {warm['output_tokens']} tokens out, "
              f"{warm['input_tokens']} in")

        results = [run_once(cfg["base_url"], cfg["model"])
                   for _ in range(args.trials)]

        trials_mod.summarize("time to first token",
                             [r["ttft_ms"] for r in results], "ms")
        trials_mod.summarize("decode rate",
                             [r["decode_tok_s"] for r in results], "tok/s")
        trials_mod.summarize("total wall clock",
                             [r["client_total_ms"] for r in results], "ms")
        print(f"\nanswer: {results[0]['response'].strip()[:200]}")

"""
Phase 10, file 1: Stage 0 - what each candidate can do before it is scored.

Four checks, all short:

    inventory   every candidate is still installed (Ollama tags, WSL's HF
                cache). Missing ones are dropped, never downloaded.
    tools       one request with two mail tools. Does Ollama accept `tools`
                for this model, does the model emit a structured call, and is
                it the right tool with the right arguments? A model Ollama
                refuses ("does not support tools") still takes the suites that
                need no tools.
    think       for models that reason by default: think=false must give an
                answer with no thinking, think=true must give both. Phase 6
                saw Qwen3.5 spend a whole budget thinking and answer nothing.
    vllm        (--vllm) vLLM with the hermes tool parser returns a parsed
                tool_calls entry, not the call as plain text.

Results go to logs/preflight.json, which later files read to skip what a
model can't do.

Usage:
    python 01_preflight.py
    python 01_preflight.py --vllm
    python 01_preflight.py --models gpt-oss:20b ministral-3:14b
"""

import argparse
import json
import os
import subprocess
import time
import urllib.request

import common

TOOLS = [
    {"type": "function", "function": {
        "name": "mail-get-unread",
        "description": "Return the oldest unread email, or an empty result if the inbox is empty.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "mail-categorize",
        "description": "Set the category of an email.",
        "parameters": {"type": "object", "properties": {
            "email_id": {"type": "string", "description": "The email's id, e.g. E-17"},
            "category": {"type": "string", "enum": ["BILLING", "SUPPORT", "SALES", "OUT_OF_SCOPE"]}},
            "required": ["email_id", "category"]}}},
]
TOOL_PROMPT = [
    {"role": "system", "content": "You are a mail assistant. Use the tools to act; do not describe what you would do."},
    {"role": "user", "content": "Email E-17 is a question about an invoice. Categorize it."},
]
EXPECTED_CALL = ("mail-categorize", {"email_id": "E-17", "category": "BILLING"})

THINK_PROMPT = [{"role": "user", "content": "What is 17 * 23? Reply with the number only."}]


def check_tools(model):
    t0 = time.perf_counter()
    r = common.ollama_chat(model, TOOL_PROMPT, tools=TOOLS, think=_think_off(model))
    out = {"wall_ms": round((time.perf_counter() - t0) * 1000)}
    if "error" in r:
        return {**out, "accepted": False, "error": r["error"]}
    msg = r["message"]
    calls = msg.get("tool_calls") or []
    out.update(accepted=True, n_calls=len(calls), content=msg.get("content", "")[:300])
    if calls:
        fn = calls[0]["function"]
        args = fn.get("arguments") or {}
        out.update(name=fn["name"], arguments=args,
                   correct=(fn["name"], args) == EXPECTED_CALL)
    else:
        out["correct"] = False
    return out


def check_think(model):
    out = {}
    for think in (False, True):
        t0 = time.perf_counter()
        r = common.ollama_chat(model, THINK_PROMPT, think=common.think_value(model, think),
                               num_predict=2048)
        msg = r.get("message", {})
        out[f"think_{str(think).lower()}"] = {
            "error": r.get("error"),
            "content": (msg.get("content") or "").strip()[:200],
            "thinking_chars": len(msg.get("thinking") or ""),
            "output_tokens": r.get("eval_count"),
            "wall_ms": round((time.perf_counter() - t0) * 1000),
            "correct": "391" in (msg.get("content") or ""),
        }
    return out


def _think_off(model):
    return common.think_value(model, False) if common.CANDIDATES[model][2] else None


def check_vllm():
    """One tool call through vLLM's OpenAI API with the hermes parser."""
    p9 = common.p9
    model = common.VLLM_CANDIDATES[0]
    extra = ("--enable-auto-tool-choice", "--tool-call-parser", "hermes")
    with p9.VllmServer(model, extra=extra, env=p9.TUNABLEOP_ENV, label="p10-preflight") as s:
        body = {"model": model, "messages": TOOL_PROMPT, "tools": TOOLS,
                "temperature": 0, "seed": 42, "max_tokens": 256}
        r = p9._post(p9.VLLM_URL + "/v1/chat/completions", body)
        startup = s.startup_s
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    out = {"model": model, "startup_s": round(startup, 1), "n_calls": len(calls),
           "content": (msg.get("content") or "")[:300]}
    if calls:
        fn = calls[0]["function"]
        args = json.loads(fn["arguments"])
        out.update(name=fn["name"], arguments=args, correct=(fn["name"], args) == EXPECTED_CALL)
    return out


def hf_cached(repo):
    path = "~/.cache/huggingface/hub/models--" + repo.replace("/", "--")
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", f"test -d {path} && echo yes"],
                       capture_output=True, text=True, creationflags=common.p9.NO_WINDOW)
    return r.stdout.strip() == "yes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", action="store_true", help="also check vLLM's tool parser")
    ap.add_argument("--models", nargs="+",
                    help="check only these; entries for other models are kept")
    args = ap.parse_args()

    path = os.path.join(common.LOG_DIR, "preflight.json")
    old = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    report = {"ollama_version": common.ollama_version(),
              "models": dict(old.get("models", {})) if args.models else {}}
    print(report["ollama_version"])
    tags = common.ollama_tags()
    for model, (quant, family, thinks) in common.CANDIDATES.items():
        if args.models and model not in args.models:
            continue
        if model not in tags:
            print(f"{model:28} MISSING - dropped")
            report["models"][model] = {"installed": False}
            continue
        entry = {"installed": True, "digest": tags[model], "quantization": quant,
                 "family": family, "tools": check_tools(model)}
        if thinks:
            entry["think"] = check_think(model)
        t = entry["tools"]
        line = (f"{model:28} tools: " +
                ("REFUSED - " + t["error"][:60] if not t["accepted"] else
                 f"{t['n_calls']} call(s) {t.get('name', '-')} {t.get('arguments', '')} "
                 f"{'OK' if t['correct'] else 'WRONG'}"))
        print(line)
        for k, v in entry.get("think", {}).items():
            print(f"{'':28} {k}: {v['content']!r} thinking={v['thinking_chars']} chars, "
                  f"{v['output_tokens']} tokens, {v['wall_ms']} ms")
        report["models"][model] = entry
        # Unload before the next model so swaps don't stack up in VRAM.
        common.ollama_chat(model, [{"role": "user", "content": "x"}], num_predict=1, keep_alive=0)

    report["hf_cache"] = {m: hf_cached(common.p9.HF[m]) for m in common.VLLM_CANDIDATES}
    print("vLLM HF cache:", report["hf_cache"])
    if args.vllm:
        report["vllm_tools"] = check_vllm()
        print("vLLM tools:", report["vllm_tools"])

    os.makedirs(common.LOG_DIR, exist_ok=True)
    if not args.vllm and "vllm_tools" in old:
        report["vllm_tools"] = old["vllm_tools"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

"""
Phase 10, file 2: Stage 1 - quality. Runs the suites against each candidate
and scores every answer in code (no LLM judge).

    A  reasoning     last 'Answer:' line, numbers within 0.01, text exact
    B  extraction    JSON object, every field compared; score = share right
    C  tool choice   right tool and required arguments, or no tool when none fits
    D  agent         mailbox.py episodes; score = share of emails right
    E  long context  the six-digit code; truncated prompts are flagged

Requests are pinned (common.OPTIONS: temperature 0, seed 42, no penalties).
Models that reason by default run with think off unless --think on, and the
think column records which. A model Ollama refuses tools for (gemma3) skips
C and D - preflight.json says which.

Rows go to lab.db task_results, one per item. Re-running the same command
skips items already stored for that run group, model, backend and think
setting, so an interrupted run resumes.

Usage:
    python 02_quality.py --models qwen2.5:1.5b qwen2.5:14b --limit 20 --run-group p10-pilot
    python 02_quality.py --run-group p10-quality            # everything
    python 02_quality.py --backend vllm --models qwen2.5-1.5b-instruct --run-group p10-quality
    python 02_quality.py --report --run-group p10-quality
"""

import argparse
import json
import math
import os
import re
import time
import urllib.error

import common
import mailbox

SUITES = "ABCDE"
E_NUM_CTX = 18432           # 16k-token haystack plus question and answer
NUM_PREDICT = {"A": 1024, "B": 512, "C": 256, "D": 512, "E": 32}
THINK_PREDICT = 8192        # thinking needs room or it never reaches the answer


def load_suite(letter):
    with open(os.path.join(common.HERE, "suites", f"{letter}.json"), encoding="utf-8") as f:
        return json.load(f)


def pilot(items, limit):
    """The first items of each category in turn, so a pilot covers every kind."""
    by_cat = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    out, i = [], 0
    while len(out) < min(limit, len(items)):
        for group in by_cat.values():
            if i < len(group) and len(out) < limit:
                out.append(group[i])
        i += 1
    return out


# --- scoring -------------------------------------------------------------------

def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"[^\d.\-]", "", str(x).replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return None


def _same(got, want):
    if isinstance(want, (int, float)):
        g = _num(got)
        return g is not None and abs(g - want) <= 0.01
    return str(got).strip().strip(".").lower() == str(want).strip().lower()


def score_a(item, content):
    lines = [l for l in content.strip().splitlines() if l.strip()]
    found = [l for l in lines if re.match(r"\W*answer\W*:", l.strip(), re.I)]
    line = found[-1] if found else (lines[-1] if lines else "")
    got = re.sub(r"^\W*answer\W*:\s*", "", line.strip(), flags=re.I).strip("*` ")
    if item["answer_type"] == "number":
        m = re.findall(r"-?\d[\d,]*\.?\d*", got)
        got = m[0] if m else got
    ok = _same(got, item["answer"])
    reason = "ok" if ok else f"got {got[:60]!r}, want {item['answer']!r}"
    if not found:
        reason += " (no Answer line)"
    return int(ok), float(ok), reason


def _json_object(content):
    s = re.sub(r"```(?:json)?", "", content)
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def score_b(item, content):
    obj = _json_object(content)
    if obj is None:
        return 0, 0.0, "no JSON object"
    want = item["answer"]
    wrong = [k for k in want if k not in obj or not _same(obj[k], want[k])]
    extra = [k for k in obj if k not in want]
    right = len(want) - len(wrong)
    reason = "; ".join([f"{k}: got {obj.get(k)!r}, want {want[k]!r}" for k in wrong] +
                       ([f"extra keys {extra}"] if extra else [])) or "ok"
    return int(not wrong), right / len(want), reason


def _norm_arg(key, v):
    s = str(v).strip()
    if key == "start_time":
        m = re.match(r"^(\d{1,2}):(\d{2})", s)
        return f"{int(m.group(1)):02d}:{m.group(2)}" if m else s
    if key == "duration_minutes":
        n = _num(s)
        return int(n) if n is not None else s
    if key in ("category", "priority"):
        return s.upper()
    return s.lower() if key in ("email", "customer_email", "email_id") else s


def score_c(item, calls, content):
    if item["tool"] is None:
        if calls:
            return 0, 0.0, f"called {calls[0]['name']} for a no-tool request"
        return 1, 1.0, "ok"
    if not calls:
        return 0, 0.0, f"no tool call; said {content[:80]!r}"
    c = calls[0]
    if c["name"] != item["tool"]:
        return 0, 0.0, f"called {c['name']}, want {item['tool']}"
    args = c["arguments"] if isinstance(c["arguments"], dict) else {}
    wrong = [k for k, v in item["args"].items()
             if _norm_arg(k, args.get(k, "")) != _norm_arg(k, v)]
    if item["category"] == "calendar" and "title" in item["args"]:
        # Title wording is free; the meeting name must appear in it.
        if item["args"]["title"].lower() in str(args.get("title", "")).lower():
            wrong = [k for k in wrong if k != "title"]
    reason = "; ".join(f"{k}: got {args.get(k)!r}, want {item['args'][k]!r}" for k in wrong)
    n = len(item["args"])
    return int(not wrong), (n - len(wrong)) / n, reason or "ok"


def score_e(item, content, prompt_tokens, num_ctx):
    codes = re.findall(r"\b\d{6}\b", content)
    ok = bool(codes) and codes[0] == item["answer"]
    reason = "ok" if ok else f"got {codes[:1] or content[:40]!r}, want {item['answer']}"
    if prompt_tokens and prompt_tokens >= num_ctx - 128:
        reason += f" (TRUNCATED at {prompt_tokens} tokens)"
    return int(ok), float(ok), reason


# --- backends ------------------------------------------------------------------

class OllamaBackend:
    name = "ollama"

    def __init__(self, model, think, base=common.OLLAMA):
        self.model, self.base = model, base
        self.think = None if think is None else common.think_value(model, think)
        self.version = common.ollama_version()
        self.digest = common.ollama_tags().get(model)
        self.quant = common.CANDIDATES[model][0]

    def chat(self, messages, tools=None, num_predict=512, num_ctx=None):
        msgs = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                msgs.append({"role": "assistant", "content": m["content"],
                             "tool_calls": [{"function": {"name": c["function"]["name"],
                                                          "arguments": c["function"]["arguments"]}}
                                            for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        t0 = time.perf_counter()
        r = common.ollama_chat(self.model, msgs, tools=tools, think=self.think,
                               num_predict=num_predict, num_ctx=num_ctx, base=self.base)
        wall = (time.perf_counter() - t0) * 1000
        if "error" in r:
            return {"error": r["error"], "wall_ms": wall}
        msg = r["message"]
        calls = [{"id": f"call_{i}", "name": c["function"]["name"],
                  "arguments": c["function"].get("arguments") or {}}
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        content = msg.get("content") or ""
        # Tokens generated but nothing returned: Ollama's tool parser dropped
        # the output. Seen with qwen2.5:1.5b writing {"name": "mail-get-unread"}
        # with no "arguments" key. Scored as the empty answer it is, but flagged.
        empty = not content.strip() and not calls and (r.get("eval_count") or 0) > 1
        return {"content": content, "thinking": msg.get("thinking") or "",
                "tool_calls": calls, "empty_output": empty,
                "input_tokens": r.get("prompt_eval_count"),
                "output_tokens": r.get("eval_count"), "wall_ms": wall,
                "prefill_ms": r.get("prompt_eval_duration", 0) / 1e6,
                "decode_ms": r.get("eval_duration", 0) / 1e6}

    def close(self):
        common.ollama_chat(self.model, [{"role": "user", "content": "x"}], num_predict=1,
                           keep_alive=0, base=self.base)


class QwenRawBackend(OllamaBackend):
    """
    Ollama with "raw": true and the conversation rendered in Qwen2.5's own
    chat format (the Hugging Face template), tool calls parsed here.

    Why: Ollama's qwen2.5 template renders tool-calling history differently
    from Qwen's. An assistant turn with text AND calls loses the calls; several
    calls share one <tool_call> block; every tool result is its own user turn.
    Multi-call agent turns then read back in a format the model never saw.
    This backend isolates that template effect from the model.
    """
    name = "ollama-raw"

    def render(self, messages, tools):
        out = []
        sys_text = next((m["content"] for m in messages if m["role"] == "system"), None)
        if sys_text or tools:
            s = "<|im_start|>system\n" + (sys_text or "You are Qwen, created by Alibaba "
                                          "Cloud. You are a helpful assistant.")
            if tools:
                s += ("\n\n# Tools\n\nYou may call one or more functions to assist with the "
                      "user query.\n\nYou are provided with function signatures within "
                      "<tools></tools> XML tags:\n<tools>")
                for t in tools:
                    s += "\n" + json.dumps(t, ensure_ascii=False)
                s += ("\n</tools>\n\nFor each function call, return a json object with "
                      "function name and arguments within <tool_call></tool_call> XML "
                      "tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
                      "<args-json-object>}\n</tool_call>")
            out.append(s + "<|im_end|>\n")
        rest = [m for m in messages if m["role"] != "system"]
        for i, m in enumerate(rest):
            if m["role"] == "user":
                out.append(f"<|im_start|>user\n{m['content']}<|im_end|>\n")
            elif m["role"] == "assistant":
                s = "<|im_start|>assistant"
                if m.get("content"):
                    s += "\n" + m["content"]
                for c in m.get("tool_calls") or []:
                    s += ("\n<tool_call>\n" + json.dumps(
                        {"name": c["function"]["name"],
                         "arguments": c["function"]["arguments"]}, ensure_ascii=False) +
                        "\n</tool_call>")
                out.append(s + "<|im_end|>\n")
            else:
                first = i == 0 or rest[i - 1]["role"] != "tool"
                last = i == len(rest) - 1 or rest[i + 1]["role"] != "tool"
                out.append(("<|im_start|>user" if first else "") +
                           f"\n<tool_response>\n{m['content']}\n</tool_response>" +
                           ("<|im_end|>\n" if last else ""))
        return "".join(out) + "<|im_start|>assistant\n"

    def chat(self, messages, tools=None, num_predict=512, num_ctx=None):
        options = {**common.OPTIONS, "num_predict": num_predict,
                   "stop": ["<|im_end|>", "<|im_start|>"]}
        if num_ctx:
            options["num_ctx"] = num_ctx
        body = {"model": self.model, "prompt": self.render(messages, tools), "raw": True,
                "stream": False, "options": options}
        t0 = time.perf_counter()
        try:
            r = common.p9._post(common.OLLAMA + "/api/generate", body)
        except urllib.error.HTTPError as e:
            return {"error": e.read().decode("utf-8", "replace")[:300],
                    "wall_ms": (time.perf_counter() - t0) * 1000}
        wall = (time.perf_counter() - t0) * 1000
        text = r.get("response") or ""
        calls = []
        for i, block in enumerate(re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)):
            try:
                obj = json.loads(block)
                calls.append({"id": f"call_{i}", "name": obj.get("name", ""),
                              "arguments": obj.get("arguments", {})})
            except json.JSONDecodeError:
                pass
        content = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", text, flags=re.S).strip()
        return {"content": content, "thinking": "", "tool_calls": calls,
                "empty_output": False, "input_tokens": r.get("prompt_eval_count"),
                "output_tokens": r.get("eval_count"), "wall_ms": wall}


class VllmBackend:
    name = "vllm"

    def __init__(self, model, think):
        p9 = common.p9
        self.model, self.think, self.p9 = model, None, p9
        self.url = p9.VLLM_URL
        self.version = p9.vllm_version()
        self.digest, self.quant = None, "FP16"
        self.server = p9.VllmServer(model, gpu_fraction=0.75, max_len=E_NUM_CTX + 2048,
                                    extra=("--enable-auto-tool-choice",
                                           "--tool-call-parser", "hermes"),
                                    env=p9.TUNABLEOP_ENV, label="p10")
        self.server.__enter__()

    def chat(self, messages, tools=None, num_predict=512, num_ctx=None):
        msgs = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{**c, "function": {**c["function"], "arguments":
                                    json.dumps(c["function"]["arguments"])}}
                                   for c in m["tool_calls"]]
            elif m["role"] == "assistant":
                m.pop("tool_calls", None)
            msgs.append(m)
        body = {"model": self.model, "messages": msgs, "temperature": 0, "seed": 42,
                "max_tokens": num_predict}
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        try:
            r = self.p9._post(self.url + "/v1/chat/completions", body)
        except urllib.error.HTTPError as e:
            return {"error": e.read().decode("utf-8", "replace")[:300],
                    "wall_ms": (time.perf_counter() - t0) * 1000}
        wall = (time.perf_counter() - t0) * 1000
        msg = r["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = c["function"]["arguments"]
            calls.append({"id": c["id"], "name": c["function"]["name"], "arguments": args})
        u = r.get("usage") or {}
        t = r.get("timings") or {}      # llama-server only; vLLM has no split
        return {"content": msg.get("content") or "", "thinking": "", "tool_calls": calls,
                "input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                "wall_ms": wall, "prefill_ms": t.get("prompt_ms"), "decode_ms": t.get("predicted_ms")}

    def close(self):
        self.server.__exit__(None, None, None)


# --- running -------------------------------------------------------------------

def run_item(backend, suite, item, think_on):
    budget = THINK_PREDICT if think_on else NUM_PREDICT[suite]
    if suite == "D":
        def step(messages, tools):
            return backend.chat(messages, tools, num_predict=budget)
        r = mailbox.run_episode(step, item)
        return {**{k: r[k] for k in ("correct", "score", "reason", "turns", "input_tokens",
                                     "output_tokens", "wall_ms")},
                "response": json.dumps(r["transcript"], ensure_ascii=False)}
    num_ctx = E_NUM_CTX if suite == "E" else None
    messages = [{"role": "user", "content": item["prompt"]}]
    tools = None
    if suite == "C":
        messages.insert(0, {"role": "system", "content": SUITE_C["system"]})
        tools = SUITE_C["tools"]
    r = backend.chat(messages, tools, num_predict=budget, num_ctx=num_ctx)
    row = {"turns": 1, "input_tokens": r.get("input_tokens"),
           "output_tokens": r.get("output_tokens"), "wall_ms": r["wall_ms"]}
    if r.get("error"):
        return {**row, "correct": 0, "score": 0.0, "reason": "error: " + r["error"][:200],
                "response": ""}
    content = r["content"]
    if suite == "A":
        c, s, why = score_a(item, content)
    elif suite == "B":
        c, s, why = score_b(item, content)
    elif suite == "C":
        c, s, why = score_c(item, r["tool_calls"], content)
        content = json.dumps({"content": content, "tool_calls": r["tool_calls"]},
                             ensure_ascii=False)
    else:
        c, s, why = score_e(item, content, r.get("input_tokens"), num_ctx or E_NUM_CTX)
    if r.get("empty_output"):
        why += f" (empty output after {r['output_tokens']} tokens: parser dropped it)"
    return {**row, "correct": c, "score": s, "reason": why, "response": content}


SUITE_C = None


def done_ids(conn, run_group, model, backend, think):
    rows = conn.execute(
        "SELECT question_id FROM task_results WHERE run_group = ? AND model = ? AND "
        "backend = ? AND think IS ?", (run_group, model, backend, think)).fetchall()
    return {r[0] for r in rows}


def supports_tools(model):
    path = os.path.join(common.LOG_DIR, "preflight.json")
    with open(path, encoding="utf-8") as f:
        entry = json.load(f)["models"].get(model, {})
    return entry.get("tools", {}).get("accepted", True)


def run_model(conn, model, args, suites):
    thinks = model in common.CANDIDATES and common.CANDIDATES[model][2]
    think_on = thinks and args.think == "on"
    think_col = (1 if think_on else 0) if thinks else None
    if args.backend != "vllm" and not supports_tools(model):
        skipped = [s for s in suites if s in "CD"]
        suites = [s for s in suites if s not in "CD"]
        if skipped:
            print(f"  {model}: no tool support, skipping {''.join(skipped)}")
    backend_cls = {"ollama": OllamaBackend, "ollama-raw": QwenRawBackend,
                   "vllm": VllmBackend}[args.backend]
    backend = backend_cls(model, think_on if thinks else None)
    try:
        for suite in suites:
            data = load_suite(suite)
            items = pilot(data["items"], args.limit) if args.limit else data["items"]
            have = done_ids(conn, args.run_group, model, backend.name, think_col)
            todo = [it for it in items if it["id"] not in have]
            print(f"  {model} {suite}: {len(todo)} to run ({len(items) - len(todo)} stored)")
            right, t0 = 0, time.perf_counter()
            for n, item in enumerate(todo, 1):
                res = run_item(backend, suite, item, think_on or model in common.THINK_OFF)
                right += res["correct"]
                common.storage.record_rows(conn, "task_results", [{
                    "run_group": args.run_group, "backend": backend.name,
                    "backend_version": backend.version, "model": model,
                    "model_digest": backend.digest, "quantization": backend.quant,
                    "question_id": item["id"], "category": item["category"],
                    "suite": suite, "config": args.config, "think": think_col, **res}])
                if n % 10 == 0 or n == len(todo):
                    print(f"    {n}/{len(todo)}  {right}/{n} right  "
                          f"{time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        backend.close()


# --- report --------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def report(conn, run_group):
    rows = conn.execute(
        "SELECT model, backend, think, suite, COUNT(*) n, SUM(correct) k, AVG(score) s, "
        "AVG(wall_ms) w, AVG(output_tokens) o FROM task_results WHERE run_group = ? "
        "GROUP BY model, backend, think, suite ORDER BY model, backend, think, suite",
        (run_group,)).fetchall()
    print(f"{'model':28} {'be':6} {'th':2} {'S':1} {'n':>4} {'acc':>6} {'95% CI':>13} "
          f"{'score':>6} {'ms/item':>8} {'out tok':>7}")
    for r in rows:
        lo, hi = wilson(r["k"], r["n"])
        th = "-" if r["think"] is None else str(r["think"])
        print(f"{r['model']:28} {r['backend']:6} {th:2} {r['suite']} {r['n']:4} "
              f"{r['k'] / r['n']:6.1%} {lo:6.1%}-{hi:5.1%} {r['s']:6.3f} {r['w']:8.0f} "
              f"{r['o'] or 0:7.0f}")


def main():
    global SUITE_C
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--suites", default=SUITES)
    ap.add_argument("--limit", type=int, help="pilot: this many items per suite, all kinds")
    ap.add_argument("--backend", choices=["ollama", "ollama-raw", "vllm"], default="ollama")
    ap.add_argument("--think", choices=["off", "on"], default="off")
    ap.add_argument("--config", default="default")
    ap.add_argument("--run-group", required=True)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    conn = common.storage.connect()
    if args.report:
        report(conn, args.run_group)
        return
    SUITE_C = load_suite("C")
    models = args.models or (list(common.CANDIDATES) if args.backend == "ollama"
                             else common.VLLM_CANDIDATES)
    for model in models:
        print(model, flush=True)
        run_model(conn, model, args, list(args.suites))
    report(conn, args.run_group)


if __name__ == "__main__":
    main()

"""
Phase 9, file 6: prefix caching on agent-shaped prompts (Stage 3).

The prompt is what an agent sends every turn: a ~2,000-token stable system
prompt (instructions, tool definitions, reference notes), then a short
varying user turn. Each block below asks one question about that prompt:

    warm        same system prompt, new user turn: the best case
    ts-<f>      a fresh timestamp line inserted at fraction f of the system
                prompt (0 = first line, 1 = last). ts-0 is the "time at the
                top" anti-pattern and is the cold case; every block is
                preceded by a prime with the unchanged prompt, so the cache
                holds exactly the prefix up to the edit
    sessions    two conversations with different system prompts (same tools)
                taking turns, A B A B A B: one agent with a subagent, or two
                users. Tests what a single cache slot does when it is shared
    multiturn   one conversation growing for 6 turns, each adding a tool
                result and the model's own previous answer. Tests whether the
                server reuses the whole history or re-prefills it

Backends, all up at once on the 1.5b so every block runs on each in turn:

    ollama-np1   side Ollama, NUM_PARALLEL 1 (the default; the agent's setup)
    ollama-np4   side Ollama, NUM_PARALLEL 4
    llama-noram  llama-server with Ollama's own flags (-np 1, -b/-ub 1024)
                 but --cache-ram 0: Ollama minus its RAM prompt cache
    vllm         vLLM 1.5B FP16, prefix caching on (its default)

Ollama 0.40.9 runs qwen2.5 on llama-server, which keeps idle slots in a RAM
prompt cache (8 GiB) and restores the best match on a later request; and it
picks a slot by longest-common-prefix similarity, falling back to LRU. Both
matter for what the plan predicted. With --model qwen2.5:14b only ollama-np1
runs (one 14b is all the card holds).

TTFT is client-side, streamed, so it includes everything a caller waits for
(a RAM-cache restore is not in Ollama's prompt_eval_duration). Cached tokens
are the server's own count.

Usage:
    python 06_prefix_cache.py                      # 1.5b, all backends, ~15 min
    python 06_prefix_cache.py --model qwen2.5:14b  # Ollama only
    python 06_prefix_cache.py --report [--group p9-prefix-...]
"""

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime

import common

REPS = 6
MAX_TOKENS = 16
TURN_TOKENS = 32          # answer length kept in the multiturn history
NUM_CTX = 8192
TS_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
VLLM_MODEL = "qwen2.5-1.5b-instruct"

PREAMBLE = """You are an operations agent for a small engineering team. You work through tasks step by step, call one tool at a time, and read each tool result before deciding the next step. Never guess file contents; read them. Keep answers short and factual. When a task is done, say DONE and summarize what changed.

You have these tools. Call one by answering with a JSON object {"tool": name, "arguments": {...}}.
"""
TOOLS = [
    ("read_file", "Read a text file from the workspace.", {"path": "string", "start_line": "integer", "end_line": "integer"}),
    ("write_file", "Create or overwrite a file in the workspace.", {"path": "string", "content": "string"}),
    ("search_code", "Search the workspace for a regular expression.", {"pattern": "string", "glob": "string", "max_results": "integer"}),
    ("list_dir", "List the entries of a directory.", {"path": "string", "recursive": "boolean"}),
    ("run_shell", "Run a PowerShell command and return stdout, stderr and the exit code.", {"command": "string", "timeout_s": "integer"}),
    ("http_get", "Fetch a URL and return the status code and the body as text.", {"url": "string", "headers": "object"}),
    ("query_db", "Run a read-only SQL query against the lab database.", {"sql": "string", "limit": "integer"}),
    ("send_message", "Post a message to the team channel.", {"channel": "string", "text": "string"}),
]


def tool_text():
    out = []
    for name, desc, params in TOOLS:
        out.append(json.dumps({"name": name, "description": desc, "parameters": {
            "type": "object", "properties": {k: {"type": v} for k, v in params.items()},
            "required": list(params)[:1]}}, indent=2))
    return "\n".join(out)


def system_prompts():
    """Two system prompts, same preamble and tools, different reference notes
    (two agents built on one framework). About 2,000 tokens each."""
    docs = common.corpus_text("docs")
    head = PREAMBLE + tool_text() + "\n\nReference notes:\n"
    return head + docs[:5200], head + docs[60000:65200]


def with_timestamp(system, fraction, stamp):
    """Insert a timestamp line at a line boundary near `fraction` of the text,
    so no word is split and the rest tokenizes as before."""
    line = f"Current time: {stamp}\n"
    if fraction <= 0:
        return line + system
    if fraction >= 1:
        return system + "\n" + line
    cut = system.rfind("\n", 0, int(len(system) * fraction)) + 1
    return system[:cut] + line + system[cut:]


def tool_results():
    code = common.corpus_text("code")
    return [code[i * 1500 + 20000:(i + 1) * 1500 + 20000] for i in range(8)]


class Backend:
    def __init__(self, name, model):
        self.name, self.model = name, model

    def ask(self, messages, max_tokens=MAX_TOKENS):
        if self.name.startswith("ollama"):
            return common.stream_ollama(self.base, self.model, messages, max_tokens, NUM_CTX)
        if self.name == "vllm":
            return common.stream_openai(common.VLLM_URL, VLLM_MODEL, messages, max_tokens,
                                        repetition_penalty=1.0)
        return common.stream_openai(common.LLAMA_URL, self.model, messages, max_tokens,
                                    cache_prompt=True, repeat_penalty=1.0)


def start_backends(model, stack):
    backends = []
    if model == "qwen2.5:1.5b":
        # vLLM first: its memory profiling assumes it is alone on the card.
        stack.append(common.VllmServer(VLLM_MODEL, gpu_fraction=0.35, max_len=NUM_CTX,
                                       extra=["--enable-prompt-tokens-details"]).__enter__())
        b = Backend("vllm", model)
        backends.append(b)
    for name, port, np_ in (("ollama-np1", 11435, 1), ("ollama-np4", 11437, 4)):
        if model != "qwen2.5:1.5b" and np_ != 1:
            continue
        s = common.OllamaSide(port, np_, label=name).__enter__()
        stack.append(s)
        b = Backend(name, model)
        b.base, b.server = s.base, s
        backends.append(b)
    if model == "qwen2.5:1.5b":
        stack.append(common.LlamaServer(model, ctx=NUM_CTX, label="noram",
                                        extra=["--cache-ram", "0", "-b", "1024", "-ub", "1024"]
                                        ).__enter__())
        backends.append(Backend("llama-noram", model))
    return backends


def sweep(model, reps):
    sys_a, sys_b = system_prompts()
    results = tool_results()
    conn = common.storage.connect()
    group = "p9-prefix-" + datetime.now().strftime("%Y%m%d-%H%M")
    versions = {"vllm": common.vllm_version(), "llama": common.llama_version(),
                "ollama": "ollama 0.40.9"}
    rng = random.Random(7)
    stack = []
    common.evict_ollama()
    try:
        backends = start_backends(model, stack)
        for b in backends:          # loads the model; not recorded
            b.ask([{"role": "system", "content": "warm-up"}, {"role": "user", "content": "hi"}])
            if hasattr(b, "server"):
                print(f"{b.name}: {b.server.runner_cmd()}")
        print(f"run_group {group}, {model}, backends {[b.name for b in backends]}", flush=True)

        def record(b, rep, block, step, r, extra=""):
            cached = r["cached_input_tokens"]
            common.storage.record_run(
                conn, run_group=group, trial_index=rep,
                backend="vllm" if b.name == "vllm" else
                        "ollama" if b.name.startswith("ollama") else "llama.cpp",
                backend_version=versions["vllm" if b.name == "vllm" else
                                         "ollama" if b.name.startswith("ollama") else "llama"],
                model=VLLM_MODEL if b.name == "vllm" else model,
                quantization="FP16" if b.name == "vllm" else "Q4_K_M",
                context_length=NUM_CTX, temperature=0, seed=42, streaming=1,
                response=r["response"], input_tokens=r["input_tokens"],
                cached_input_tokens=cached, output_tokens=r["output_tokens"],
                ttft_ms=r["ttft_ms"], client_total_ms=r["client_total_ms"],
                load_ms=r.get("load_ms"), prefill_ms=r["prefill_ms"], decode_ms=r["decode_ms"],
                server_num_parallel=4 if b.name == "ollama-np4" else 1,
                workload=block, hardware_id=common.HARDWARE_ID,
                notes=f"prefix backend={b.name} block={block} step={step}{extra}")
            print(f"  rep{rep} {b.name:<12} {block:<9} {step:<8} in {r['input_tokens']:>5} "
                  f"cached {cached if cached is not None else '-':>5}  "
                  f"TTFT {r['ttft_ms']:7.1f} ms", flush=True)

        user = lambda tag: {"role": "user", "content": f"Task {tag}: list the tools you "
                                                       "would call first, in order."}
        for rep in range(reps):
            order = backends[rep % len(backends):] + backends[:rep % len(backends)]
            for b in order:
                b.ask([{"role": "system", "content": sys_a}, user(f"prime-{rep}")])
                record(b, rep, "warm", "warm", b.ask([{"role": "system", "content": sys_a},
                                                       user(f"r{rep}")]))
            for f in TS_FRACTIONS:
                for b in order:
                    b.ask([{"role": "system", "content": sys_a}, user(f"prime-{rep}")])
                    stamp = f"2026-10-23 {rng.randrange(24):02d}:{rng.randrange(60):02d}:" \
                            f"{rng.randrange(60):02d}.{rng.randrange(10**6):06d}"
                    msgs = [{"role": "system", "content": with_timestamp(sys_a, f, stamp)},
                            user(f"r{rep}")]
                    record(b, rep, f"ts-{f:g}", "edit", b.ask(msgs))
            for b in order:
                for turn in range(6):
                    s = (sys_a, sys_b)[turn % 2]
                    record(b, rep, "sessions", f"{'AB'[turn % 2]}{turn // 2}",
                           b.ask([{"role": "system", "content": s}, user(f"r{rep}-t{turn}")]))
            for b in order:
                msgs = [{"role": "system", "content": sys_a}]
                prev = None
                for turn in range(6):
                    msgs.append({"role": "user", "content":
                                 f"Tool result (read_file):\n{results[(rep + turn) % 8]}\n\n"
                                 f"What does this code do? One sentence. ({rep}-{turn})"})
                    r = b.ask(msgs, TURN_TOKENS)
                    extra = (f" prev_total={prev['input_tokens'] + prev['output_tokens']}"
                             if prev else "")
                    record(b, rep, "multiturn", f"t{turn}", r, extra)
                    msgs.append({"role": "assistant", "content": r["response"]})
                    prev = r
    finally:
        for s in reversed(stack):
            s.__exit__(None, None, None)
    print(f"done: {group}")


def report(group):
    conn = common.storage.connect()
    if not group:
        group = conn.execute("SELECT run_group FROM inference_runs WHERE run_group LIKE "
                             "'p9-prefix-%' ORDER BY id DESC LIMIT 1").fetchone()[0]
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM inference_runs WHERE run_group = ? ORDER BY id", (group,))]
    for r in rows:
        # backend=, block=, step=, prev_total= from the notes; backend becomes
        # the full name (ollama-np4, not ollama).
        r.update(dict(kv.split("=", 1) for kv in r["notes"].split()[1:]))
    print(f"run_group {group}: {len(rows)} rows, model {rows[0]['model']}\n")
    backends = list(dict.fromkeys(r["backend"] for r in rows))
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")

    by = defaultdict(list)
    for r in rows:
        by[(r["block"], r["step"], r["backend"])].append(r)

    print("TTFT median ms (cached / input tokens, median)")
    print(f"{'block':<10}" + "".join(f"{b:>22}" for b in backends))
    for block in ["warm"] + [f"ts-{f:g}" for f in TS_FRACTIONS]:
        step = "warm" if block == "warm" else "edit"
        cells = []
        for b in backends:
            g = by.get((block, step, b), [])
            cells.append(f"{med([r['ttft_ms'] for r in g]):8.1f} "
                         f"({med([r['cached_input_tokens'] or 0 for r in g]):>4}/"
                         f"{med([r['input_tokens'] for r in g]):>4})" if g else "-")
        print(f"{block:<10}" + "".join(f"{c:>22}" for c in cells))

    for block, steps in (("sessions", [f"{s}{i}" for i in range(3) for s in "AB"]),
                         ("multiturn", [f"t{i}" for i in range(6)])):
        print(f"\n{block}: TTFT median ms (uncached tokens, median)")
        print(f"{'step':<10}" + "".join(f"{b:>22}" for b in backends))
        for step in steps:
            cells = []
            for b in backends:
                g = by.get((block, step, b), [])
                cells.append(f"{med([r['ttft_ms'] for r in g]):8.1f} "
                             f"({med([r['input_tokens'] - (r['cached_input_tokens'] or 0) for r in g]):>5})"
                             if g else "-")
            print(f"{step:<10}" + "".join(f"{c:>22}" for c in cells))
    print("\nmultiturn uncached tokens beyond the new turn (input - cached - (input - prev_total)):")
    for b in backends:
        excess = [r["input_tokens"] - (r["cached_input_tokens"] or 0)
                  - (r["input_tokens"] - int(r["prev_total"]))
                  for r in rows if r["block"] == "multiturn" and "prev_total" in r
                  and r["backend"] == b]
        print(f"  {b:<12} median {med(excess)}, min {min(excess) if excess else '-'}, "
              f"max {max(excess) if excess else '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b", choices=["qwen2.5:1.5b", "qwen2.5:14b"])
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--group")
    a = ap.parse_args()
    if a.report:
        report(a.group)
    else:
        sweep(a.model, a.reps)


if __name__ == "__main__":
    main()

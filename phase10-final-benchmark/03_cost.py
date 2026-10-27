"""
Phase 10, file 3: Stage 2 - what the same work costs.

The quality run timed every episode already, but one at a time on Ollama's
defaults. This file measures cost on its own, per shortlisted model, on the
first N episodes of suite D (default 20):

    seq      one episode at a time on the user's Ollama (the default config):
             seconds per episode and per turn, prefill vs decode, pass rate
    vram     the model's llama-server process once loaded: Dedicated Usage and
             Shared Usage (WDDM overflow into system RAM) from the per-process
             counter (Phase 9), plus Ollama's own size and size_vram
    fanout   a side Ollama on 11435 with NUM_PARALLEL 4 and 16, 2n episodes at
             once: episodes per minute and pass rate. num_ctx stays 8192 (the
             quality run's), so n slots need n times the KV cache; a model that
             doesn't fit on the GPU at n is recorded and skipped, not paged
    best     only where Phases 7-9 measured something better than the default:
               qwen2.5:14b   direct llama-server with the qwen2.5:0.5b draft,
                             interleaved with the default in blocks of 5
               qwen2.5:1.5b  vLLM FP16 with the TunableOp table, alone and at
                             4 and 16 at once. Ollama drops this model's tool
                             calls (Stage 1), so it has no Ollama rows.

Why no "best" for the others: Ollama 0.40.11 already starts llama-server with
--flash-attn auto and -b/-ub 1024, which is most of what a direct server gained
in Phase 9. For the newer models a direct server would also swap Ollama's
prompt rendering and tool parsing for llama.cpp's, and ministral showed what a
parser alone does to a score.

Episode rows go to task_results (run group p10-cost, config default,
spec-0.5b, vllm-tuned, np4, np16, vllm-c4, vllm-c16); unit summaries to
logs/<run group>.json, which also lets a rerun skip finished units.

Usage:
    python 03_cost.py                                    # all of it
    python 03_cost.py --models qwen2.5:14b --episodes 2 --fanout 4 --run-group p10-cost-pilot
    python 03_cost.py --report
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import common
import mailbox

q = common._load("p10quality", "02_quality.py")
p9 = common.p9

SHORTLIST = ["qwen3.5:4b-bf16", "nemotron-3.5-bolt:e4b", "gpt-oss:20b", "qwen3:14b",
             "gemma4:12b", "qwen2.5:14b", "qwen2.5:1.5b"]
# Configs run one at a time, in this order, interleaved per block of episodes.
SEQ_CONFIGS = {"qwen2.5:14b": ["default", "spec-0.5b"], "qwen2.5:1.5b": ["vllm-tuned"]}
VLLM_MODEL = {"qwen2.5:1.5b": "qwen2.5-1.5b-instruct"}
BLOCK = 5
SIDE_PORT = 11435
SHARED_LIMIT_MB = 512       # more than this in system RAM means it is paging


class SpecBackend(q.VllmBackend):
    """llama-server's OpenAI API with a draft model; chat() is vLLM's."""
    name = "llama-server"

    def __init__(self, model, draft):
        self.model, self.think, self.p9 = model, None, p9
        self.url = p9.LLAMA_URL
        self.version = p9.llama_version()
        self.digest = common.ollama_tags().get(model)
        self.quant = common.CANDIDATES[model][0]
        self.server = p9.LlamaServer(model, draft=draft, ctx=8192, label="p10-cost")
        self.server.__enter__()


def make_backend(model, config, base=common.OLLAMA):
    if config == "spec-0.5b":
        return SpecBackend(model, "qwen2.5:0.5b")
    if config == "vllm-tuned":
        return q.VllmBackend(VLLM_MODEL[model], None)
    thinks = common.CANDIDATES[model][2]
    return q.OllamaBackend(model, False if thinks else None, base=base)


def budget(model):
    return q.THINK_PREDICT if model in common.THINK_OFF else q.NUM_PREDICT["D"]


def warm(backend):
    """Load the model before anything is timed (its load time is not a turn)."""
    backend.chat([{"role": "user", "content": "Say OK."}], num_predict=8)
    time.sleep(2)


def run_one(backend, model, ep):
    """One episode, with prefill and decode summed over its turns."""
    split = {"prefill_ms": 0.0, "decode_ms": 0.0, "ok": True}

    def step(messages, tools):
        r = backend.chat(messages, tools, num_predict=budget(model))
        if r.get("prefill_ms") is None:
            split["ok"] = False
        else:
            split["prefill_ms"] += r["prefill_ms"]
            split["decode_ms"] += r.get("decode_ms") or 0
        return r

    r = mailbox.run_episode(step, ep)
    return {"id": ep["id"], "category": ep["category"],
            **{k: r[k] for k in ("correct", "score", "reason", "turns", "input_tokens",
                                 "output_tokens", "wall_ms", "transcript")},
            "prefill_ms": split["prefill_ms"] if split["ok"] else None,
            "decode_ms": split["decode_ms"] if split["ok"] else None}


# --- VRAM ----------------------------------------------------------------------

def llama_pids():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-Process llama-server -ErrorAction SilentlyContinue).Id"],
                         capture_output=True, text=True, creationflags=p9.NO_WINDOW).stdout
    return [int(x) for x in out.split()]


def vram(base=None):
    pids = llama_pids()
    out = {"processes": len(pids),
           "dedicated_mb": round(sum(p9.process_vram_mb(p) or 0 for p in pids)),
           "shared_mb": round(sum(p9.process_vram_mb(p, "Shared Usage") or 0 for p in pids))}
    if base:
        ps = json.load(urllib.request.urlopen(base + "/api/ps", timeout=10)).get("models", [])
        if ps:
            out["ollama_size_mb"] = round(ps[0]["size"] / 2**20)
            out["ollama_size_vram_mb"] = round(ps[0]["size_vram"] / 2**20)
    return out


# --- summaries -----------------------------------------------------------------

def summarize(rows, wall_s):
    n, turns = len(rows), sum(r["turns"] for r in rows)
    walls = [r["wall_ms"] / 1000 for r in rows]
    out = {"n": n, "correct": sum(r["correct"] for r in rows),
           "s_per_episode": statistics.mean(walls),
           "s_per_episode_median": statistics.median(walls),
           "s_per_turn": sum(walls) / turns, "turns_per_episode": turns / n,
           "out_tokens_per_episode": sum(r["output_tokens"] for r in rows) / n,
           "wall_s": wall_s, "episodes_per_min": n / wall_s * 60}
    if all(r["prefill_ms"] is not None for r in rows):
        pre, dec = sum(r["prefill_ms"] for r in rows), sum(r["decode_ms"] for r in rows)
        out["prefill_share"] = pre / (pre + dec) if pre + dec else None
    return out


def store(conn, run_group, backend, config, rows):
    think = 0 if common.CANDIDATES.get(backend.model, (0, 0, False))[2] else None
    common.storage.record_rows(conn, "task_results", [{
        "run_group": run_group, "backend": backend.name,
        "backend_version": backend.version, "model": backend.model,
        "model_digest": backend.digest, "quantization": backend.quant,
        "question_id": r["id"], "category": r["category"], "suite": "D",
        "config": config, "think": think,
        **{k: r[k] for k in ("correct", "score", "reason", "turns", "input_tokens",
                             "output_tokens", "wall_ms")},
        "response": json.dumps(r["transcript"], ensure_ascii=False)} for r in rows])


def clear(conn, run_group, model, config):
    """A unit that crashed half-way is rerun whole; drop its partial rows."""
    conn.execute("DELETE FROM task_results WHERE run_group = ? AND model = ? AND config = ?",
                 (run_group, model, config))
    conn.commit()


# --- units ---------------------------------------------------------------------

def seq_unit(conn, args, model, eps):
    """Every config of the model, one episode at a time, interleaved in blocks
    so between-session drift (~4%) falls on both sides alike."""
    configs = SEQ_CONFIGS.get(model, ["default"])
    rows = {c: [] for c in configs}
    mem, names = {}, {}
    blocks = [eps[i:i + BLOCK] for i in range(0, len(eps), BLOCK)] if len(configs) > 1 else [eps]
    for block in blocks:
        for config in configs:
            backend = make_backend(model, config)
            try:
                warm(backend)
                if config not in mem:
                    mem[config] = ({"note": "vLLM runs in WSL: 0.75 of 16 GB by "
                                    "--gpu-memory-utilization, not a per-process figure"}
                                   if config.startswith("vllm") else
                                   vram(common.OLLAMA if config == "default" else None))
                names[config] = backend
                rows[config] += [run_one(backend, model, e) for e in block]
                print(f"  {model} {config}: {len(rows[config])}/{len(eps)} "
                      f"{sum(r['correct'] for r in rows[config])} right", flush=True)
            finally:
                backend.close()
    out = {}
    for config in configs:
        b = names[config]
        clear(conn, args.run_group, b.model, config)
        store(conn, args.run_group, b, config, rows[config])
        wall = sum(r["wall_ms"] for r in rows[config]) / 1000
        out[config] = {**summarize(rows[config], wall), "concurrency": 1, "vram": mem[config]}
    return out


def fan(backend, model, eps, n):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(n) as ex:
        rows = list(ex.map(lambda e: run_one(backend, model, e), eps))
    return rows, time.perf_counter() - t0


def fanout_unit(conn, args, model, n, eps):
    """2n episodes at once on a side Ollama with NUM_PARALLEL n."""
    config = f"np{n}"
    p9.evict_ollama()
    tag = re.sub(r"[^\w.-]", "_", model)
    with p9.OllamaSide(SIDE_PORT, num_parallel=n, label=f"p10-cost-{tag}-np{n}") as side:
        backend = make_backend(model, "default", base=side.base)
        try:
            warm(backend)
            mem = vram(side.base)
            cmd = side.runner_cmd() or ""
            m = re.search(r" -np (\d+)", cmd)
            mem["runner_np"] = int(m.group(1)) if m else None
            # Ollama bundles a draft model with some (nemotron-3.5-bolt), and
            # Phase 9 found drafting loses once requests are batched.
            mem["ollama_draft"] = "--spec-draft-model" in cmd
            # /api/ps sizes are not reliable here (1.1 GB for a 7.9 GB model),
            # so "on the GPU" is llama-server's own layer count, model and draft.
            with open(side.log_path, encoding="utf-8", errors="replace") as f:
                layers = re.findall(r"offloaded (\d+)/(\d+) layers", f.read())
            mem["offloaded"] = ["/".join(x) for x in layers]
            on_gpu = bool(layers) and all(a == b for a, b in layers)
            if not on_gpu or mem["shared_mb"] > SHARED_LIMIT_MB:
                print(f"  {model} {config}: does not fit on the GPU {mem}", flush=True)
                return {config: {"skipped": "does not fit on the GPU", "concurrency": n,
                                 "vram": mem}}
            rows, wall = fan(backend, model, eps, n)
        finally:
            backend.close()
    clear(conn, args.run_group, model, config)
    store(conn, args.run_group, backend, config, rows)
    print(f"  {model} {config}: {len(rows)} episodes in {wall:.0f} s, "
          f"{sum(r['correct'] for r in rows)} right", flush=True)
    return {config: {**summarize(rows, wall), "concurrency": n, "vram": mem}}


def vllm_unit(conn, args, model, items, eps, fanouts):
    """The vLLM model on one server: sequential, then each fan-out level."""
    backend = make_backend(model, "vllm-tuned")
    out = {}
    try:
        warm(backend)
        mem = {"note": "vLLM runs in WSL: 0.75 of 16 GB by --gpu-memory-utilization"}
        t0 = time.perf_counter()
        rows = [run_one(backend, model, e) for e in eps]
        levels = [("vllm-tuned", 1, rows, time.perf_counter() - t0)]
        for n in fanouts:
            rows, wall = fan(backend, model, items[:2 * n], n)
            levels.append((f"vllm-c{n}", n, rows, wall))
        for config, n, rows, wall in levels:
            clear(conn, args.run_group, backend.model, config)
            store(conn, args.run_group, backend, config, rows)
            out[config] = {**summarize(rows, wall), "concurrency": n, "vram": mem}
            print(f"  {model} {config}: {len(rows)} episodes in {wall:.0f} s, "
                  f"{sum(r['correct'] for r in rows)} right", flush=True)
    finally:
        backend.close()
    return out


def mailbox_items():
    return q.load_suite("D")["items"]


# --- report --------------------------------------------------------------------

def report(path):
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    print(f"{'model':24} {'config':10} {'n':>3} {'pass':>6} {'s/ep':>6} {'s/turn':>6} "
          f"{'prefill':>7} {'VRAM GB':>7} {'shared':>6} {'ep/min':>6} {'right/h':>7}")
    for model, units in res["models"].items():
        for config, u in units.items():
            v = u.get("vram", {})
            ded = f"{v['dedicated_mb'] / 1024:.1f}" if "dedicated_mb" in v else "-"
            sh = f"{v['shared_mb'] / 1024:.1f}" if "shared_mb" in v else "-"
            if "skipped" in u:
                print(f"{model:24} {config:10} {'':>3} {'':>6} {'':>6} {'':>6} {'':>7} "
                      f"{ded:>7} {sh:>6}  {u['skipped']}")
                continue
            pre = f"{u['prefill_share']:.0%}" if u.get("prefill_share") is not None else "-"
            acc = u["correct"] / u["n"]
            print(f"{model:24} {config:10} {u['n']:3} {acc:6.0%} {u['s_per_episode']:6.1f} "
                  f"{u['s_per_turn']:6.2f} {pre:>7} {ded:>7} {sh:>6} "
                  f"{u['episodes_per_min']:6.1f} {acc * u['episodes_per_min'] * 60:7.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=SHORTLIST)
    ap.add_argument("--episodes", type=int, default=20, help="sequential episodes per config")
    ap.add_argument("--fanout", default="4,16", help="concurrency levels, comma-separated")
    ap.add_argument("--run-group", default="p10-cost")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    path = os.path.join(common.LOG_DIR, args.run_group + ".json")
    if args.report:
        report(path)
        return

    res = {"models": {}}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
    res["ollama_version"] = common.ollama_version()
    conn = common.storage.connect()
    items = mailbox_items()
    eps = items[:args.episodes]
    fanouts = [int(x) for x in args.fanout.split(",") if x]

    def save():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    for model in args.models:
        print(model, flush=True)
        done = res["models"].setdefault(model, {})
        if model in VLLM_MODEL:
            if "vllm-tuned" not in done:
                p9.evict_ollama()
                done.update(vllm_unit(conn, args, model, items, eps, fanouts))
                save()
            continue
        if not all(c in done for c in SEQ_CONFIGS.get(model, ["default"])):
            done.update(seq_unit(conn, args, model, eps))
            save()
        for n in fanouts:
            if f"np{n}" not in done:
                done.update(fanout_unit(conn, args, model, n, items[:2 * n]))
                save()
    report(path)


if __name__ == "__main__":
    main()

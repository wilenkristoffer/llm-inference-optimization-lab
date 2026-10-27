"""
Phase 10, Stage 2 follow-ups: why fan-out on Ollama bought nothing.

03_cost.py found three things it could not explain:

    gemma4      at NUM_PARALLEL 4, 4 of 8 episodes flip to wrong (it calls
                "mail-get_unread" and loops). Check: the same 8 episodes on a
                side Ollama with NUM_PARALLEL 4, sent one at a time. Broken
                again -> the 4-slot setup; fine -> batched decode.
    qwen3.5     4.8 / 4.5 / 4.3 episodes per minute at 1 / 4 / 16. Check:
                plain decode throughput at 1 vs 4 requests at once for the
                BF16 model, a Q8_0 copy of it (llama-quantize, logs/), and the
                dense qwen2.5:3b as a control. Q8 flat too -> the hybrid
                (linear-attention) architecture, not BF16.
    nemotron    flat to worse; Ollama ships it with its own draft model. Check:
                the same decode test with and without that draft.

The decode tests run Ollama's own llama-server.exe with the flags Ollama gave
it (-b/-ub 1024, flash attention auto, 8192 per slot), on /completion with a
raw prompt so no template or tool parser is involved.

Usage:
    python cost_checks.py              # all three
    python cost_checks.py --only decode
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import common

cost = common._load("p10cost", "03_cost.py")
p9 = common.p9

BLOB = os.path.expanduser(r"~\.ollama\models\blobs")
DECODE = {
    "qwen2.5:3b": [os.path.join(BLOB, p9.GGUF["qwen2.5:3b"])],
    "qwen3.5:4b-bf16": [os.path.join(BLOB, "sha256-bb5b8b1308515427ca45544a46ec828f7a5472a505f2e8c9e60f26ce665a4a4d")],
    "qwen3.5:4b-q8_0": [os.path.join(common.LOG_DIR, "qwen35-4b-q8_0.gguf")],
    "nemotron no draft": [os.path.join(BLOB, "sha256-c8af2b8aea10a690a1d9491db744a9446660173cb38fc489612b556615a69612")],
    "nemotron + draft": [os.path.join(BLOB, "sha256-c8af2b8aea10a690a1d9491db744a9446660173cb38fc489612b556615a69612"),
                         "--spec-draft-model",
                         os.path.join(BLOB, "sha256-8103aa42fd31fb661c3fbe8701d04d698ea2c5d09351dc3e8c9da639958de989"),
                         "--spec-draft-n-max", "7"],
}
SLOTS = 4
N_PREDICT = 192
PROMPT = ("Write a detailed guide for a support team on how to triage a shared email "
          "inbox: how to categorize messages, when to reply, when to escalate, and "
          "when to archive. Use numbered sections.\n\n1.")


def serve(model_args, log_path):
    args = [os.path.join(p9.OLLAMA_LIB, "llama-server.exe"), "-m", *model_args,
            "-ngl", "99", "--flash-attn", "auto", "-b", "1024", "-ub", "1024",
            "-c", str(8192 * SLOTS), "-np", str(SLOTS),
            "--host", "127.0.0.1", "--port", str(p9.LLAMA_PORT), "--no-webui"]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(args, env=p9._llama_env(), stdout=log, stderr=subprocess.STDOUT,
                            creationflags=p9.NO_WINDOW)
    p9._wait_healthy(p9.LLAMA_URL, proc, 180, log_path)
    return proc, log


def completion():
    body = {"prompt": PROMPT, "n_predict": N_PREDICT, "temperature": 0, "seed": 42,
            "ignore_eos": True, "cache_prompt": False}
    t = p9._post(p9.LLAMA_URL + "/completion", body)["timings"]
    return {"tokens": t["predicted_n"], "decode_ms": t["predicted_ms"],
            "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted")}


def at_once(n, rounds):
    """n requests at a time, `rounds` times: aggregate and per-request decode."""
    walls, per_req, total = [], [], 0
    for _ in range(rounds):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(n) as ex:
            rs = list(ex.map(lambda _: completion(), range(n)))
        walls.append(time.perf_counter() - t0)
        total += sum(r["tokens"] for r in rs)
        per_req += [r["tokens"] / r["decode_ms"] * 1000 for r in rs]
        last = rs
    out = {"aggregate_tok_s": total / sum(walls),
           "per_request_tok_s": sum(per_req) / len(per_req)}
    if last[0]["draft_n"]:
        out["draft_accept"] = (sum(r["draft_accepted"] for r in last) /
                               sum(r["draft_n"] for r in last))
    return out


def decode_check():
    p9.evict_ollama()
    res = {}
    for name, model_args in DECODE.items():
        log_path = os.path.join(common.LOG_DIR, "check-" + name.replace(":", "_").replace(" ", "_") + ".log")
        proc, log = serve(model_args, log_path)
        try:
            completion()                      # warmup
            r1, r4 = at_once(1, 3), at_once(SLOTS, 2)
        finally:
            proc.kill()
            proc.wait(timeout=10)
            log.close()
        res[name] = {"c1": r1, "c4": r4, "scaling": r4["aggregate_tok_s"] / r1["aggregate_tok_s"]}
        print(f"  {name:20} c1 {r1['aggregate_tok_s']:6.1f} tok/s   c4 {r4['aggregate_tok_s']:6.1f} "
              f"tok/s aggregate ({r4['per_request_tok_s']:5.1f} each)   x{res[name]['scaling']:.2f}"
              + (f"   draft accept c1 {r1['draft_accept']:.0%} c4 {r4['draft_accept']:.0%}"
                 if "draft_accept" in r1 else ""), flush=True)
    return res


def gemma_check(conn):
    """The 8 episodes that broke at np4, at np4 but one at a time."""
    model = "gemma4:12b"
    eps = cost.mailbox_items()[:8]
    p9.evict_ollama()
    with p9.OllamaSide(cost.SIDE_PORT, num_parallel=SLOTS, label="p10-check-gemma4-np4-seq") as side:
        backend = cost.make_backend(model, "default", base=side.base)
        try:
            cost.warm(backend)
            rows = [cost.run_one(backend, model, e) for e in eps]
        finally:
            backend.close()
    cost.clear(conn, "p10-cost-checks", model, "np4-seq")
    cost.store(conn, "p10-cost-checks", backend, "np4-seq", rows)
    bad = sum("get_unread" in json.dumps(r["transcript"]) for r in rows)
    out = {"n": len(rows), "correct": sum(r["correct"] for r in rows),
           "episodes_with_get_unread": bad,
           "per_episode": {r["id"]: r["correct"] for r in rows}}
    print(f"  gemma4 np4 one-at-a-time: {out['correct']}/8 right, "
          f"{bad} episodes with the bad tool name", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["decode", "gemma"])
    args = ap.parse_args()
    path = os.path.join(common.LOG_DIR, "p10-cost-checks.json")
    res = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
    if args.only != "decode":
        res["gemma4_np4_sequential"] = gemma_check(common.storage.connect())
    if args.only != "gemma":
        res["decode"] = decode_check()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

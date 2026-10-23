"""
Phase 9, file 1: speculative decoding in llama.cpp, draft qwen2.5:0.5b.

For each target (14b, 7b, 3b, all Q4_K_M) one server per draft length K,
since llama-server fixes K at start. K = 0 is the same server with no draft,
run first AND last so drift within the session is visible (Phase 1: ~4%
between sessions, so nothing is compared across them). Every config runs all
nine workload prompts, twice.

What each row carries, besides tok/s: the backend's draft counters, so
acceptance (accepted / drafted) and tokens per target pass (output / passes)
come straight from llama.cpp, and the VRAM the server process holds.

Why the cost model matters. The textbook speedup is

    (1 - a^(K+1)) / (1 - a)  /  (1 + K*c)

with c = draft step / target step. Stage 0 measured 1.34x on the 14b where
this predicts ~2.2x, so --report also derives where the time actually goes:
the draft's step time from the 0.5b run alone, and from that the verify pass
cost per spec row, (decode_ms - drafted * t_draft) / passes, as a multiple of
one plain target step.

Losslessness: greedy here is not bit-deterministic even without a draft
(Stage 0: plain 3b gave 2 distinct texts of 4). So a spec text is judged by
where it first diverges from the plain run, against where the plain runs
diverge from each other.

Usage:
    python 01_spec_llamacpp.py                       # full sweep, ~30 min
    python 01_spec_llamacpp.py --targets qwen2.5:3b --ks 0,4
    python 01_spec_llamacpp.py --report [--group p9-spec-llamacpp-...]
"""

import argparse
import statistics
from collections import defaultdict
from datetime import datetime

import common

DRAFT = "qwen2.5:0.5b"
TARGETS = ["qwen2.5:14b", "qwen2.5:7b", "qwen2.5:3b"]
KS = [0, 1, 2, 3, 4, 6, 8]
REPS = 2


BATCH_NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16]


def batch_cost(conn, group, version, target, reps=8):
    """One forward pass over n new tokens on top of a cached 300-token prefix -
    exactly what a verify pass of K = n - 1 drafts does. Measured with prompt
    processing: the prefix stays cached and n random tokens are appended, so
    llama.cpp's prompt_ms is the pass alone.

    Found while building this file: cost rises ~linearly to n = 8 and DROPS at
    9 (14b: 47.5 -> 36.1 ms). Up to 8 columns llama.cpp uses a vector kernel
    that does per-token work; from 9 a matrix kernel. Verify is not ~free here.
    """
    import random
    toks = common._post(common.LLAMA_URL + "/tokenize", {"content": "\n".join(
        p for _, _, p in common.workloads.PROMPTS)})["tokens"]
    prefix, pool = toks[:300], toks[300:]
    rng = random.Random(0)
    common._post(common.LLAMA_URL + "/completion",
                 {"prompt": prefix, "n_predict": 1, "cache_prompt": True})
    for n in BATCH_NS:
        for rep in range(reps):
            r = common._post(common.LLAMA_URL + "/completion",
                             {"prompt": prefix + rng.sample(pool, n), "n_predict": 1,
                              "cache_prompt": True, "temperature": 0})
            t = r["timings"]
            if rep < 2:                 # first passes at a new size warm up
                continue
            common.storage.record_run(
                conn, run_group=group, trial_index=n * 100 + rep, backend="llama.cpp",
                backend_version=version, model=target, quantization="Q4_K_M",
                context_length=4096, input_tokens=t["prompt_n"],
                cached_input_tokens=t["cache_n"], prefill_ms=t["prompt_ms"],
                hardware_id=common.HARDWARE_ID, notes=f"prompt=batch phase=batch-cost")
        print(f"  {target} pass over {n} tokens: {t['prompt_ms']:.2f} ms", flush=True)


def run_config(conn, group, version, target, k, phase, reps):
    draft = DRAFT if k else None
    with common.LlamaServer(target, draft=draft, k=k or None) as s:
        vram = common.process_vram_mb(s.proc.pid)
        s.chat("Say hello.", max_tokens=8)          # warm-up, not stored
        if phase == "start":
            batch_cost(conn, group, version, target)
        for rep in range(reps):
            for i, (workload, pid, prompt) in enumerate(common.workloads.PROMPTS):
                r = s.chat(prompt)
                common.storage.record_run(
                    conn, run_group=group, trial_index=rep * 100 + i,
                    backend="llama.cpp", backend_version=version, model=target,
                    quantization="Q4_K_M", context_length=4096, temperature=0,
                    seed=42, streaming=0, prompt=prompt, response=r["response"],
                    input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
                    prefill_ms=r["prefill_ms"], decode_ms=r["decode_ms"],
                    client_total_ms=r["client_total_ms"],
                    decode_tokens_per_s=r["decode_tokens_per_s"],
                    spec_method="draft" if k else None, spec_k=k,
                    draft_model=draft, draft_tokens=r["draft_tokens"],
                    draft_accepted=r["draft_accepted"],
                    target_passes=r["target_passes"], workload=workload,
                    server_vram_mb=vram, server_num_parallel=1,
                    gpu_library="ROCm", hardware_id=common.HARDWARE_ID,
                    notes=f"prompt={pid} phase={phase}")
                acc = (f" acc {r['draft_accepted']}/{r['draft_tokens']}"
                       if r["draft_tokens"] else "")
                print(f"  {target} K={k} {phase} rep{rep} {workload}/{pid}: "
                      f"{r['decode_tokens_per_s']:.1f} tok/s n={r['output_tokens']}{acc}",
                      flush=True)


def sweep(targets, ks, reps):
    conn = common.storage.connect()
    group = "p9-spec-llamacpp-" + datetime.now().strftime("%Y%m%d-%H%M")
    version = common.llama_version()
    print(f"run_group {group}, {version}")
    # The draft alone: its step time is the c in the cost model.
    run_config(conn, group, version, DRAFT, 0, "draft-alone", 1)
    for target in targets:
        plain_last = 0 in ks
        for k in ks:
            run_config(conn, group, version, target, k, "start" if k == 0 else "spec", reps)
        if plain_last:
            run_config(conn, group, version, target, 0, "end", reps)
    print(f"done: {group}")


def report(group):
    conn = common.storage.connect()
    if not group:
        group = conn.execute("SELECT run_group FROM inference_runs WHERE run_group "
                             "LIKE 'p9-spec-llamacpp-%' ORDER BY id DESC LIMIT 1").fetchone()[0]
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM inference_runs WHERE run_group = ? ORDER BY id", (group,))]
    print(f"run_group {group}: {len(rows)} rows\n")

    def phase(r):
        return r["notes"].split("phase=")[1]

    def prompt_id(r):
        return r["notes"].split()[0].split("=")[1]

    draft_rows = [r for r in rows if phase(r) == "draft-alone"]
    t_draft = statistics.median(r["decode_ms"] / r["output_tokens"] for r in draft_rows)
    print(f"draft {DRAFT} alone: {1000 / t_draft:.0f} tok/s, {t_draft:.2f} ms/token\n")

    for target in [t for t in TARGETS if any(r["model"] == t for r in rows)]:
        cost = defaultdict(list)
        for r in rows:
            if r["model"] == target and phase(r) == "batch-cost":
                cost[r["input_tokens"]].append(r["prefill_ms"])
        verify_ms = {n: statistics.median(v) for n, v in cost.items()}
        mine = [r for r in rows if r["model"] == target and phase(r) != "batch-cost"]
        plain = [r for r in mine if not r["spec_k"]]
        t_plain = statistics.median(r["decode_ms"] / r["output_tokens"] for r in plain)
        c = t_draft / t_plain
        start = [r["decode_tokens_per_s"] for r in plain if phase(r) == "start"]
        end = [r["decode_tokens_per_s"] for r in plain if phase(r) == "end"]
        drift = (f", drift start->end {statistics.median(end) / statistics.median(start) - 1:+.1%}"
                 if start and end else "")
        vram0 = plain[0]["server_vram_mb"]
        print(f"== {target}: plain {1000 / t_plain:.1f} tok/s ({t_plain:.2f} ms/step), "
              f"c = {c:.3f}{drift}, VRAM {vram0:.0f} MB")
        if verify_ms:
            print("  one pass over n tokens, ms: " + "  ".join(
                f"{n}:{verify_ms[n]:.1f}" for n in sorted(verify_ms)))
        print("  per pass, ms: verify = measured pass over K+1 tokens; draft = drafted"
              " tokens x draft step; other = the rest (draft catch-up, sync, sampling)")

        # Plain reference text per prompt = first plain run of that prompt.
        ref = {}
        for r in plain:
            ref.setdefault(prompt_id(r), (r["id"], r["response"]))

        base = defaultdict(list)
        for r in plain:
            base[r["workload"]].append(r["decode_tokens_per_s"])
        print(f"  {'K':>2} {'workload':<11} {'tok/s':>7} {'speedup':>8} {'accept':>7} "
              f"{'tok/pass':>8} {'verify':>7} {'draft':>6} {'other':>6} {'formula':>8} "
              f"{'diverged':>9} {'VRAM+':>6}")
        for k in sorted({r["spec_k"] for r in mine}):
            for wl in ("code", "extraction", "prose"):
                g = [r for r in mine if r["spec_k"] == k and r["workload"] == wl]
                if not g:
                    continue
                tps = statistics.median(r["decode_tokens_per_s"] for r in g)
                speed = tps / statistics.median(base[wl])
                div = [common.divergence(ref[prompt_id(r)][1], r["response"]) for r in g
                       if r["id"] != ref[prompt_id(r)][0]]
                ndiv = sum(d is not None for d in div)
                if k:
                    a = sum(r["draft_accepted"] for r in g) / sum(r["draft_tokens"] for r in g)
                    tpp = sum(r["output_tokens"] for r in g) / sum(r["target_passes"] for r in g)
                    passes = sum(r["target_passes"] for r in g)
                    per_pass = sum(r["decode_ms"] for r in g) / passes
                    # A pass verifies however many tokens were drafted for it,
                    # up to K; K+1 is the upper bound and what the kernel sees
                    # when the draft runs full length.
                    verify = verify_ms.get(k + 1, float("nan"))
                    draft = sum(r["draft_tokens"] for r in g) / passes * t_draft
                    other = per_pass - verify - draft
                    # Per-token acceptance a is what the formula wants;
                    # llama.cpp's ratio is per drafted token, close enough.
                    formula = (1 - a ** (k + 1)) / (1 - a) / (1 + k * c) if a < 1 else (k + 1) / (1 + k * c)
                    vplus = g[0]["server_vram_mb"] - vram0
                    print(f"  {k:>2} {wl:<11} {tps:7.1f} {speed:7.2f}x {a:7.2f} {tpp:8.2f} "
                          f"{verify:7.1f} {draft:6.1f} {other:6.1f} {formula:7.2f}x "
                          f"{ndiv:>4}/{len(div):<4} {vplus:5.0f}")
                else:
                    print(f"  {k:>2} {wl:<11} {tps:7.1f} {speed:7.2f}x {'':>7} {'':>8} "
                          f"{'':>7} {'':>6} {'':>6} {'':>8} {ndiv:>4}/{len(div):<4}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--group")
    a = ap.parse_args()
    if a.report:
        report(a.group)
    else:
        sweep(a.targets.split(","), [int(k) for k in a.ks.split(",")], a.reps)


if __name__ == "__main__":
    main()

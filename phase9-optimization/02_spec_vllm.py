"""
Phase 9, file 2: speculative decoding in vLLM, under load as well as alone.

Target Qwen2.5-3B-Instruct (FP16), draft Qwen2.5-0.5B-Instruct. One server per
config, since the speculative config is fixed at start:

    3B plain                       first AND last, for drift within the session
    3B + draft_model, K = 1..6, 8
    3B + ngram (prompt lookup 2-4 tokens, K = 4)
    1.5B plain, 1.5B + ngram       Stage 0's 0.90x, now per workload

Every config runs the nine workload prompts single-stream (REPS times), then
a concurrency sweep at 1, 4, 16, 32 simultaneous requests. That second part is
the question Stage 1 is really about: speculation spends spare compute to cut
latency, and batching uses up the spare compute. At some concurrency the
verify work stops being free and speculation should turn into a loss.

vLLM's API carries no per-request draft counters, so single-stream rows get
the /metrics delta around each request (exact, as only one is in flight).
Concurrent rows carry the delta over the whole batch in notes instead.

Usage (from Windows; vLLM runs in WSL, only this side writes lab.db):
    python 02_spec_vllm.py                       # full sweep, ~40 min
    python 02_spec_vllm.py --configs 3b,3b-k3 --reps 1 --conc 1,4
    python 02_spec_vllm.py --report [--group p9-spec-vllm-...]
"""

import argparse
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import common

DRAFT = common.HF["qwen2.5-0.5b-instruct"]
T3, T15 = "qwen2.5-3b-instruct", "qwen2.5-1.5b-instruct"
NGRAM = {"method": "ngram", "num_speculative_tokens": 4,
         "prompt_lookup_max": 4, "prompt_lookup_min": 2}

# name -> (target, speculative config or None)
CONFIGS = {"3b": (T3, None)}
for _k in (1, 2, 3, 4, 5, 6, 8):
    CONFIGS[f"3b-k{_k}"] = (T3, {"method": "draft_model", "model": DRAFT,
                                 "num_speculative_tokens": _k})
CONFIGS["3b-ngram"] = (T3, NGRAM)
CONFIGS["1.5b"] = (T15, None)
CONFIGS["1.5b-ngram"] = (T15, NGRAM)
CONFIGS["3b-end"] = (T3, None)

REPS = 2
CONCURRENCY = [1, 4, 16, 32]


def _row(r, **extra):
    return dict(
        backend="vLLM", quantization="FP16", context_length=4096, temperature=0,
        seed=42, streaming=1, response=r["response"], input_tokens=r["input_tokens"],
        output_tokens=r["output_tokens"], ttft_ms=r["ttft_ms"], decode_ms=r["decode_ms"],
        client_total_ms=r["client_total_ms"], decode_tokens_per_s=r["decode_tokens_per_s"],
        gpu_library="ROCm", hardware_id=common.HARDWARE_ID, **extra)


def run_config(conn, group, version, name, reps, concurrency):
    target, spec = CONFIGS[name]
    k = spec["num_speculative_tokens"] if spec else None
    meta = dict(run_group=group, backend_version=version, model=target,
                spec_method=spec["method"] if spec else None, spec_k=k,
                draft_model=spec.get("model") if spec else None)
    with common.VllmServer(target, spec=spec) as s:
        print(f"{name}: up in {s.startup_s:.0f} s", flush=True)
        s.chat("Say hello.", max_tokens=8)              # warm-up, not stored
        for rep in range(reps):
            for i, (workload, pid, prompt) in enumerate(common.workloads.PROMPTS):
                before = s.counters()
                r = s.chat(prompt)
                after = s.counters()
                drafted = int(after["draft_tokens"] - before["draft_tokens"])
                accepted = int(after["accepted"] - before["accepted"])
                common.storage.record_run(conn, **meta, **_row(
                    r, trial_index=rep * 100 + i, prompt=prompt, workload=workload,
                    draft_tokens=drafted if spec else None,
                    draft_accepted=accepted if spec else None,
                    target_passes=r["output_tokens"] - accepted if spec else None,
                    load_ms=s.startup_s * 1000, concurrency=1,
                    notes=f"prompt={pid} phase={name}"))
                acc = f" acc {accepted}/{drafted}" if spec else ""
                print(f"  {name} rep{rep} {workload}/{pid}: "
                      f"{r['decode_tokens_per_s']:.1f} tok/s n={r['output_tokens']}{acc}",
                      flush=True)

        prompts = common.workloads.PROMPTS
        for c in concurrency:
            # The same prompt mix at every level: request j gets prompt j mod 9.
            batch = [prompts[j % len(prompts)] for j in range(c)]
            before = s.counters()
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as ex:
                results = list(ex.map(lambda p: s.chat(p[2]), batch))
            wall = (time.perf_counter() - t0) * 1000
            after = s.counters()
            drafted = int(after["draft_tokens"] - before["draft_tokens"])
            accepted = int(after["accepted"] - before["accepted"])
            for j, ((workload, pid, prompt), r) in enumerate(zip(batch, results)):
                common.storage.record_run(conn, **meta, **_row(
                    r, trial_index=10000 + c * 100 + j, prompt=prompt, workload=workload,
                    concurrency=c, batch_wall_ms=wall, load_ms=s.startup_s * 1000,
                    notes=(f"prompt={pid} phase={name}-conc "
                           f"batch_drafted={drafted} batch_accepted={accepted}")))
            total = sum(r["output_tokens"] for r in results)
            print(f"  {name} concurrency {c}: {total / wall * 1000:.0f} tok/s aggregate, "
                  f"{statistics.median(r['decode_tokens_per_s'] for r in results):.1f} "
                  f"per stream", flush=True)


def sweep(names, reps, concurrency):
    conn = common.storage.connect()
    group = "p9-spec-vllm-" + datetime.now().strftime("%Y%m%d-%H%M")
    version = common.vllm_version()
    print(f"run_group {group}, {version}")
    for name in names:
        run_config(conn, group, version, name, reps, concurrency)
    print(f"done: {group}")


def report(group):
    conn = common.storage.connect()
    if not group:
        group = conn.execute("SELECT run_group FROM inference_runs WHERE run_group "
                             "LIKE 'p9-spec-vllm-%' ORDER BY id DESC LIMIT 1").fetchone()[0]
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM inference_runs WHERE run_group = ? ORDER BY id", (group,))]
    print(f"run_group {group}: {len(rows)} rows\n")

    def name(r):
        return r["notes"].split("phase=")[1].split()[0]

    def prompt_id(r):
        return r["notes"].split()[0].split("=")[1]

    single = [r for r in rows if not name(r).endswith("-conc")]
    conc = [r for r in rows if name(r).endswith("-conc")]
    names = list(dict.fromkeys(name(r) for r in single))

    for target, plain_names in ((T3, ("3b", "3b-end")), (T15, ("1.5b",))):
        plain = [r for r in single if name(r) in plain_names]
        if not plain:
            continue
        base = defaultdict(list)
        ref = {}
        for r in plain:
            base[r["workload"]].append(r["decode_tokens_per_s"])
            ref.setdefault(prompt_id(r), (r["id"], r["response"]))
        first = [r["decode_tokens_per_s"] for r in plain if name(r) == plain_names[0]]
        last = [r["decode_tokens_per_s"] for r in plain if name(r) == plain_names[-1]]
        drift = (f", drift start->end {statistics.median(last) / statistics.median(first) - 1:+.1%}"
                 if len(plain_names) > 1 and last else "")
        print(f"== {target} single stream{drift}")
        print(f"  {'config':<11} {'workload':<11} {'tok/s':>7} {'speedup':>8} {'accept':>7} "
              f"{'tok/pass':>8} {'diverged':>9} {'startup':>8}")
        for n in names:
            if CONFIGS[n][0] != target:
                continue
            for wl in ("code", "extraction", "prose"):
                g = [r for r in single if name(r) == n and r["workload"] == wl]
                if not g:
                    continue
                tps = statistics.median(r["decode_tokens_per_s"] for r in g)
                speed = tps / statistics.median(base[wl])
                div = [common.divergence(ref[prompt_id(r)][1], r["response"]) for r in g
                       if r["id"] != ref[prompt_id(r)][0]]
                ndiv = sum(d is not None for d in div)
                if g[0]["spec_k"]:
                    a = sum(r["draft_accepted"] for r in g) / sum(r["draft_tokens"] for r in g)
                    tpp = sum(r["output_tokens"] for r in g) / sum(r["target_passes"] for r in g)
                    spec = f"{a:7.2f} {tpp:8.2f}"
                else:
                    spec = f"{'':>7} {'':>8}"
                print(f"  {n:<11} {wl:<11} {tps:7.1f} {speed:7.2f}x {spec} "
                      f"{ndiv:>4}/{len(div):<4} {g[0]['load_ms'] / 1000:7.0f}s")
        print()

    if conc:
        print("== concurrency: aggregate tok/s (per-stream median tok/s), acceptance")
        levels = sorted({r["concurrency"] for r in conc})
        print(f"  {'config':<11} " + " ".join(f"{'c=' + str(c):>22}" for c in levels))
        agg = {}
        for n in names:
            cells = []
            for c in levels:
                g = [r for r in conc if name(r) == n + "-conc" and r["concurrency"] == c]
                if not g:
                    cells.append(f"{'':>22}")
                    continue
                total = sum(r["output_tokens"] for r in g) / g[0]["batch_wall_ms"] * 1000
                agg[n, c] = total
                per = statistics.median(r["decode_tokens_per_s"] for r in g)
                notes = dict(kv.split("=") for kv in g[0]["notes"].split()[2:])
                d = int(notes["batch_drafted"])
                acc = f"{int(notes['batch_accepted']) / d:.2f}" if d else "   -"
                cells.append(f"{total:7.0f} ({per:5.1f}) {acc:>5}")
            print(f"  {n:<11} " + " ".join(cells))
        print("\n  aggregate vs the plain server at the same concurrency:")
        for n in names:
            plain_n = "1.5b" if CONFIGS[n][0] == T15 else "3b"
            if n in ("3b", "3b-end", "1.5b"):
                continue
            cells = [f"{agg[n, c] / agg[plain_n, c]:6.2f}x" if (n, c) in agg and (plain_n, c) in agg
                     else f"{'':>7}" for c in levels]
            print(f"  {n:<11} " + " ".join(f"{x:>22}" for x in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--conc", default=",".join(map(str, CONCURRENCY)))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--group")
    a = ap.parse_args()
    if a.report:
        report(a.group)
    else:
        sweep(a.configs.split(","), a.reps,
              [int(c) for c in a.conc.split(",") if c])


if __name__ == "__main__":
    main()

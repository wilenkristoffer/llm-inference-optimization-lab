"""
Phase 9, file 8: concurrency and batching settings (Stage 4).

Two parts, one backend on the card at a time (each wants all of it):

    sweep   the nine workload prompts at 1, 4, 8, 16, 32 simultaneous
            requests (request j gets prompt j mod 9, 256 tokens, as in
            Phase 5 and Stage 1), REPS batches per level:
              ollama-np8/16/32  side Ollama, NUM_PARALLEL 8/16/32, qwen2.5:1.5b
              vllm              vLLM 1.5B FP16, default GEMMs
              vllm-tuned        the same with the TunableOp table (07_tune_gemms.py)
              vllm-seqs8/16     vllm-tuned with --max-num-seqs 8/16, at 32 only:
                                what capping the batch does to per-request latency
    mixed   inter-token latency under mixed load: 8 streams decoding long
            answers, then one ~4,000-token prompt arrives. How long do the
            8 streams stall while it is prefilled?
              ollama-np9        NUM_PARALLEL 9, so the long prompt gets its own slot
              vllm-mbt512/2048/8192
                                vllm-tuned with --max-num-batched-tokens
                                512 / 2048 (vLLM's default) / 8192

Per steady stream the mixed part records: base ITL (median gap before the long
prompt arrived), max gap and excess (sum over gaps of gap - base ITL) inside the
window from its arrival to its first token + 100 ms. Raw chunk times go to
logs/<group>-mixed.json.

Usage:
    python 08_concurrency.py --part sweep    # ~25 min
    python 08_concurrency.py --part mixed    # ~10 min
    python 08_concurrency.py --part sweep --backends vllm --conc 1,4 --reps 1
    python 08_concurrency.py --report [--group p9-conc-...]
"""

import argparse
import json
import os
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import common

OLLAMA_MODEL, VLLM_MODEL = "qwen2.5:1.5b", "qwen2.5-1.5b-instruct"
OLLAMA_PORT = 11437
REPS = 3
CONCURRENCY = [1, 4, 8, 16, 32]
SWEEP_CTX = 4096
MIXED_CTX = 8192
STEADY = 8
STEADY_TOKENS = 512
LONG_CHARS = 14000        # ~4,000 tokens of the docs corpus; rows record the real count
LONG_TOKENS = 16
MIXED_REPS = 3

# name -> (kind, server settings). kind "ollama": num_parallel; "vllm": tuned, extra flags
SWEEP = {
    "vllm": ("vllm", False, []),
    "ollama-np8": ("ollama", 8),
    "ollama-np16": ("ollama", 16),
    "ollama-np32": ("ollama", 32),
    "vllm-tuned": ("vllm", True, []),
    "vllm-seqs8": ("vllm", True, ["--max-num-seqs", "8"]),
    "vllm-seqs16": ("vllm", True, ["--max-num-seqs", "16"]),
}
SEQS_ONLY_AT = [32]
MIXED = {
    "ollama-np9": ("ollama", 9),
    "vllm-mbt512": ("vllm", True, ["--max-num-batched-tokens", "512"]),
    "vllm-mbt2048": ("vllm", True, ["--max-num-batched-tokens", "2048"]),
    "vllm-mbt8192": ("vllm", True, ["--max-num-batched-tokens", "8192"]),
}

TOPICS = ["a lighthouse keeper who finds a message in a bottle",
          "the first train to cross a desert", "a chef who loses their sense of taste",
          "two rival clockmakers in an old city", "a mapmaker charting an unknown island",
          "a beekeeper during a long drought", "a radio operator on a polar station",
          "a violin passed down through four generations"]


class Backend:
    """One server and a client for it: ask(messages, max_tokens, times)."""

    def __init__(self, name, spec, ctx):
        self.name, self.kind, self.ctx = name, spec[0], ctx
        if self.kind == "ollama":
            self.num_parallel = spec[1]
            self.server = common.OllamaSide(OLLAMA_PORT, spec[1], label=name)
        else:
            self.tuned, self.num_parallel = spec[1], None
            self.server = common.VllmServer(
                VLLM_MODEL, gpu_fraction=0.6, max_len=ctx, extra=spec[2],
                env=common.TUNABLEOP_ENV if self.tuned else None, label=name)

    def __enter__(self):
        if self.kind == "ollama":
            common.evict_ollama()
        self.server.__enter__()
        self.ask([{"role": "user", "content": "Say hello."}], 8)     # load / warm
        if self.kind == "ollama":
            print(f"{self.name}: {self.server.runner_cmd()}", flush=True)
        else:
            print(f"{self.name}: up in {self.server.startup_s:.0f} s", flush=True)
        return self

    def __exit__(self, *exc):
        self.server.__exit__(*exc)

    def ask(self, messages, max_tokens, times=None):
        if self.kind == "ollama":
            return common.stream_ollama(self.server.base, OLLAMA_MODEL, messages,
                                        max_tokens, self.ctx, times=times)
        return common.stream_openai(common.VLLM_URL, VLLM_MODEL, messages, max_tokens,
                                    times=times, repetition_penalty=1.0)


def _record(conn, group, versions, b, r, times, **extra):
    n = r["output_tokens"] or 0
    span = times[-1] - times[0] if len(times) > 1 else 0
    common.storage.record_run(
        conn, run_group=group, backend=b.kind, backend_version=versions[b.kind],
        model=OLLAMA_MODEL if b.kind == "ollama" else VLLM_MODEL,
        quantization="Q4_K_M" if b.kind == "ollama" else "FP16",
        context_length=b.ctx, temperature=0, seed=42, streaming=1,
        response=r["response"], input_tokens=r["input_tokens"],
        cached_input_tokens=r["cached_input_tokens"], output_tokens=n,
        ttft_ms=r["ttft_ms"], client_total_ms=r["client_total_ms"],
        prefill_ms=r["prefill_ms"], decode_ms=r["decode_ms"],
        decode_tokens_per_s=(n - 1) / span if span else None,
        server_num_parallel=b.num_parallel,
        gpu_library="ROCm", hardware_id=common.HARDWARE_ID, **extra)


def run_sweep(conn, group, versions, names, reps, concurrency):
    prompts = common.workloads.PROMPTS
    for name in names:
        with Backend(name, SWEEP[name], SWEEP_CTX) as b:
            levels = SEQS_ONLY_AT if "seqs" in name else concurrency
            for c in levels:
                batch = [prompts[j % len(prompts)] for j in range(c)]
                for rep in range(reps):
                    def one(p):
                        times = []
                        return b.ask(common.workloads.messages(p[2]),
                                     common.workloads.MAX_TOKENS, times), times
                    t0 = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=c) as ex:
                        results = list(ex.map(one, batch))
                    wall = (time.perf_counter() - t0) * 1000
                    for j, ((workload, pid, prompt), (r, times)) in enumerate(zip(batch, results)):
                        _record(conn, group, versions, b, r, times,
                                trial_index=rep * 1000 + c * 10 + j, prompt=prompt,
                                workload=workload, concurrency=c, batch_wall_ms=wall,
                                notes=f"conc backend={name} part=sweep rep={rep} prompt={pid}")
                    total = sum(r["output_tokens"] for r, _ in results)
                    lat = statistics.median(r["client_total_ms"] for r, _ in results)
                    print(f"  {name} c={c} rep{rep}: {total / wall * 1000:6.0f} tok/s aggregate, "
                          f"median latency {lat:6.0f} ms", flush=True)


def _mixed_stats(times, t_arrive, t_until):
    """Base ITL from gaps that ended before t_arrive; max gap and excess for
    gaps that ended inside [t_arrive, t_until]."""
    gaps = [(b - a, b) for a, b in zip(times, times[1:])]
    base = [g for g, end in gaps if end < t_arrive]
    hit = [g for g, end in gaps if t_arrive <= end <= t_until]
    if len(base) < 5 or not hit:
        return None
    itl = statistics.median(base)
    return {"itl_ms": itl * 1000, "max_gap_ms": max(hit) * 1000,
            "excess_ms": sum(max(0.0, g - itl) for g in hit) * 1000}


def run_mixed(conn, group, versions, names, reps):
    corpus = common.corpus_text("docs")
    raw = []
    for name in names:
        with Backend(name, MIXED[name], MIXED_CTX) as b:
            for rep in range(reps):
                # A fresh first line, so no prefix of the long prompt is cached.
                long_msg = (f"Request {name}-{rep}-{time.time():.0f}.\n" + common.LONG_HEADER
                            + corpus[:LONG_CHARS] + common.LONG_QUESTION)
                steady = [[] for _ in range(STEADY)]
                out = [None] * STEADY

                def stream(i):
                    out[i] = b.ask([{"role": "user", "content":
                                     f"Write a long, detailed story about {TOPICS[i]}. "
                                     f"At least 800 words. ({rep})"}],
                                   STEADY_TOKENS, steady[i])
                threads = [threading.Thread(target=stream, args=(i,)) for i in range(STEADY)]
                for t in threads:
                    t.start()
                while min(len(s) for s in steady) < 32:
                    time.sleep(0.02)
                time.sleep(0.5)
                long_times = []
                t_arrive = time.perf_counter()
                lr = b.ask([{"role": "user", "content": long_msg}], LONG_TOKENS, long_times)
                for t in threads:
                    t.join()
                t_until = (long_times[0] if long_times else time.perf_counter()) + 0.1

                stats = [_mixed_stats(s, t_arrive, t_until) for s in steady]
                _record(conn, group, versions, b, lr, long_times, trial_index=rep * 100 + 99,
                        prompt=long_msg, workload="mixed-long", concurrency=STEADY + 1,
                        notes=f"conc backend={name} part=mixed rep={rep} role=long")
                for i, (r, s, st) in enumerate(zip(out, steady, stats)):
                    extra = (f" itl_ms={st['itl_ms']:.1f} max_gap_ms={st['max_gap_ms']:.1f} "
                             f"excess_ms={st['excess_ms']:.1f}" if st else " stats=none")
                    _record(conn, group, versions, b, r, s, trial_index=rep * 100 + i,
                            prompt=TOPICS[i], workload="mixed-steady", concurrency=STEADY + 1,
                            notes=f"conc backend={name} part=mixed rep={rep} role=steady{extra}")
                raw.append({"backend": name, "rep": rep, "t_arrive": t_arrive,
                            "long_ttft_ms": lr["ttft_ms"], "long_input_tokens": lr["input_tokens"],
                            "long_times": long_times, "steady_times": steady})
                ok = [st for st in stats if st]
                print(f"  {name} rep{rep}: long prompt {lr['input_tokens']} tok, TTFT "
                      f"{lr['ttft_ms']:.0f} ms; steady ITL "
                      f"{statistics.median(st['itl_ms'] for st in ok):.1f} ms, max gap "
                      f"{statistics.median(st['max_gap_ms'] for st in ok):.0f} ms, excess "
                      f"{statistics.median(st['excess_ms'] for st in ok):.0f} ms "
                      f"({len(ok)}/{STEADY} streams)" if ok else "  no usable streams",
                      flush=True)
    path = os.path.join(common.LOG_DIR, f"{group}-mixed.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    print(f"raw chunk times: {path}")


def report(group):
    conn = common.storage.connect()
    if not group:
        group = conn.execute("SELECT run_group FROM inference_runs WHERE run_group LIKE "
                             "'p9-conc-%' ORDER BY id DESC LIMIT 1").fetchone()[0]
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM inference_runs WHERE run_group = ? ORDER BY id", (group,))]
    for r in rows:
        r.update(dict(kv.split("=", 1) for kv in r["notes"].split()[1:]))
    print(f"run_group {group}: {len(rows)} rows\n")
    med = statistics.median

    sweep = [r for r in rows if r["part"] == "sweep"]
    if sweep:
        batches = defaultdict(list)
        for r in sweep:
            batches[(r["backend"], r["concurrency"], r["rep"])].append(r)
        agg, lat, per = defaultdict(list), defaultdict(list), defaultdict(list)
        for (b, c, _), g in batches.items():
            agg[(b, c)].append(sum(r["output_tokens"] for r in g) / g[0]["batch_wall_ms"] * 1000)
            lat[(b, c)] += [r["client_total_ms"] for r in g]
            per[(b, c)] += [r["decode_tokens_per_s"] for r in g if r["decode_tokens_per_s"]]
        backends = list(dict.fromkeys(r["backend"] for r in sweep))
        levels = sorted({r["concurrency"] for r in sweep})
        for title, table, fmt in (("aggregate tok/s (median of batches)", agg, "{:7.0f}"),
                                  ("per-stream decode tok/s (median)", per, "{:7.1f}"),
                                  ("request latency ms (median)", lat, "{:7.0f}")):
            print(title)
            print(f"{'backend':<13}" + "".join(f"{'c=' + str(c):>8}" for c in levels))
            for b in backends:
                print(f"{b:<13}" + "".join(
                    f"{fmt.format(med(table[(b, c)])):>8}" if table.get((b, c)) else f"{'-':>8}"
                    for c in levels))
            print()

    mixed = [r for r in rows if r["part"] == "mixed"]
    if mixed:
        print("mixed load: 8 steady streams + one long prompt (medians over streams and reps)")
        print(f"{'backend':<14}{'long tok':>9}{'TTFT ms':>9}{'base ITL':>10}"
              f"{'max gap':>9}{'excess':>8}")
        for b in dict.fromkeys(r["backend"] for r in mixed):
            longs = [r for r in mixed if r["backend"] == b and r["role"] == "long"]
            st = [r for r in mixed if r["backend"] == b and r["role"] == "steady"
                  and "itl_ms" in r]
            print(f"{b:<14}{med(r['input_tokens'] for r in longs):>9.0f}"
                  f"{med(r['ttft_ms'] for r in longs):>9.0f}"
                  f"{med(float(r['itl_ms']) for r in st):>10.1f}"
                  f"{med(float(r['max_gap_ms']) for r in st):>9.0f}"
                  f"{med(float(r['excess_ms']) for r in st):>8.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["sweep", "mixed"])
    ap.add_argument("--backends")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--conc", default=",".join(map(str, CONCURRENCY)))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--group")
    a = ap.parse_args()
    if a.report:
        return report(a.group)
    table = SWEEP if a.part == "sweep" else MIXED
    names = a.backends.split(",") if a.backends else list(table)
    conn = common.storage.connect()
    group = "p9-conc-" + datetime.now().strftime("%Y%m%d-%H%M")
    versions = {"vllm": common.vllm_version(), "ollama": "ollama 0.40.9"}
    print(f"run_group {group}, part {a.part}, backends {names}", flush=True)
    try:
        if a.part == "sweep":
            run_sweep(conn, group, versions, names, a.reps or REPS,
                      [int(c) for c in a.conc.split(",")])
        else:
            run_mixed(conn, group, versions, names, a.reps or MIXED_REPS)
    finally:
        common.evict_ollama()
    print(f"done: {group}")


if __name__ == "__main__":
    main()

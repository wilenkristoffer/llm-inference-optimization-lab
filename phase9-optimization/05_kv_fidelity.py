"""
Phase 9, file 5: what KV cache quantization costs in quality (Stage 2,
Part B). The Phase 6 teacher-forced fidelity method, moved to llama-server
and to long prompts, since KV error grows with the amount of KV attended to.

Per (model, depth): the f16-KV server generates the reference - 256 greedy
tokens after a long prompt - then scores it at every position. That pass is
both the self-test (top-1 against its own greedy text should be ~100%) and the
source of the reference distributions. The q8_0 and q4_0 servers then score
the SAME token ids at every position. Only the KV type differs: same Q4_K_M
weights, same prompt ids, same server build.

Scoring position i sends prompt + reference[:i] as token ids with
cache_prompt, so after the first request every position is a one-token
prefill on top of the cached prefix - and that prefix went through the
quantized KV, exactly as in real decoding. n_probs returns the model's top-k
logprobs before sampling, so greedy decoding costs nothing here. A position
is aligned when cache_n + prompt_n equals the expected length.

Metrics as in Phase 6 (phase6-quantization/02_fidelity.py, whose kl_topk and
summarize are used as-is): top-1 agreement, perplexity of the reference text,
and approximate KL(f16 || quantized) over the f16 top-k.

Two sources per depth: the lab's docs (prose) and its Python (code), since
attention patterns differ - code is full of exact copies from far back.

Usage:
    python 05_kv_fidelity.py                     # full run, ~20 min
    python 05_kv_fidelity.py --configs qwen2.5:1.5b@2048 --sources docs
    python 05_kv_fidelity.py --report [--group p9-kvfid-...]
"""

import argparse
import time
from collections import defaultdict
from datetime import datetime

import common

phase6 = common._load("fidelity6", "../phase6-quantization/02_fidelity.py")

CONFIGS = [("qwen2.5:1.5b", 2048), ("qwen2.5:1.5b", 16384), ("qwen2.5:1.5b", 31744),
           ("qwen2.5:14b", 2048), ("qwen2.5:14b", 16384)]
SOURCES = ["docs", "code"]
# "K/V" = different types for K and V. q8_0/q4_0 is in because a probe found
# that it is K that q4_0 breaks on the small Qwen2.5 models, not V.
KV_TYPES = ["f16", "q8_0", "q4_0", "q8_0/q4_0"]
REF_TOKENS = 256
TOP_K = 20


def completion(body):
    return common._post(common.LLAMA_URL + "/completion",
                        {"temperature": 0, "seed": 42, "repeat_penalty": 1.0, **body},
                        timeout=3600)


def prefill(prompt):
    """Evaluate the prompt from position 0, ignoring whatever the slot holds.

    Why this exists: logits depend on how the prompt is cut into 512-token
    prefill chunks. A request that reuses a cached prefix (e.g. the 34 tokens
    two corpora share) starts chunking at the end of that prefix, so the last
    prompt token lands in a big batch instead of a small one, takes another
    kernel, and its logprobs move by ~0.04 nats - enough to flip a near-tie.
    That broke the first run's f16 self-test (98% top-1) for every source that
    was not the first on its server. So every prompt is prefilled the same way,
    and the reference and every scored position then re-evaluate only the last
    token, alone, on top of it."""
    completion({"prompt": prompt, "n_predict": 1, "cache_prompt": False})


def probs_of(r):
    if "completion_probabilities" not in r:
        raise RuntimeError("no completion_probabilities in response: "
                           + str({k: v for k, v in r.items() if k != "generation_settings"}))
    return r["completion_probabilities"]


def make_reference(prompt):
    """256 greedy tokens after the prompt, as ids and as text pieces."""
    prefill(prompt)
    r = completion({"prompt": prompt, "n_predict": REF_TOKENS, "n_probs": 1,
                    "cache_prompt": True})
    probs = probs_of(r)
    return [p["id"] for p in probs], [p["token"] for p in probs]


def score(prompt, ref_ids):
    """Per position: the model's top-k as {id: logprob}, and whether the
    server saw exactly prompt + i tokens."""
    prefill(prompt)
    out = []
    for i in range(len(ref_ids)):
        r = completion({"prompt": prompt + ref_ids[:i], "n_predict": 1,
                        "n_probs": TOP_K, "cache_prompt": True})
        if "completion_probabilities" not in r:
            # The model's token was part of a multi-byte UTF-8 character:
            # llama-server holds incomplete characters back, probabilities
            # included. Generating on until the character completes releases
            # them; the first entry is still this position's.
            r = completion({"prompt": prompt + ref_ids[:i], "n_predict": 8,
                            "n_probs": TOP_K, "cache_prompt": True})
        t = r["timings"]
        top = probs_of(r)[0]["top_logprobs"]
        out.append({"dist": {e["id"]: e["logprob"] for e in top}, "top1": top[0]["id"],
                    "aligned": t["prompt_n"] + t.get("cache_n", 0) == len(prompt) + i})
    return out


def sweep(configs, sources):
    conn = common.storage.connect()
    group = "p9-kvfid-" + datetime.now().strftime("%Y%m%d-%H%M")
    version = common.llama_version()
    print(f"run_group {group}, {version}, top-{TOP_K}, {REF_TOKENS} reference tokens")
    for model, depth in configs:
        ctx = min(depth + REF_TOKENS + 256, 32768)
        refs = {}               # source -> (prompt ids, ref ids, ref pieces, f16 positions)
        for kv in KV_TYPES:
            k_type, v_type = (kv.split("/") * 2)[:2]
            with common.LlamaServer(model, ctx=ctx, extra=["-ctk", k_type, "-ctv", v_type],
                                    label=f"fid-c{ctx}-{k_type}-{v_type}"):
                for source in sources:
                    si = SOURCES.index(source)
                    t0 = time.perf_counter()
                    if kv == "f16":
                        prompt = common.prompt_tokens(common.long_prompt(depth, source))
                        ids, pieces = make_reference(prompt)
                        refs[source] = (prompt, ids, pieces, None)
                    prompt, ids, pieces, ref_pos = refs[source]
                    positions = score(prompt, ids)
                    if kv == "f16":
                        refs[source] = (prompt, ids, pieces, positions)
                        ref_pos = positions
                    rows = []
                    for i, (pos, rp) in enumerate(zip(positions, ref_pos)):
                        aligned = pos["aligned"] and rp["aligned"]
                        lp = pos["dist"].get(ids[i])
                        rows.append({
                            "run_group": group, "backend": "llama.cpp",
                            "backend_version": version, "model": model,
                            "quantization": "Q4_K_M", "kv_cache_type": kv,
                            "context_depth": len(prompt),
                            "reference_model": f"{model} kv=f16",
                            "prompt_index": si, "position": i, "ref_token": pieces[i],
                            "aligned": int(aligned),
                            "top1_match": int(pos["top1"] == ids[i]) if aligned else None,
                            "ref_logprob": lp if aligned else None,
                            "ref_in_topk": int(lp is not None) if aligned else None,
                            "kl_topk": phase6.kl_topk(rp["dist"], pos["dist"]) if aligned else None,
                            "top_k": TOP_K,
                        })
                    common.storage.record_rows(conn, "fidelity_positions", rows)
                    s = phase6.summarize(rows)
                    print(f"  {model} depth {len(prompt)} {source:<4} {kv:<4}: "
                          f"top-1 {s['top1']:6.1%}  KL {s['kl']:.4f}  ppl {s['ppl']:.3f}  "
                          f"({s['n']} pos, {s['unaligned']} unaligned, "
                          f"{s['outside_topk']} outside top-k, "
                          f"{time.perf_counter() - t0:.0f} s)", flush=True)
                    if kv == "f16" and s["top1"] < 0.99:
                        print("    SELF-TEST below 99%: teacher-forced f16 disagrees with "
                              "its own greedy text", flush=True)
    print(f"done: {group}")


def report(group):
    conn = common.storage.connect()
    if not group:
        group = conn.execute("SELECT run_group FROM fidelity_positions WHERE run_group "
                             "LIKE 'p9-kvfid-%' ORDER BY id DESC LIMIT 1").fetchone()[0]
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM fidelity_positions WHERE run_group = ? ORDER BY id", (group,))]
    print(f"run_group {group}: {len(rows)} positions\n")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["prompt_index"], r["context_depth"], r["kv_cache_type"])].append(r)
    print(f"{'model':<13} {'source':<6} {'depth':>6} {'KV':<5} {'top-1':>7} {'KL':>8} "
          f"{'ppl':>7} {'outside':>7} {'first miss':>10}")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], KV_TYPES.index(k[3]))):
        model, si, depth, kv = key
        g = groups[key]
        s = phase6.summarize(g)
        # Where the first top-1 disagreement falls: under free-running greedy
        # decoding, the text diverges from f16 at this token.
        miss = next((r["position"] for r in g if r["aligned"] and not r["top1_match"]), None)
        print(f"{model:<13} {SOURCES[si]:<6} {depth:>6} {kv:<5} {s['top1']:7.1%} "
              f"{s['kl']:8.4f} {s['ppl']:7.3f} {s['outside_topk']:>7} "
              f"{'-' if miss is None else miss:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", help="model@depth,... (default: all)")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--group")
    a = ap.parse_args()
    if a.report:
        report(a.group)
        return
    configs = CONFIGS
    if a.configs:
        configs = [(c.split("@")[0], int(c.split("@")[1])) for c in a.configs.split(",")]
    sweep(configs, a.sources.split(","))


if __name__ == "__main__":
    main()

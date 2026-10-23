"""
Phase 9, file 3 (side experiment): Jacobi decoding - guessing several tokens
at once with no draft model at all.

The model itself checks its own guesses. Each pass feeds [last, g1..gw]: the
last accepted token plus w guessed continuations. Out comes o0..ow, the
greedy next token after each input position. o0 is always right (it follows
an accepted token). g1 was a guess for o0; if it matches, o1 was computed on
a correct prefix and is right too, and so on:

    k = number of leading j with g_j == o_(j-1)      accept o0..ok  (k+1 tokens)

The KV cache is cropped back to the accepted prefix, the new last is ok, and
the unverified tail o(k+1)..ow becomes the next guesses - each pass refines
the guesses like a fixed-point (Jacobi) iteration. w = 0 is plain greedy
decoding in the same loop, so the baseline shares every overhead.

Why it is expected to do little (plan, Stage 1): a guess is only useful when
the model's outputs at later positions already settled, which needs the
tokens before them right. Published numbers for vanilla Jacobi on chat models
are 1.05-1.1 tokens per pass; the extraction workload, which copies text, is
where it should do best.

Runs in WSL (HF transformers + ROCm PyTorch), Qwen2.5-1.5B-Instruct FP16:
    wsl -d Ubuntu -- bash -lc "source ~/vllm-env/bin/activate && \\
        HF_HUB_OFFLINE=1 python 03_jacobi.py --run --out logs/jacobi.json"
Then on Windows (only Windows writes lab.db):
    python 03_jacobi.py --import logs/jacobi.json
    python 03_jacobi.py --report [--group p9-jacobi-...]
"""

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
WINDOWS = [0, 2, 4, 8, 16]
REPS = 2


def jacobi(model, cache_cls, prompt_ids, w, max_tokens, eos):
    """Greedy-decode up to max_tokens with window w. Returns (tokens, passes,
    accepted_guesses, decode_s). Prefill is outside the timed part."""
    import torch
    cache = cache_cls()
    with torch.no_grad():
        model(prompt_ids[:, :-1], past_key_values=cache, use_cache=True)
        last = int(prompt_ids[0, -1])
        guesses = [last] * w
        out, passes, accepted = [], 0, 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while len(out) < max_tokens:
            base = cache.get_seq_length()
            ids = torch.tensor([[last] + guesses], device=prompt_ids.device)
            o = model(ids, past_key_values=cache, use_cache=True).logits[0].argmax(-1).tolist()
            passes += 1
            k = 0
            while k < w and guesses[k] == o[k]:
                k += 1
            accepted += k
            new = o[:k + 1]
            # Cache keeps last and the k guesses that proved right.
            cache.crop(base + 1 + k)
            if eos in new:
                new = new[:new.index(eos) + 1]
            out += new
            if out[-1] == eos:
                break
            last = o[k]
            tail = o[k + 1:]
            guesses = (tail + [tail[-1] if tail else last] * w)[:w]
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0
    out = out[:max_tokens]
    return out, passes, accepted, decode_s


def run(out_path, reps):
    import sys
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    sys.path.insert(0, HERE)
    import workloads

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    version = f"transformers {transformers.__version__}, torch {torch.__version__}"
    print(version, torch.cuda.get_device_name(0), flush=True)

    def ids_for(prompt):
        text = tok.apply_chat_template(workloads.messages(prompt), tokenize=False,
                                       add_generation_prompt=True)
        return tok(text, return_tensors="pt").input_ids.to("cuda")

    jacobi(model, DynamicCache, ids_for("Say hello."), 4, 16, eos)     # warm-up
    results = []
    # Interleaved: every window runs on every prompt within each rep.
    for rep in range(reps):
        for i, (workload, pid, prompt) in enumerate(workloads.PROMPTS):
            ids = ids_for(prompt)
            for w in WINDOWS:
                toks, passes, accepted, s = jacobi(model, DynamicCache, ids, w,
                                                   workloads.MAX_TOKENS, eos)
                results.append(dict(
                    rep=rep, index=i, workload=workload, prompt_id=pid, prompt=prompt,
                    window=w, input_tokens=ids.shape[1], output_tokens=len(toks),
                    passes=passes, accepted=accepted, decode_ms=s * 1000,
                    response=tok.decode(toks, skip_special_tokens=True)))
                print(f"  rep{rep} {workload}/{pid} w={w:2d}: {len(toks) / s:6.1f} tok/s "
                      f"{len(toks) / passes:.2f} tok/pass", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"version": version, "model": MODEL, "results": results}, f)
    print(f"wrote {out_path}")


def import_json(path):
    import common
    data = json.load(open(path, encoding="utf-8"))
    conn = common.storage.connect()
    group = "p9-jacobi-" + datetime.now().strftime("%Y%m%d-%H%M")
    for r in data["results"]:
        w = r["window"]
        common.storage.record_run(
            conn, run_group=group, trial_index=r["rep"] * 1000 + r["index"] * 100 + w,
            backend="transformers", backend_version=data["version"],
            model="qwen2.5-1.5b-instruct", quantization="FP16", context_length=4096,
            temperature=0, streaming=0, prompt=r["prompt"], response=r["response"],
            input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
            decode_ms=r["decode_ms"],
            decode_tokens_per_s=r["output_tokens"] / r["decode_ms"] * 1000,
            spec_method="jacobi" if w else None, spec_k=w,
            draft_tokens=w * r["passes"] if w else None,
            draft_accepted=r["accepted"] if w else None, target_passes=r["passes"],
            workload=r["workload"], gpu_library="ROCm", hardware_id=common.HARDWARE_ID,
            notes=f"prompt={r['prompt_id']} phase=jacobi-w{w}")
    print(f"imported {len(data['results'])} rows as {group}")


def report(group):
    import common
    conn = common.storage.connect()
    if not group:
        group = conn.execute("SELECT run_group FROM inference_runs WHERE run_group "
                             "LIKE 'p9-jacobi-%' ORDER BY id DESC LIMIT 1").fetchone()[0]
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM inference_runs WHERE run_group = ? ORDER BY id", (group,))]
    print(f"run_group {group}: {len(rows)} rows\n")

    def prompt_id(r):
        return r["notes"].split()[0].split("=")[1]

    plain = [r for r in rows if not r["spec_k"]]
    base = defaultdict(list)
    ref = {}
    for r in plain:
        base[r["workload"]].append(r["decode_tokens_per_s"])
        ref.setdefault(prompt_id(r), (r["id"], r["response"]))
    print(f"  {'w':>2} {'workload':<11} {'tok/s':>7} {'speedup':>8} {'tok/pass':>8} "
          f"{'ms/pass':>8} {'guess ok':>8} {'diverged':>9}")
    for w in sorted({r["spec_k"] for r in rows}):
        for wl in ("code", "extraction", "prose"):
            g = [r for r in rows if r["spec_k"] == w and r["workload"] == wl]
            tps = statistics.median(r["decode_tokens_per_s"] for r in g)
            passes = sum(r["target_passes"] for r in g)
            tpp = sum(r["output_tokens"] for r in g) / passes
            ms = sum(r["decode_ms"] for r in g) / passes
            ok = (f"{sum(r['draft_accepted'] for r in g) / sum(r['draft_tokens'] for r in g):8.3f}"
                  if w else f"{'':>8}")
            div = [common.divergence(ref[prompt_id(r)][1], r["response"]) for r in g
                   if r["id"] != ref[prompt_id(r)][0]]
            ndiv = sum(d is not None for d in div)
            print(f"  {w:>2} {wl:<11} {tps:7.1f} {tps / statistics.median(base[wl]):7.2f}x "
                  f"{tpp:8.2f} {ms:8.2f} {ok} {ndiv:>4}/{len(div):<4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "jacobi.json"))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--import", dest="import_path")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--group")
    a = ap.parse_args()
    if a.run:
        run(a.out, a.reps)
    elif a.import_path:
        import_json(a.import_path)
    else:
        report(a.group)


if __name__ == "__main__":
    main()

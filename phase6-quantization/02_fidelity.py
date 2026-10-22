"""
Phase 6, file 2: how far does a quantized model drift from its reference?

Task accuracy on a few dozen questions is a blunt instrument - two levels have
to differ by ~15 points on 50 questions before the difference beats noise, and
quantization damage starts long before answers flip. So this compares the
NEXT-TOKEN DISTRIBUTIONS directly, token by token, on the same text. It is the
same idea as llama.cpp's KL-divergence tool, done through Ollama's API.

METHOD - teacher forcing

1. The reference model (FP16) writes a greedy continuation of each prompt.
2. For every position i in that text, each model is shown the prompt plus the
   first i reference tokens and asked for ONE token with its top-k
   alternatives. Every model is scored on the reference's text, never its
   own, so a single early divergence cannot snowball into "everything after
   differs".
3. Per position: did the model's top token match the reference's? What
   probability did it give the reference token? How far is its distribution
   from the reference's (KL, approximated over the top-k)?

The reference is scored against itself first, as a self-test: teacher forcing
must reproduce greedy decoding, so its top-1 agreement must be ~100%. If it is
not, the method is broken and nothing after it means anything.

Three details that make it trustworthy:

- raw=true with the chat template written out by hand. Every other endpoint
  applies a template (Phase 2), and the text has to be exactly what we think.
- ALIGNMENT IS CHECKED, not assumed. Re-tokenizing a decoded prefix does not
  always reproduce the same tokens. The server's prompt_eval_count must equal
  template tokens + i; positions where it does not are stored with aligned=0
  and excluded from the summary.
- Each request extends the previous one by one token, so the prefix cache
  (Phase 5 file 04) makes ~2,000 requests per model take about a minute.

THE KL IS AN APPROXIMATION. Ollama returns the top-k alternatives, not the full
150k-token distribution. KL(P || Q) is summed over the reference's top-k
tokens, and where one of them is missing from the model's top-k its
probability is taken as the model's k-th (an upper bound on it, so the KL is
an underestimate there). Fine for confident tokens, where the mass outside the
top 20 is negligible; less so for uncertain ones.

Usage:
    python 02_fidelity.py --reference qwen2.5:1.5b-instruct-fp16 --models qwen2.5:1.5b
    python 02_fidelity.py --reference qwen2.5:3b-instruct-fp16 \\
        --models qwen2.5:3b-instruct-q8_0,qwen2.5:3b,qwen2.5:3b-instruct-q2_K
    python 02_fidelity.py --report
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


storage = _load("storage", "../phase1-ollama-baseline/04_storage.py")
trials = _load("trials", "../phase1-ollama-baseline/03_repeated_trials.py")

OLLAMA = "http://127.0.0.1:11434"

# Chat templates written out, because raw=true bypasses templating entirely.
# Every model in one run must use the same one - the comparison is only fair on
# identical text. The reference's self-test (~100% top-1) is what confirms a
# template is right; a wrong one shows up there first.
TEMPLATES = {
    # Qwen2.5's ChatML with its default system prompt, as Ollama renders it.
    "qwen2.5": ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You "
                "are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}"
                "<|im_end|>\n<|im_start|>assistant\n"),
    # Qwen3.5's non-thinking form: an empty think block, so the model answers
    # directly. Ollama renders this model in Go code (RENDERER qwen3.5), not a
    # template, so this is written from the model's documented format.
    "qwen3.5": ("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n"),
}
TEMPLATE = TEMPLATES["qwen2.5"]

# Pinned exactly as everywhere else in the lab (temperature 0 = greedy). The
# presence and frequency penalties too: Qwen3.5's model file defaults
# presence_penalty to 1.5, and a penalty changes which token wins even at
# temperature 0. Qwen2.5's default is 0, so pinning it changes nothing there.
OPTIONS = {"temperature": 0, "seed": 42, "repeat_penalty": 1.0,
           "presence_penalty": 0, "frequency_penalty": 0, "num_ctx": 4096}

# Varied on purpose: quantization damage is not uniform. Prose is forgiving,
# while arithmetic, code and exact structure tend to show it first. Swedish is
# in because it tokenizes worse (Phase 5: 1.31x English) and is the user's own
# language.
PROMPTS = [
    # explanation
    "Explain why the sky is blue in three sentences.",
    "Explain what a hash table is to a beginner programmer.",
    "Describe how a refrigerator keeps food cold.",
    "What causes the seasons on Earth?",
    "Explain the difference between weather and climate.",
    # reasoning and arithmetic
    "A train leaves at 14:20 and the journey takes 2 hours 55 minutes. When does it arrive? Show your working.",
    "If 3 pencils cost 1.20 dollars, how much do 7 pencils cost? Explain step by step.",
    "What is 17 multiplied by 23? Show the calculation.",
    "A rectangle has a perimeter of 30 cm and one side is 6 cm. What is its area?",
    "Is 221 a prime number? Explain how you checked.",
    # code
    "Write a Python function that checks whether a number is prime.",
    "Write a SQL query that returns the ten most recent orders from a table called orders.",
    "Write a JavaScript function that reverses a string.",
    "Explain what this Python does: [x * x for x in range(10) if x % 2 == 0]",
    # structured output
    'Return a JSON object with the fields "name", "age" and "city" for a fictional person.',
    "List the planets of the solar system in order from the Sun, one per line.",
    "Give me a markdown table with three fruits and their colours.",
    "Convert this to a bulleted list: apples, bread, milk, eggs, coffee.",
    # factual
    "Who wrote the novel Pride and Prejudice, and when was it published?",
    "What is the chemical symbol for gold, and what is its atomic number?",
    "Name the largest ocean on Earth and give one fact about it.",
    "What does the abbreviation CPU stand for and what does it do?",
    # instruction following
    "Write exactly three words that describe the ocean.",
    "Answer with only yes or no: is water wet?",
    "Rewrite this sentence in the passive voice: The cat chased the mouse.",
    # creative
    "Write the opening two sentences of a mystery story set in a lighthouse.",
    "Write a haiku about autumn.",
    # Swedish - escaped so the source file stays ASCII
    "F\u00f6rklara kort vad en dator \u00e4r.",
    "Vad \u00e4r huvudstaden i Sverige, och vad \u00e4r den k\u00e4nd f\u00f6r?",
    "Skriv tre meningar om sommaren i Sverige.",
]


def generate(model, prompt, num_predict, top_logprobs=0):
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False,
            "logprobs": True, "options": {**OPTIONS, "num_predict": num_predict}}
    if top_logprobs:
        body["top_logprobs"] = top_logprobs
    request = urllib.request.Request(
        OLLAMA + "/api/generate", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def token_bytes(entry):
    """A token's exact bytes. Some entries (special tokens) carry no bytes
    field, only text."""
    if entry.get("bytes") is not None:
        return bytes(entry["bytes"])
    return entry["token"].encode("utf-8")


def make_reference(model, tokens):
    """Greedy text from the reference model, kept as its exact token bytes."""
    refs = []
    for index, prompt in enumerate(PROMPTS):
        base = TEMPLATE.format(prompt=prompt)
        out = generate(model, base, tokens)
        refs.append({"index": index, "base": base,
                     "base_tokens": out["prompt_eval_count"],
                     "tokens": [token_bytes(t) for t in out["logprobs"]]})
    return refs


def score(model, refs, top_k):
    """
    Teacher-forced pass of `model` over every reference position. Returns one
    dict per position with the model's top-k {token_bytes: logprob} kept in
    memory (for KL against the reference) but not stored.
    """
    positions = []
    for ref in refs:
        for i, expected in enumerate(ref["tokens"]):
            prefix = b"".join(ref["tokens"][:i])
            try:
                text = prefix.decode("utf-8")
            except UnicodeDecodeError:
                # The prefix ends inside a multi-byte character; it cannot be
                # sent as text. Rare in practice, and recorded as unaligned.
                positions.append({"prompt_index": ref["index"], "position": i,
                                  "expected": expected, "aligned": 0, "dist": None})
                continue
            out = generate(model, ref["base"] + text, 1, top_logprobs=top_k)
            if not out.get("logprobs"):
                # The model's top choice was END-OF-TURN: Ollama stops without
                # emitting it, so the reply is empty and no logprobs come
                # back. Seen on 3B Q2_K, mid-sentence, where FP16 had " that"
                # or " because". A real top-1 miss whose KL is unknown - stored
                # as top1_match=0 with kl_topk NULL rather than guessed.
                positions.append({
                    "prompt_index": ref["index"], "position": i, "expected": expected,
                    "aligned": int(out.get("prompt_eval_count") == ref["base_tokens"] + i),
                    "dist": None, "top1": b"<end-of-turn>", "stopped": True,
                })
                continue
            top = out["logprobs"][0]["top_logprobs"]
            positions.append({
                "prompt_index": ref["index"], "position": i, "expected": expected,
                "aligned": int(out.get("prompt_eval_count") == ref["base_tokens"] + i),
                "dist": {token_bytes(t): t["logprob"] for t in top},
                "top1": token_bytes(top[0]),
            })
    return positions


def kl_topk(ref_dist, model_dist):
    """KL(reference || model) over the reference's top-k. See the docstring."""
    floor = min(model_dist.values())
    return sum(math.exp(lp) * (lp - model_dist.get(tok, floor))
               for tok, lp in ref_dist.items())


def describe(model):
    """Quantization and digest from the server, never typed in by hand."""
    show = trials._ollama("/api/show", {"model": model})
    digest = next((m.get("digest") for m in trials._ollama("/api/tags")["models"]
                   if m.get("name") == model or m.get("model") == model), None)
    return {"quantization": show.get("details", {}).get("quantization_level"),
            "model_digest": digest}


def summarize(rows):
    """Summary over ALIGNED positions only."""
    ok = [r for r in rows if r["aligned"]]
    if not ok:
        return None
    lps = [r["ref_logprob"] for r in ok if r["ref_logprob"] is not None]
    kls = [r["kl_topk"] for r in ok if r["kl_topk"] is not None]
    return {
        "n": len(ok), "unaligned": len(rows) - len(ok),
        "top1": statistics.mean(r["top1_match"] for r in ok),
        "outside_topk": sum(1 for r in ok if not r["ref_in_topk"]),
        # Aligned positions with no KL: the model chose to end its turn.
        "stopped": len(ok) - len(kls),
        # Perplexity of the reference's text under this model, over positions
        # where the reference token was in the top-k (the rest are counted
        # separately above rather than given an invented probability).
        "ppl": math.exp(-statistics.mean(lps)) if lps else None,
        # Over positions where it could be computed - an UNDERESTIMATE for a
        # model that stops early, since its worst positions are the missing ones.
        "kl": statistics.mean(kls) if kls else None,
    }


def report(conn):
    rows = conn.execute("""
        SELECT run_group, model, quantization, reference_model,
               COUNT(*) AS n,
               SUM(aligned) AS aligned,
               AVG(CASE WHEN aligned = 1 THEN top1_match END)  AS top1,
               AVG(CASE WHEN aligned = 1 THEN kl_topk END)     AS kl,
               EXP(-AVG(CASE WHEN aligned = 1 THEN ref_logprob END)) AS ppl,
               SUM(CASE WHEN aligned = 1 AND ref_in_topk = 0 THEN 1 ELSE 0 END) AS outside,
               -- aligned but no KL = the model chose to end its turn there
               SUM(CASE WHEN aligned = 1 AND kl_topk IS NULL THEN 1 ELSE 0 END) AS ended,
               MIN(timestamp_utc) AS started
        FROM fidelity_positions
        GROUP BY run_group, model
        ORDER BY started, kl
    """).fetchall()
    if not rows:
        print("no fidelity runs recorded yet")
        return
    print(f"{'model':<30} {'quant':<8} {'top-1':>7} {'KL':>8} {'ppl':>7} "
          f"{'out':>4} {'ended':>5} {'n':>5}  reference")
    print("-" * 102)
    for r in rows:
        print(f"{r['model']:<30} {str(r['quantization']):<8} {r['top1']:>7.1%} "
              f"{r['kl']:>8.4f} {r['ppl']:>7.3f} {r['outside']:>4} {r['ended']:>5} "
              f"{r['aligned']:>5}  {r['reference_model']}  [{r['run_group'][-15:]}]")
    print("\nout = reference token outside the model's top-k; ended = the model chose "
          "to end its turn there\n(no logprobs, so excluded from KL - which is then "
          "an underestimate for that model).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="qwen2.5:1.5b-instruct-fp16")
    parser.add_argument("--models", default="qwen2.5:1.5b",
                        help="comma-separated tags to score against the reference")
    parser.add_argument("--tokens", type=int, default=64,
                        help="reference length per prompt")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--template", default="qwen2.5", choices=sorted(TEMPLATES))
    parser.add_argument("--notes", default=None)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    conn = storage.connect()
    if args.report:
        report(conn)
        return

    global TEMPLATE
    TEMPLATE = TEMPLATES[args.template]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    version = trials._ollama("/api/version").get("version")
    run_group = f"fidelity|{args.reference}|{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    print(f"reference {args.reference}, {len(PROMPTS)} prompts x {args.tokens} tokens, "
          f"top-{args.top_k}, template {args.template}, ollama {version}")

    t0 = time.perf_counter()
    refs = make_reference(args.reference, args.tokens)
    print(f"reference text generated ({sum(len(r['tokens']) for r in refs)} tokens, "
          f"{time.perf_counter() - t0:.0f} s)")

    # The reference scored against itself: the self-test, and the source of
    # the reference distributions every other model is compared with.
    ref_positions = None
    results = []
    for model in [args.reference] + models:
        t0 = time.perf_counter()
        positions = score(model, refs, args.top_k)
        if ref_positions is None:
            ref_positions = positions
        meta = describe(model)

        rows = []
        for pos, ref_pos in zip(positions, ref_positions):
            aligned = pos["aligned"] and ref_pos["aligned"]
            stopped = pos.get("stopped", False)
            dist = pos["dist"] or {}
            lp = dist.get(pos["expected"])
            rows.append({
                "run_group": run_group, "backend": "ollama", "backend_version": version,
                "model": model, "model_digest": meta["model_digest"],
                "quantization": meta["quantization"],
                "reference_model": args.reference,
                "prompt_index": pos["prompt_index"], "position": pos["position"],
                "ref_token": pos["expected"].decode("utf-8", "replace"),
                "aligned": int(bool(aligned)),
                "top1_match": int(pos.get("top1") == pos["expected"]) if aligned else None,
                "ref_logprob": lp if aligned else None,
                "ref_in_topk": int(lp is not None) if aligned else None,
                "kl_topk": (kl_topk(ref_pos["dist"], dist)
                            if aligned and not stopped else None),
                "top_k": args.top_k,
            })
        storage.record_rows(conn, "fidelity_positions", rows)

        s = summarize(rows)
        results.append((model, meta["quantization"], s))
        print(f"  {model:<32} {str(meta['quantization']):<7} top-1 {s['top1']:6.1%}  "
              f"KL {s['kl']:.4f}  ppl {s['ppl']:.3f}  "
              f"({s['n']} positions, {s['unaligned']} unaligned, "
              + (f"{s['stopped']} ENDED TURN, " if s["stopped"] else "")
              + f"{time.perf_counter() - t0:.0f} s)")

        if model == args.reference and s["top1"] < 0.99:
            print(f"\nSELF-TEST FAILED: the reference agrees with its own greedy "
                  f"text only {s['top1']:.1%} of the time. Teacher forcing is not "
                  f"reproducing generation, so no comparison below would mean "
                  f"anything. Stopping.")
            sys.exit(1)

    print(f"\nstored in {run_group}")


if __name__ == "__main__":
    main()

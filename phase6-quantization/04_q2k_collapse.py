"""
Phase 6, file 4: why does Q2_K destroy Qwen2.5-3B but not the 1.5B?

Established before this file (docs/phase6.md):
- 3B Q2_K emits "433333..." on GPU AND CPU, so it is not a ROCm kernel bug.
- A Q2_K made locally from our own FP16 is the same size and the same
  garbage, so it is not a bad file on Ollama's hub.
- The 1.5B and 3B Q2_K files use the IDENTICAL recipe (same type per tensor
  role, token embeddings kept at Q6_K), and the 1.5B is coherent.

So the same recipe is survivable for one model and fatal for the other. The
working hypothesis, from published work on outliers, is that a few layers of
the 3B - typically ffn_down near the first and last blocks - carry extreme
values that 2-3 bit quantization cannot represent. Q3_K_M, which works, gives
exactly two ffn_down layers extra precision; Q2_K gives none.

METHOD. Build Q2_K variants from FP16 with llama.cpp's own llama-quantize
(shipped inside Ollama), each raising the precision of one set of tensors,
and score each with file 02's teacher-forced fidelity against FP16. A variant
that recovers names the tensors that were breaking. Everything is built in a
temp directory and deleted afterwards, models included.

A variant is NAME=OVERRIDES, overrides being llama-quantize --tensor-type
specs (a regex on the tensor name = a type), separated by '+':

    base=                              plain Q2_K, the reproduction
    down4=ffn_down=q4_k                every ffn_down at Q4_K
    down_ends=blk\\.(0|1|34|35)\\.ffn_down=q5_k

Usage:
    python 04_q2k_collapse.py --variants "base=,down4=ffn_down=q4_k"
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
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
fidelity = _load("fidelity", "02_fidelity.py")

QUANTIZE = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                        "Programs", "Ollama", "lib", "ollama", "llama-quantize.exe")


def model_blob(model):
    modelfile = trials._ollama("/api/show", {"model": model})["modelfile"]
    return next(l.split(None, 1)[1].strip() for l in modelfile.splitlines()
                if l.startswith("FROM "))


def modelfile_body(model):
    """Everything but FROM, so a variant gets the hub model's exact template."""
    modelfile = trials._ollama("/api/show", {"model": model})["modelfile"]
    return "\n".join(l for l in modelfile.splitlines()
                     if not l.startswith("FROM ") and not l.startswith("#"))


def build(source, out_path, overrides):
    cmd = [QUANTIZE]
    for spec in overrides:
        cmd += ["--tensor-type", spec]
    cmd += [source, out_path, "Q2_K"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stdout[-2000:] + result.stderr[-2000:])


def sample(model):
    """One short answer, to show what the numbers mean in words."""
    out = fidelity.generate(
        model, fidelity.TEMPLATE.format(prompt="What is the capital of Sweden? Answer briefly."), 20)
    return out.get("response", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--reference", default="qwen2.5:3b-instruct-fp16",
                        help="full-precision model the variants are built from and scored against")
    parser.add_argument("--template-from", default="qwen2.5:3b-instruct-q2_K",
                        help="model whose Modelfile (template, renderer, parameters) the variants copy")
    parser.add_argument("--template", default="qwen2.5", choices=sorted(fidelity.TEMPLATES))
    args = parser.parse_args()
    fidelity.TEMPLATE = fidelity.TEMPLATES[args.template]
    REFERENCE, TEMPLATE_FROM = args.reference, args.template_from

    variants = []
    for item in args.variants.split(","):
        name, _, spec = item.partition("=")
        variants.append((name.strip(), [s for s in spec.split("+") if s]))

    if not os.path.exists(QUANTIZE):
        sys.exit(f"llama-quantize not found at {QUANTIZE}")
    source = model_blob(REFERENCE)
    body = modelfile_body(TEMPLATE_FROM)
    conn = storage.connect()
    version = trials._ollama("/api/version").get("version")
    run_group = f"q2k|{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    workdir = tempfile.mkdtemp(prefix="lab-q2k-")

    print(f"reference {REFERENCE}, template {args.template}; building variants in {workdir}")
    refs = fidelity.make_reference(REFERENCE, args.tokens)
    ref_positions = fidelity.score(REFERENCE, refs, 20)
    # The self-test (file 02): the reference must reproduce its own greedy
    # text. Below ~99% the template or method is wrong for this model, and
    # every variant score would be meaningless.
    ok = [p for p in ref_positions if p["aligned"] and p["dist"]]
    self_top1 = sum(p["top1"] == p["expected"] for p in ok) / max(len(ok), 1)
    print(f"reference self-test: top-1 {self_top1:.1%} over {len(ok)} positions")
    if self_top1 < 0.99:
        sys.exit("self-test failed - fix the template before trusting any variant")

    try:
        print(f"\n{'variant':<12} {'size MB':>8} {'top-1':>7} {'KL':>8} {'ended':>6}  sample")
        print("-" * 90)
        for name, overrides in variants:
            gguf = os.path.join(workdir, f"{name}.gguf")
            family = REFERENCE.split(":")[0].replace(".", "")
            tag = f"lab-q2k-{family}-{name.lower()}"
            t0 = time.perf_counter()
            build(source, gguf, overrides)
            size_mb = os.path.getsize(gguf) / 2**20
            mf = os.path.join(workdir, f"{name}.Modelfile")
            with open(mf, "w", encoding="utf-8") as f:
                f.write(f"FROM {gguf}\n{body}\n")
            subprocess.run(["ollama", "create", tag, "-f", mf],
                           capture_output=True, check=True)
            try:
                positions = fidelity.score(tag, refs, 20)
                rows = []
                for pos, ref_pos in zip(positions, ref_positions):
                    aligned = pos["aligned"] and ref_pos["aligned"]
                    stopped = pos.get("stopped", False)
                    dist = pos["dist"] or {}
                    lp = dist.get(pos["expected"])
                    rows.append({
                        "run_group": run_group, "backend": "ollama",
                        "backend_version": version, "model": tag,
                        "quantization": "Q2_K+" + ("+".join(overrides) or "none"),
                        "reference_model": REFERENCE,
                        "prompt_index": pos["prompt_index"], "position": pos["position"],
                        "ref_token": pos["expected"].decode("utf-8", "replace"),
                        "aligned": int(bool(aligned)),
                        "top1_match": int(pos.get("top1") == pos["expected"]) if aligned else None,
                        "ref_logprob": lp if aligned else None,
                        "ref_in_topk": int(lp is not None) if aligned else None,
                        "kl_topk": (fidelity.kl_topk(ref_pos["dist"], dist)
                                    if aligned and not stopped else None),
                        "top_k": 20,
                    })
                storage.record_rows(conn, "fidelity_positions", rows)
                s = fidelity.summarize(rows)
                text = sample(tag)
                kl = f"{s['kl']:.4f}" if s["kl"] is not None else "-"
                print(f"{name:<12} {size_mb:>8,.0f} {s['top1']:>7.1%} {kl:>8} "
                      f"{s['stopped']:>6}  {text[:40]!r}   ({time.perf_counter() - t0:.0f} s)")
            finally:
                subprocess.run(["ollama", "rm", tag], capture_output=True)
                os.remove(gguf)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"\nstored in {run_group}; temp models and files removed")


if __name__ == "__main__":
    main()

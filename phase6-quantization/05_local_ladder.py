"""
Phase 6, file 5: build a quantization ladder locally from full-precision weights.

Ollama's library carries only Q4_K_M, Q8_0 and BF16 for Qwen3.5-4B. Mixing
those with levels made elsewhere would compare tools as well as precisions, so
EVERY level here is built the same way: from the BF16 weights, with the
llama-quantize that ships inside Ollama, using its default recipe for that
type. Each is registered with Ollama under a lab-* name, with the original's
Modelfile (renderer, parser, template) so it behaves like the library model.

The temporary GGUF is deleted once Ollama has copied it into its own store.
--remove deletes the registered models again (docs/cleanup-plan.md, Stage 1).

Usage:
    python 05_local_ladder.py --source qwen3.5:4b-bf16 --name lab-qwen3.5-4b
    python 05_local_ladder.py --name lab-qwen3.5-4b --remove
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collapse = _load("collapse", "04_q2k_collapse.py")
trials = _load("trials", "../phase1-ollama-baseline/03_repeated_trials.py")

LEVELS = ["Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


def tag(name, level):
    return f"{name}:{level.lower()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="qwen3.5:4b-bf16")
    parser.add_argument("--name", default="lab-qwen3.5-4b")
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]

    if args.remove:
        for level in levels:
            out = subprocess.run(["ollama", "rm", tag(args.name, level)],
                                 capture_output=True, text=True)
            print(f"  {tag(args.name, level):<28} "
                  f"{'removed' if out.returncode == 0 else 'not present'}")
        return

    if not os.path.exists(collapse.QUANTIZE):
        sys.exit(f"llama-quantize not found at {collapse.QUANTIZE}")
    source = collapse.model_blob(args.source)
    body = collapse.modelfile_body(args.source)
    existing = {m["name"] for m in trials._ollama("/api/tags")["models"]}

    with tempfile.TemporaryDirectory(prefix="lab-ladder-") as workdir:
        for level in levels:
            name = tag(args.name, level)
            if name in existing:
                print(f"  {name:<28} already built")
                continue
            t0 = time.perf_counter()
            gguf = os.path.join(workdir, f"{level}.gguf")
            result = subprocess.run([collapse.QUANTIZE, source, gguf, level],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                sys.exit(f"llama-quantize failed for {level}:\n{result.stderr[-1500:]}")
            size_mb = os.path.getsize(gguf) / 2**20
            modelfile = os.path.join(workdir, f"{level}.Modelfile")
            with open(modelfile, "w", encoding="utf-8") as f:
                f.write(f"FROM {gguf}\n{body}\n")
            subprocess.run(["ollama", "create", name, "-f", modelfile],
                           capture_output=True, check=True)
            os.remove(gguf)
            print(f"  {name:<28} {size_mb:>7,.0f} MB  ({time.perf_counter() - t0:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main()

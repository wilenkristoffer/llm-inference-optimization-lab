"""
Phase 11, file 9: the adapter served unmerged by vLLM.

vLLM 0.29 (~/vllm-env, the Phase 3 install) loads Qwen3.5-4B as
Qwen3_5ForConditionalGeneration, the vision-language wrapper, whose text
layers sit under model.language_model.layers.N. PEFT saved the adapter with
model.layers.N (the training model was the text-only class), and vLLM drops
LoRA weights that match no module WITHOUT A WARNING: the output is the base
model's, token for token. So the adapter is first copied to runs/<name>/vllm/
with its keys renamed to the checkpoint's prefix.

The check is 08's: greedy continuations of the scripted val episodes' assistant
turns, compared with llama-server's texts in logs/serve/<name>/check.json.
Two servers differ a little in arithmetic order, so a near-tie can flip a
token; base vs base gives that floor, and lora vs lora should sit at it if the
adapter is applied in full. Matching texts mean the D/D' results carry over.

Not measured: speed. The texts are generated as one batch, so the wall time
is not comparable with llama-server's single stream.

Usage (Windows side; unloads Ollama's models, vLLM runs in WSL):
    python 09_vllm_adapter.py --name t373-r16
    python 09_vllm_adapter.py --name t373-r16 --configs lora
    python 09_vllm_adapter.py --name t373-r16 --report
"""

# vLLM's engine core is a spawned process that re-imports this file, in
# vllm-env: stdlib only at the top, the Windows-side modules inside functions.
import argparse
import importlib
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "logs", "vllm")
VLLM_PYTHON = "source ~/vllm-env/bin/activate && HF_HUB_OFFLINE=1 python"
MODEL = "Qwen/Qwen3.5-4B"  # ftcommon.MODEL / REVISION; ftcommon needs torch
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER_PREFIX = "base_model.model.model.layers."
VLLM_PREFIX = "base_model.model.model.language_model.layers."
CHECK_TOKENS = 96  # 08's
ENGINE = dict(dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.75,
              limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=16, seed=42)


# ---- WSL side -------------------------------------------------------------

def rename(name):
    """runs/<name>/vllm/: the adapter with the VL wrapper's module names."""
    import shutil
    from safetensors.torch import load_file, save_file

    src = os.path.join(HERE, "runs", name)
    dst = os.path.join(src, "vllm")
    os.makedirs(dst, exist_ok=True)
    t = load_file(os.path.join(src, "adapter_model.safetensors"))
    assert all(k.startswith(ADAPTER_PREFIX) for k in t), "unexpected adapter keys"
    save_file({VLLM_PREFIX + k[len(ADAPTER_PREFIX):]: v for k, v in t.items()},
              os.path.join(dst, "adapter_model.safetensors"), metadata={"format": "pt"})
    shutil.copy(os.path.join(src, "adapter_config.json"), dst)
    return dst, len(t)


def generate(name, config):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    d = os.path.join(OUT, name)
    prompts = json.load(open(os.path.join(HERE, "runs", name, "vllm-prompts.json"),
                             encoding="utf-8"))
    kw = dict(ENGINE)
    lora = None
    if config == "lora":
        adapter, n = rename(name)
        r = json.load(open(os.path.join(adapter, "adapter_config.json")))["r"]
        kw.update(enable_lora=True, max_lora_rank=r, max_loras=1)
        lora = LoRARequest(name, 1, adapter)
        print(f"{config}: {n} adapter tensors renamed", flush=True)
    t0 = time.perf_counter()
    llm = LLM(model=MODEL, revision=REVISION, **kw)
    load_s = round(time.perf_counter() - t0, 1)
    print(f"{config}: loaded in {load_s} s", flush=True)
    sp = SamplingParams(temperature=0, max_tokens=CHECK_TOKENS, repetition_penalty=1.0,
                        stop=["<|im_end|>", "<|im_start|>"])
    outs = llm.generate(prompts, sp, lora_request=lora)
    with open(os.path.join(d, f"{config}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"load_s": load_s, "texts": [o.outputs[0].text for o in outs]}, f,
                  ensure_ascii=False)


# ---- Windows side ---------------------------------------------------------

def run(name, configs):
    serve = importlib.import_module("08_serve_adapter")
    merge_export = serve.merge_export
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    with open(os.path.join(HERE, "runs", name, "vllm-prompts.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(serve.check_prompts(), f, ensure_ascii=False)
    serve.unload_ollama()
    for config in configs:
        log = merge_export.wsl_path(os.path.join(OUT, name, f"{config}.engine.log"))
        cmd = (f"cd {merge_export.wsl_path(HERE)} && {VLLM_PYTHON} 09_vllm_adapter.py "
               f"--wsl-generate --name {name} --configs {config} > {log} 2>&1")
        print(f"{config}: starting vLLM (engine log: logs/vllm/{name}/{config}.engine.log)",
              flush=True)
        subprocess.run(["wsl", "-e", "bash", "-lc", cmd], check=True)
    report(name)


def report(name):
    d = os.path.join(OUT, name)
    ref = json.load(open(os.path.join(HERE, "logs", "serve", name, "check.json"), encoding="utf-8"))
    out = {"name": name, "engine": ENGINE, "configs": {}}
    for config in ("base", "lora"):
        path = os.path.join(d, f"{config}.json")
        if not os.path.exists(path):
            continue
        mine = json.load(open(path, encoding="utf-8"))
        e = {"load_s": mine["load_s"]}
        for c in ("base", "merged", "lora"):
            pairs = list(zip(mine["texts"], ref[c]["texts"]))
            e[f"vs llama-server {c}"] = {
                "turns": len(pairs), "identical": sum(a == b for a, b in pairs),
                "same_first_40_chars": sum(a[:40] == b[:40] for a, b in pairs)}
        out["configs"][config] = e
    print(json.dumps(out, indent=1))
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--configs", nargs="+", default=["base", "lora"])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--wsl-generate", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.wsl_generate:
        generate(args.name, args.configs[0])
    elif args.report:
        report(args.name)
    else:
        run(args.name, args.configs)


if __name__ == "__main__":
    main()

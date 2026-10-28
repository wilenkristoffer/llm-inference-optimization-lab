"""
Phase 11, file 6: a trained adapter -> an Ollama tag.

No llama.cpp conversion. The merge patches Ollama's own bf16 GGUF of the
student - the file being served - so the architecture metadata, tokenizer,
tensor layout and the MTP layer stay byte-identical, and only the LoRA-targeted
tensors change: W + (alpha / r) * B @ A, rounded once to bf16.

The GGUF is not laid out exactly like the HF checkpoint. The converter
reorders the value heads of the linear-attention layers: attn_qkv and
attn_gate (in_proj_qkv, in_proj_z) have their rows permuted, ssm_out
(out_proj) its columns. The permutation is not assumed: for every target
tensor it is found by matching the HF base weight to the GGUF base weight, and
the merge stops unless the permuted HF weight equals the GGUF tensor bit for
bit. The same permutation is then applied to the delta.

bf16 has 8 bits of mantissa, so part of a small delta rounds away when it is
added to a large weight. The share lost is measured and logged; if it is
large, the merged model is not the trained model and the unmerged adapter
(Ollama ADAPTER, 08_serve_adapter.py) is the honest comparison.

The new tag gets the student's Modelfile unchanged (projector, RENDERER,
PARSER, sampling PARAMETERs) apart from the weights FROM line.

Usage (Windows side, Ollama running; the merge itself runs in WSL on the CPU):
    python 06_merge_export.py --name t400-r16             # -> qwen3.5-mail:t400-r16
    python 06_merge_export.py --name t400-r16 --no-create # merge only
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "qwen3.5-mail"
# activate sets HSA_ENABLE_DXG_DETECTION and the rocprofiler LD_PRELOAD stub;
# without them ROCm sees no GPU (the merge is CPU-only, but stays uniform)
FT_PYTHON = "source ~/ft-env/bin/activate && python"

# HF module (under model.language_model.layers.N.) -> GGUF tensor (blk.N.)
GGUF_NAMES = {
    "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}
# Training loads the text-only Qwen3_5ForCausalLM, so adapter keys say
# model.layers.N; the checkpoint (with its vision tower) says
# model.language_model.layers.N.
ADAPTER_PREFIX = "model.layers."
CKPT_PREFIX = "model.language_model.layers."


def wsl_path(p):
    """F:\\a\\b -> /mnt/f/a/b"""
    p = os.path.abspath(p)
    return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")


# ---- WSL side -------------------------------------------------------------

def _layout(torch, hf_w, gg_w):
    """How the GGUF stores the HF weight: (None | 'rows' | 'cols', index)."""
    if torch.equal(hf_w, gg_w):
        return None, None
    for axis in ("rows", "cols"):
        a, b = (hf_w, gg_w) if axis == "rows" else (hf_w.T.contiguous(), gg_w.T.contiguous())
        where = {bytes(r.view(torch.int16).numpy()): i for i, r in enumerate(a)}
        idx = [where.get(bytes(r.view(torch.int16).numpy())) for r in b]
        if None not in idx and torch.equal(a[idx], b):
            return axis, torch.tensor(idx)
    raise SystemExit("GGUF tensor is not the HF weight, even up to a row or column order")


def merge(name, blob):
    import numpy as np
    import torch
    import gguf
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    import ftcommon

    run_dir = os.path.join(HERE, "runs", name)
    cfg = json.load(open(os.path.join(run_dir, "adapter_config.json")))
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "plain LoRA only"
    scale = cfg["lora_alpha"] / cfg["r"]
    snap = snapshot_download(ftcommon.MODEL, revision=ftcommon.REVISION, local_files_only=True)
    index = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]

    adapter = {}
    with safe_open(os.path.join(run_dir, "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            base, ab = k.split(".lora_")  # base_model.model.<hf module>.lora_A.weight
            adapter.setdefault(base.removeprefix("base_model.model."), {})[ab[0]] = f.get_tensor(k)

    reader = gguf.GGUFReader(blob)
    tensors = {t.name: t for t in reader.tensors}
    out = os.path.join(run_dir, "merged.gguf")
    t0 = time.perf_counter()
    shutil.copyfile(blob, out)
    print(f"copied base GGUF in {time.perf_counter() - t0:.0f} s", flush=True)

    stats, layouts = {}, {}
    with open(out, "r+b") as fo:
        for module, ab in sorted(adapter.items()):
            assert module.startswith(ADAPTER_PREFIX), f"LoRA outside the decoder layers: {module}"
            layer, kind = module[len(ADAPTER_PREFIX):].split(".", 1)
            t = tensors[f"blk.{layer}.{GGUF_NAMES[kind]}.weight"]
            assert t.tensor_type.name == "BF16", t.name
            hf_name = f"{CKPT_PREFIX}{layer}.{kind}.weight"
            with safe_open(os.path.join(snap, index[hf_name]), "pt") as f:
                hf_w = f.get_tensor(hf_name)
            raw = np.asarray(t.data)
            gg_w = torch.from_numpy(raw.view(np.int16).copy()).view(torch.bfloat16).reshape(hf_w.shape)
            axis, idx = _layout(torch, hf_w, gg_w)
            layouts.setdefault(kind, set()).add(axis or "same")

            delta = scale * (ab["B"].float() @ ab["A"].float())
            if axis == "rows":
                delta = delta[idx]
            elif axis == "cols":
                delta = delta[:, idx]
            new = (gg_w.float() + delta).to(torch.bfloat16)
            applied = new.float() - gg_w.float()
            s = stats.setdefault(kind, {"delta_sq": 0.0, "lost_sq": 0.0, "weight_sq": 0.0,
                                        "changed": 0, "n": 0, "tensors": 0})
            s["delta_sq"] += float(delta.pow(2).sum())
            s["lost_sq"] += float((applied - delta).pow(2).sum())
            s["weight_sq"] += float(gg_w.float().pow(2).sum())
            s["changed"] += int((new != gg_w).sum())
            s["n"] += new.numel()
            s["tensors"] += 1
            data = new.view(torch.int16).numpy().tobytes()
            assert len(data) == t.n_bytes, t.name
            fo.seek(t.data_offset)
            fo.write(data)

    report = {"name": name, "base_blob": blob, "scale": scale, "rank": cfg["r"],
              "minutes": round((time.perf_counter() - t0) / 60, 1), "kinds": {}}
    tot = {"delta_sq": 0.0, "lost_sq": 0.0, "changed": 0, "n": 0}
    for kind, s in stats.items():
        report["kinds"][kind] = {
            "tensors": s["tensors"], "layout": sorted(layouts[kind]),
            "delta_rel_norm": round((s["delta_sq"] / s["weight_sq"]) ** 0.5, 6),
            "delta_lost": round((s["lost_sq"] / s["delta_sq"]) ** 0.5, 4),
            "weights_changed": round(s["changed"] / s["n"], 4)}
        for k in tot:
            tot[k] += s[k]
    report["delta_lost"] = round((tot["lost_sq"] / tot["delta_sq"]) ** 0.5, 4)
    report["weights_changed"] = round(tot["changed"] / tot["n"], 4)
    os.makedirs(os.path.join(HERE, "logs", "merge"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "merge", f"{name}.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)


# ---- Windows side ---------------------------------------------------------

def create(name, keep, tag=True):
    import importlib
    sys.path.insert(0, os.path.join(HERE, "..", "phase10-final-benchmark"))
    import common
    student = importlib.import_module("03_baseline").STUDENT

    modelfile = common.p9._post(common.OLLAMA + "/api/show", {"model": student})["modelfile"]
    froms = [l.split(None, 1)[1].strip() for l in modelfile.splitlines() if l.startswith("FROM ")]
    weights = max(froms, key=os.path.getsize)  # the other FROM is the vision projector
    print(f"{student}: weights {weights}", flush=True)

    cmd = (f"cd {wsl_path(HERE)} && {FT_PYTHON} 06_merge_export.py --merge --name {name} "
           f"--blob {wsl_path(weights)}")
    subprocess.run(["wsl", "-e", "bash", "-lc", cmd], check=True)
    if not tag:
        return

    run_dir = os.path.join(HERE, "runs", name)
    merged = os.path.join(run_dir, "merged.gguf")
    lines = []
    for l in modelfile.splitlines():
        if l.startswith("#"):
            continue
        if l.startswith("FROM ") and l.split(None, 1)[1].strip() == weights:
            l = f"FROM {merged}"
        lines.append(l)
    path = os.path.join(run_dir, "Modelfile")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    subprocess.run(["ollama", "create", f"{TAG}:{name}", "-f", path], check=True)
    print(f"created {TAG}:{name}", flush=True)
    if not keep:
        os.remove(merged)  # Ollama copied it into its blobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="runs/<name>/ from 04_lora_train.py")
    ap.add_argument("--no-create", action="store_true", help="merge only, no Ollama tag")
    ap.add_argument("--keep", action="store_true", help="keep runs/<name>/merged.gguf")
    ap.add_argument("--merge", action="store_true", help=argparse.SUPPRESS)  # WSL side
    ap.add_argument("--blob", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.merge:
        merge(args.name, args.blob)
    else:
        create(args.name, args.keep, tag=not args.no_create)


if __name__ == "__main__":
    main()

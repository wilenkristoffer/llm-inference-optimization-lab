"""
Phase 11, file 0: can the student be trained on this card at all?

Runs in WSL, in ~/ft-env (setup_ft_env.sh). Three checks:

    kernels   Qwen3.5 is a hybrid: 3 of every 4 layers are linear attention
              (Gated DeltaNet), not softmax attention. transformers runs them
              through flash-linear-attention (fla, Triton) when it is
              installed and a pure-PyTorch reference otherwise. fla was
              written for NVIDIA; this runs both on the same random inputs and
              compares output, gradients and time, so we know fla is right on
              gfx1101 before trusting a loss curve to it.
    model     loads the text half of the checkpoint (the vision tower is not
              needed for mail) in bf16 and lists the Linear layers LoRA can
              target, per layer type.
    memory    LoRA forward + backward + optimizer step at growing sequence
              lengths, with and without gradient checkpointing: peak VRAM (the
              process's own, torch.cuda.max_memory_allocated - not the
              board-wide figure), train tokens/s, OOM where it happens.
              Random token ids: speed and memory do not depend on the text.

Results go to logs/preflight.json.

Usage:
    python 00_preflight.py                  # all three
    python 00_preflight.py --only kernels
    python 00_preflight.py --lens 1024 4096 8192
"""

import argparse
import gc
import json
import os
import time

import torch

import ftcommon
from ftcommon import TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "logs", "preflight.json")


def sync():
    torch.cuda.synchronize()


def timed(fn, reps=3):
    fn()
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t) / reps


# ---------------------------------------------------------------- kernels

def check_kernels(T=2048):
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    reference = m.torch_chunk_gated_delta_rule.__wrapped__
    out = {"bound": {}}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        f = getattr(m, name)
        # The wrapper hides what it calls; its closure holds the implementation.
        impl = next((c.cell_contents for c in (f.__closure__ or ())
                     if callable(c.cell_contents) and c.cell_contents is not f.__wrapped__
                     and hasattr(c.cell_contents, "__module__")), None)
        out["bound"][name] = getattr(impl, "__module__", "torch reference")
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
    except Exception as e:
        out["fla"] = f"import failed: {e!r}"
        return out

    # Shapes as the 4B calls it: 32 value heads (keys repeated up to 32), dim 128.
    g = torch.Generator(device="cuda").manual_seed(0)
    B, H, D = 1, 32, 128

    def inputs():
        q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
        a = torch.randn(B, T, H, device="cuda", dtype=torch.float32, generator=g)
        beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16, generator=g)
        decay = -torch.nn.functional.softplus(a) * 0.1
        return [x.requires_grad_() for x in (q, k, v, decay, beta)]

    base = inputs()
    results = {}
    for label, fn in (("torch", reference), ("fla", fla_chunk)):
        xs = [x.detach().clone().requires_grad_() for x in base]
        y, _ = fn(*xs[:3], g=xs[3], beta=xs[4], use_qk_l2norm_in_kernel=True)
        y.float().pow(2).mean().backward()
        results[label] = {"y": y.detach().float(), "grads": [x.grad.float() for x in xs]}

        def step():
            ys, _ = fn(*xs[:3], g=xs[3], beta=xs[4], use_qk_l2norm_in_kernel=True)
            ys.float().pow(2).mean().backward()
        results[label]["ms_fwd_bwd"] = round(1000 * timed(step), 2)

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    t, f = results["torch"], results["fla"]
    out["fla"] = {
        "tokens": T,
        "out_rel_err": round(rel(f["y"], t["y"]), 5),
        "grad_rel_err": {n: round(rel(a, b), 5) for n, a, b in
                         zip(("q", "k", "v", "g", "beta"), f["grads"], t["grads"])},
        "ms_fwd_bwd": {"torch": t["ms_fwd_bwd"], "fla": f["ms_fwd_bwd"]},
    }
    return out


# ---------------------------------------------------------------- model

def check_model():
    t = time.perf_counter()
    model, info = ftcommon.load_model()
    out = {"class": type(model).__name__, "load_s": round(time.perf_counter() - t, 1),
           "missing_keys": len(info["missing_keys"]),
           "unexpected_keys": len(info["unexpected_keys"]),
           "missing_sample": sorted(info["missing_keys"])[:5],
           "unexpected_sample": sorted(info["unexpected_keys"])[:5],
           "weights_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
           "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3)}
    linear = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            layer = int(name.split(".layers.")[1].split(".")[0])
            kind = model.config.layer_types[layer] if hasattr(model.config, "layer_types") \
                else model.config.get_text_config().layer_types[layer]
            linear.setdefault(kind, set()).add(name.rsplit(".", 1)[1])
    out["linear_by_layer_type"] = {k: sorted(v) for k, v in linear.items()}
    found = set().union(*linear.values())
    wanted = [n for names in TARGETS.values() for n in names]
    out["lora_targets"] = [n for n in wanted if n in found]
    out["lora_targets_missing"] = [n for n in wanted if n not in found]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- memory

def check_memory(lens, rank=16):
    # Every position labelled: the worst case. Real episodes label only the
    # assistant turns, so the loss part shrinks further there.
    model, _ = ftcommon.load_model()
    model = ftcommon.prepare(model, rank)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    vocab = model.config.get_text_config().vocab_size
    rows = []
    for ckpt in (True, False):
        if ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        model.train()  # see ftcommon: checkpointing is skipped in eval mode
        for T in lens:
            ids = torch.randint(0, vocab, (1, T), device="cuda")

            def step():
                loss = ftcommon.lm_loss(model, ids, ids)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            row = {"grad_ckpt": ckpt, "tokens": T}
            try:
                s = timed(step, reps=2)
                row.update(ok=True, s_per_step=round(s, 3), train_tok_s=round(T / s),
                           peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            except torch.OutOfMemoryError:
                row.update(ok=False, error="OOM")
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            print(row, flush=True)
            rows.append(row)
            if not row["ok"]:
                break  # longer ones will not fit either
    return {"rank": rank, "targets": [n for names in TARGETS.values() for n in names],
            "loss": "chunked, every position labelled",
            "trainable_m": round(trainable / 1e6, 2), "steps": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["kernels", "model", "memory"])
    ap.add_argument("--lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 12288, 16384])
    args = ap.parse_args()
    assert torch.version.hip, "not a ROCm torch - ft-env was broken by an install"
    report = {"torch": torch.__version__, "hip": torch.version.hip,
              "device": torch.cuda.get_device_name(0),
              "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)}
    if os.path.exists(OUT):
        with open(OUT) as f:
            report = {**json.load(f), **report}
    for name, fn in (("kernels", check_kernels), ("model", check_model),
                     ("memory", lambda: check_memory(args.lens))):
        if args.only in (None, name):
            print(f"== {name}", flush=True)
            report[name] = fn()
            print(json.dumps(report[name], indent=1), flush=True)
            with open(OUT, "w") as f:
                json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()

"""Phase 9 Stage 4: tune the FP16 GEMMs vLLM runs, offline, with PyTorch TunableOp.

Runs INSIDE WSL with the vLLM venv (it imports only torch):
    wsl -d Ubuntu -- bash -lc "source ~/vllm-env/bin/activate && \
        python /mnt/h/.../phase9-optimization/07_tune_gemms.py"

Why offline: vLLM captures CUDA graphs at startup, and TunableOp cannot tune
inside a capture, so the table has to exist before the server starts. With
PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_TUNING=0 the server then only
reads it.

Which shapes: on gfx11, vLLM sends 1-5 token GEMMs to its own skinny kernels and
everything wider to torch.nn.functional.linear (rocm_unquantized_gemm in
vllm/model_executor/layers/utils.py), which is what TunableOp hooks. So every
layer GEMM at each CUDA-graph size of 6+ tokens, plus lm_head at every row count
up to 48 (logits are computed outside the graph, on the sampled rows only, so
their count is not padded). The calls below are F.linear exactly as vLLM makes
them, so the table keys match.

--step times one decode step's GEMMs (every layer with its own weights, so
nothing is served from the 64 MB Infinity Cache, plus lm_head) and exits. Run
it once plain and once with common.TUNABLEOP_ENV set to compare default and
tuned kernels (the env var overrides torch.cuda.tunable.enable(False)).
"""
import argparse
import os
import sys
import time

MODELS = {   # hidden, intermediate, kv_dim, vocab (all three tie lm_head)
    "0.5B": (896, 4864, 128, 151936),
    "1.5B": (1536, 8960, 256, 151936),
    "3B": (2048, 11008, 256, 151936),
}
# Mirrors common.CAPTURE_SIZES (common.py cannot be imported here: Windows paths).
CAPTURE_SIZES = list(range(1, 17)) + [20, 24, 28, 32, 40, 48, 64, 80, 96, 128,
                                      144, 160, 192, 224, 256, 288]
GRAPH_MS = [m for m in CAPTURE_SIZES if m >= 6]
LM_HEAD_MS = sorted(set(range(6, 49)) | set(GRAPH_MS))


def shapes(models):
    for name in models:
        h, inter, kv, vocab = MODELS[name]
        for gemm, k, n, bias in (("qkv", h, h + 2 * kv, True), ("o", h, h, False),
                                 ("gate_up", h, 2 * inter, False),
                                 ("down", inter, h, False)):
            for m in GRAPH_MS:
                yield name, gemm, m, k, n, bias
        for m in LM_HEAD_MS:
            yield name, "lm_head", m, h, vocab, False


def bench(F, torch, x, w, b, reps=30):
    for _ in range(3):
        F.linear(x, w, b)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        F.linear(x, w, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e6


def step_times(torch, F):
    for name in ("1.5B", "3B"):
        h, inter, kv, vocab = MODELS[name]
        layers = {"1.5B": 28, "3B": 36}[name]
        weights = [[(torch.randn(n, k, dtype=torch.float16, device="cuda"),
                     torch.randn(n, dtype=torch.float16, device="cuda") if bias else None)
                    for k, n, bias in ((h, h + 2 * kv, True), (h, h, False),
                                       (h, 2 * inter, False), (inter, h, False))]
                   for _ in range(layers)]
        lm_head = torch.randn(vocab, h, dtype=torch.float16, device="cuda")
        for m in (8, 16, 32):
            xs = {k: torch.randn(m, k, dtype=torch.float16, device="cuda") for k in (h, inter)}

            def step():
                for layer in weights:
                    for w, b in layer:
                        F.linear(xs[w.shape[1]], w, b)
                F.linear(xs[h], lm_head)
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                step()
            torch.cuda.synchronize()
            print(f"{name} m={m}: {(time.perf_counter() - t) / 10 * 1000:.1f} ms per step",
                  flush=True)
        del weights, lm_head
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default=os.path.expanduser("~/tunableop/qwen25.csv"))
    ap.add_argument("--step", action="store_true",
                    help="time one decode step's GEMMs and exit (no tuning)")
    ap.add_argument("--compare", action="store_true",
                    help="also time default vs tuned for the 3B gate_up")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import torch
    import torch.nn.functional as F
    if args.step:
        print("TunableOp", os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "off"), flush=True)
        return step_times(torch, F)
    tun = torch.cuda.tunable
    tun.enable(True)
    tun.tuning_enable(True)
    # Insert the device ordinal (qwen25.csv -> qwen250.csv) the way the
    # PYTORCH_TUNABLEOP_FILENAME reader does, or the server finds no table.
    tun.set_filename(args.out, insert_device_ordinal=True)
    print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0),
          "->", args.out, flush=True)

    todo = list(shapes(args.models))
    t0 = time.time()
    for i, (name, gemm, m, k, n, bias) in enumerate(todo):
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda")
        b = torch.randn(n, dtype=torch.float16, device="cuda") if bias else None
        F.linear(x, w, b)              # first call with a new shape tunes it
        torch.cuda.synchronize()
        if i % 25 == 0 or i == len(todo) - 1:
            print(f"  {i + 1}/{len(todo)} {name} {gemm} m={m}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    print(f"tuned {len(todo)} shapes (the table is written at exit) in {time.time() - t0:.0f} s", flush=True)

    if args.compare:
        print("\n3B gate_up (2048 -> 22016), us per call: default vs tuned")
        tun.tuning_enable(False)
        w = torch.randn(22016, 2048, dtype=torch.float16, device="cuda")
        for m in (6, 8, 16, 32, 64, 128):
            x = torch.randn(m, 2048, dtype=torch.float16, device="cuda")
            tun.enable(True)
            tuned = bench(F, torch, x, w, None)
            tun.enable(False)
            default = bench(F, torch, x, w, None)
            print(f"  m={m:<4} default {default:7.0f}   tuned {tuned:7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

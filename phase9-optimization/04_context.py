"""
Phase 9, file 4: decode speed and VRAM against context depth, and what KV
cache quantization buys (Stage 2, Parts A and B speed).

Every generated token reads all the weights and the whole KV cache so far, so
the Phase 1 bandwidth arithmetic predicts

    tok/s(ctx) ~= tok/s(0) * W / (W + b * ctx)

with W the weight bytes and b the KV bytes per token. --report puts that
prediction beside each measurement, with tok/s(0) extrapolated from the
shortest depth.

llama.cpp: llama-server directly, not Ollama - the same engine, but the KV
type is a server flag (-ctk/-ctv) instead of a service-wide environment
variable, and the direct server is the faster baseline (Stage 0). One server
per (model, depth, KV type), with -c = depth + 512, so the KV allocation (made
up front for the whole -c) is what the VRAM column measures. The loop order is
model -> depth -> KV type, so the three types at one depth run back to back,
and depth 512 f16 runs again at the end of each model to show drift.

VRAM: the server process's Dedicated Usage counter, plus Shared Usage. When
dedicated VRAM runs out, Windows pages GPU memory into system RAM instead of
failing the allocation (Phase 7), so a config "fits" only while shared stays
near zero. A paged config runs once, not twice: it is slow, and the point is
that it happens.

vLLM: Qwen2.5-1.5B FP16 at the same depths and the same prompt texts, one
server (max_len 32768) with prefix caching off so every rep prefills. Its KV is
FP16; the KV-type sweep is llama.cpp's only. No VRAM column: the server lives in WSL, outside the
Windows per-process counter.

Usage:
    python 04_context.py                          # full run, ~40 min
    python 04_context.py --models qwen2.5:1.5b --depths 512,8192 --no-vllm
    python 04_context.py --report [--group p9-context-...]
"""

import argparse
import os
import statistics
from collections import defaultdict
from datetime import datetime

import common

# KV elements per token: layers x 2 (K and V) x kv heads x head dim.
KV_ELEMS = {
    "qwen2.5:1.5b": 28 * 2 * 2 * 128,
    "qwen2.5:14b": 48 * 2 * 8 * 128,
}
# Bytes per element: q8_0 is 32 int8 + one fp16 scale per block, q4_0 is
# 32 nibbles + one fp16 scale.
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
MODELS = list(KV_ELEMS)
DEPTHS = [512, 2048, 8192, 16384, 31744]
KV_TYPES = list(KV_BYTES)
GEN = 128
REPS = 2
PAGED_MB = 500          # shared usage above this = the server is paging
VLLM_MODEL = "qwen2.5-1.5b-instruct"
VLLM_WEIGHTS = 1_543_714_304 * 2       # Qwen2.5-1.5B parameters, FP16
VLLM_KV = 28 * 2 * 2 * 128 * 2


def notes(depth, kv, phase, paged):
    return f"prompt=docs-{depth} depth={depth} kv={kv} phase={phase} paged={int(paged)}"


def run_llama(conn, group, version, model, depth, kv, prompt, phase, reps):
    ctx = min(depth + 512, 32768)
    with common.LlamaServer(model, ctx=ctx, extra=["-ctk", kv, "-ctv", kv],
                            label=f"c{ctx}-{kv}") as s:
        s.chat("Say hello.", max_tokens=8)          # warm-up, not stored
        rep = 0
        while rep < reps:
            r = s.chat(prompt, max_tokens=GEN, timeout=3600, ignore_eos=True)
            # Read after a full-depth request, when every buffer has been touched.
            ded = common.process_vram_mb(s.proc.pid)
            shared = common.process_vram_mb(s.proc.pid, "Shared Usage")
            paged = shared is not None and shared > PAGED_MB
            common.storage.record_run(
                conn, run_group=group, trial_index=rep, backend="llama.cpp",
                backend_version=version, model=model, quantization="Q4_K_M",
                context_length=ctx, temperature=0, seed=42, streaming=0,
                prompt=f"<docs corpus cut to {depth} tokens>", response=r["response"],
                input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
                prefill_ms=r["prefill_ms"],
                prefill_tokens_per_s=r["input_tokens"] / r["prefill_ms"] * 1000,
                decode_ms=r["decode_ms"], client_total_ms=r["client_total_ms"],
                decode_tokens_per_s=r["decode_tokens_per_s"], kv_cache_type=kv,
                server_vram_mb=ded, server_shared_mb=shared, server_num_parallel=1,
                gpu_library="ROCm", hardware_id=common.HARDWARE_ID,
                notes=notes(depth, kv, phase, paged))
            print(f"  {model} depth {depth} {kv} {phase} rep{rep}: "
                  f"prefill {r['input_tokens']} tok in {r['prefill_ms'] / 1000:.1f} s, "
                  f"decode {r['decode_tokens_per_s']:.1f} tok/s, "
                  f"VRAM {ded or 0:.0f} MB + shared {shared or 0:.0f} MB"
                  + ("  PAGED" if paged else ""), flush=True)
            rep += 1
            if paged:
                break


def run_vllm(conn, group, prompts, reps):
    version = common.vllm_version()
    with common.VllmServer(VLLM_MODEL, max_len=32768,
                           extra=["--no-enable-prefix-caching"]) as s:
        s.chat("Say hello.", max_tokens=8)
        for rep in range(reps):
            for depth, prompt in prompts.items():
                r = s.chat(prompt, max_tokens=GEN, ignore_eos=True)
                common.storage.record_run(
                    conn, run_group=group, trial_index=rep, backend="vllm",
                    backend_version=version, model=VLLM_MODEL, quantization="FP16",
                    context_length=32768, temperature=0, seed=42, streaming=1,
                    prompt=f"<docs corpus cut to {depth} tokens>",
                    response=r["response"], input_tokens=r["input_tokens"],
                    output_tokens=r["output_tokens"], ttft_ms=r["ttft_ms"],
                    decode_ms=r["decode_ms"], client_total_ms=r["client_total_ms"],
                    decode_tokens_per_s=r["decode_tokens_per_s"], kv_cache_type="f16",
                    server_num_parallel=1, gpu_library="ROCm",
                    hardware_id=common.HARDWARE_ID,
                    notes=notes(depth, "f16", "vllm", False))
                print(f"  vLLM 1.5B depth {depth} rep{rep}: TTFT {r['ttft_ms'] / 1000:.2f} s, "
                      f"decode {r['decode_tokens_per_s']:.1f} tok/s", flush=True)


def sweep(models, depths, reps, vllm):
    conn = common.storage.connect()
    group = "p9-context-" + datetime.now().strftime("%Y%m%d-%H%M")
    version = common.llama_version()
    print(f"run_group {group}, {version}")
    # The prompts, built once with a small server for its tokenizer.
    with common.LlamaServer("qwen2.5:1.5b", label="tokenizer"):
        prompts = {d: common.long_prompt(d, "docs") for d in depths}
    for model in models:
        for depth in depths:
            for kv in KV_TYPES:
                run_llama(conn, group, version, model, depth, kv, prompts[depth], "sweep", reps)
        run_llama(conn, group, version, model, depths[0], "f16", prompts[depths[0]], "end", reps)
    if vllm:
        run_vllm(conn, group, prompts, reps)
    print(f"done: {group}")


def field(r, key):
    return dict(kv.split("=") for kv in r["notes"].split())[key]


def predict(rows_by_depth, weights, kv_bytes):
    """Prediction per depth, anchored on the shallowest depth's measurement.
    The KV read during decode averages over the generated tokens, so ctx is
    taken at the middle of the generation."""
    d0 = min(rows_by_depth)
    ctx0 = statistics.median(r["input_tokens"] for r in rows_by_depth[d0]) + GEN / 2
    tps0 = statistics.median(r["decode_tokens_per_s"] for r in rows_by_depth[d0])
    base = tps0 * (weights + kv_bytes * ctx0) / weights
    out = {}
    for d, g in rows_by_depth.items():
        ctx = statistics.median(r["input_tokens"] for r in g) + GEN / 2
        out[d] = base * weights / (weights + kv_bytes * ctx)
    return out


def slope_kb(points):
    """Least-squares slope of VRAM MB against context tokens, in KB/token."""
    xs, ys = zip(*points)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return (sum((x - mx) * (y - my) for x, y in points)
            / sum((x - mx) ** 2 for x in xs) * 2**20 / 1000)


def report(group):
    conn = common.storage.connect()
    if not group:
        group = conn.execute("SELECT run_group FROM inference_runs WHERE run_group "
                             "LIKE 'p9-context-%' ORDER BY id DESC LIMIT 1").fetchone()[0]
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM inference_runs WHERE run_group = ? ORDER BY id", (group,))]
    print(f"run_group {group}: {len(rows)} rows\n")

    for model in [m for m in MODELS if any(r["model"] == m for r in rows)]:
        weights = os.path.getsize(os.path.join(common.BLOBS, common.GGUF[model]))
        mine = [r for r in rows if r["model"] == model]
        f16_at = {}
        print(f"== {model}: weights {weights / 1e9:.2f} GB, KV "
              + ", ".join(f"{k} {KV_ELEMS[model] * b / 1000:.1f} KB/token"
                          for k, b in KV_BYTES.items()))
        start = [r["decode_tokens_per_s"] for r in mine
                 if field(r, "phase") == "sweep" and r["kv_cache_type"] == "f16"
                 and int(field(r, "depth")) == min(int(field(x, "depth")) for x in mine)]
        end = [r["decode_tokens_per_s"] for r in mine if field(r, "phase") == "end"]
        if start and end:
            print(f"  drift, shallowest f16 start -> end: "
                  f"{statistics.median(end) / statistics.median(start) - 1:+.1%}")
        for kv in KV_TYPES:
            by_depth = defaultdict(list)
            for r in mine:
                if r["kv_cache_type"] == kv and field(r, "phase") == "sweep":
                    by_depth[int(field(r, "depth"))].append(r)
            if not by_depth:
                continue
            fitted = {d: g for d, g in by_depth.items() if field(g[0], "paged") == "0"}
            pred = predict(fitted, weights, KV_ELEMS[model] * KV_BYTES[kv]) if fitted else {}
            vram = [(r["context_length"], r["server_vram_mb"]) for g in fitted.values()
                    for r in g if r["server_vram_mb"]]
            slope = f", VRAM slope {slope_kb(vram):.1f} KB/token" if len(
                {c for c, _ in vram}) > 1 else ""
            print(f"  KV {kv}{slope}")
            print(f"    {'depth':>6} {'input':>6} {'prefill/s':>9} {'decode':>7} "
                  f"{'vs first':>8} {'predicted':>9} {'vs f16':>7} {'VRAM MB':>8} {'shared':>7}")
            first = None
            for d in sorted(by_depth):
                g = by_depth[d]
                tps = statistics.median(r["decode_tokens_per_s"] for r in g)
                first = first or tps
                if kv == "f16":
                    f16_at[d] = tps
                vs16 = f"{tps / f16_at[d] - 1:+6.0%}" if d in f16_at and kv != "f16" else ""
                p = f"{pred[d]:9.1f}" if d in pred else f"{'paged':>9}"
                print(f"    {d:>6} {g[0]['input_tokens']:>6} "
                      f"{statistics.median(r['prefill_tokens_per_s'] for r in g):9.0f} "
                      f"{tps:7.1f} {tps / first - 1:+8.0%} {p} {vs16:>7} "
                      f"{g[0]['server_vram_mb'] or 0:8.0f} {g[0]['server_shared_mb'] or 0:7.0f}")
        print()

    v = [r for r in rows if r["backend"] == "vllm"]
    if v:
        by_depth = defaultdict(list)
        for r in v:
            by_depth[int(field(r, "depth"))].append(r)
        pred = predict(by_depth, VLLM_WEIGHTS, VLLM_KV)
        print(f"== vLLM {VLLM_MODEL} FP16: weights {VLLM_WEIGHTS / 1e9:.2f} GB, "
              f"KV {VLLM_KV / 1000:.1f} KB/token")
        print(f"    {'depth':>6} {'input':>6} {'TTFT s':>7} {'decode':>7} {'vs first':>8} "
              f"{'predicted':>9}")
        first = None
        for d in sorted(by_depth):
            g = by_depth[d]
            tps = statistics.median(r["decode_tokens_per_s"] for r in g)
            first = first or tps
            print(f"    {d:>6} {g[0]['input_tokens']:>6} "
                  f"{statistics.median(r['ttft_ms'] for r in g) / 1000:7.2f} {tps:7.1f} "
                  f"{tps / first - 1:+8.0%} {pred[d]:9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--group")
    a = ap.parse_args()
    if a.report:
        report(a.group)
    else:
        sweep(a.models.split(","), [int(d) for d in a.depths.split(",")], a.reps,
              not a.no_vllm)


if __name__ == "__main__":
    main()

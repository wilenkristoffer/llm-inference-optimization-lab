"""
Phase 6, file 1: what each quantization level costs in VRAM and buys in speed.

For every level: VRAM footprint, decode rate, prefill rate - measured the way
Phases 1-5 taught, because the differences here are small enough that every
one of those lessons matters.

1. CLEAN FOOTPRINTS. Evict everything, wait for VRAM to settle, then load. A
   model left resident by the previous level would be counted as desktop.
2. ONE SLOT. Each Ollama slot preallocates its own FP16 KV cache, whatever the
   weight quantization - at 8 slots it was ~40% of the 1.5B Q4's footprint.
   The script refuses to run at anything but 1 (and offers to restart).
3. INTERLEAVED ROUNDS. Between-session drift is ~4%, the size of some of the
   differences expected between adjacent levels. So the ladder runs in
   rounds - every level once per round, in a freshly shuffled order - and
   drift lands on every level alike instead of on whichever ran last.
4. FIXED DECODE WORK. A long-form prompt with num_predict pinned, so every
   level generates the same number of tokens (checked, not assumed - a level
   that stops early is flagged).
5. UNIQUE PREFILL PROMPTS. Repeating one would hit the prefix cache and
   measure caching, not prefill (Phase 5).

Effective bandwidth is weight-file size x decode rate: the Phase 1 model says
decode streams every weight once per token, so a smaller file should decode
proportionally faster. Where it does not, the difference is the cost of
dequantizing - which is the interesting part.

Usage:
    python 01_quant_ladder.py                      # the 3B ladder, 3 rounds
    python 01_quant_ladder.py --models qwen2.5:7b,qwen2.5:7b-instruct-q8_0
    python 01_quant_ladder.py --rounds 1 --trials 1   # smoke test
"""

import argparse
import importlib.util
import json
import os
import random
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
sampler_mod = _load("sampler", "../phase1-ollama-baseline/06_hardware_sampler.py")

HARDWARE_ID = "Ryzen 5600X / RX 7800 XT / 32GB DDR4"
PEAK_BANDWIDTH_GBS = 624
LAB_API = "http://127.0.0.1:8010"

LADDER_3B = [
    "qwen2.5:3b-instruct-q2_K",
    "qwen2.5:3b-instruct-q3_K_M",
    "qwen2.5:3b-instruct-q4_0",
    "qwen2.5:3b",                      # = q4_K_M, Ollama's default tag
    "qwen2.5:3b-instruct-q5_K_M",
    "qwen2.5:3b-instruct-q6_K",
    "qwen2.5:3b-instruct-q8_0",
    "qwen2.5:3b-instruct-fp16",
]

# Long-form on purpose: the model should still be writing when num_predict
# cuts it off, so every level does identical decode work.
DECODE_PROMPT = "Write a 500-word description of a coastal lagoon at dawn."
DECODE_TOKENS = 256

# Same filler vocabulary idea as Phase 5 file 02: ordinary prose, so the token
# count means what it would for a real prompt.
VOCAB = ("the quick brown fox jumps over a lazy dog while morning light falls "
         "across quiet fields and rivers carry water toward distant hills where "
         "old stone walls divide green pasture from darker woodland").split()
PREFILL_WORDS = 1200


def prefill_prompt(seed):
    rng = random.Random(seed)
    filler = " ".join(rng.choice(VOCAB) for _ in range(PREFILL_WORDS))
    return f"{filler}\n\nSummarise the text above in one sentence."


def ensure_one_slot(model):
    """Load the model once and check the runner's -np. Restart via the lab API
    if it is not 1 - footprints at more slots include idle KV cache."""
    trials.run_once(model, 4096, prompt="hi", num_predict=1)
    runner = trials.runner_config(model) or {}
    if runner.get("num_parallel") == 1:
        return
    print(f"Ollama runner has -np {runner.get('num_parallel')}; footprints need 1.")
    try:
        request = urllib.request.Request(
            f"{LAB_API}/api/servers/ollama/start?num_parallel=1", data=b"",
            method="POST")
        urllib.request.urlopen(request, timeout=60).read()
    except OSError:
        sys.exit("Could not reach the lab API to restart Ollama. Start "
                 "phase4-observability/api.py, or restart Ollama with "
                 "OLLAMA_NUM_PARALLEL=1, and run again.")
    for _ in range(60):
        try:
            trials._ollama("/api/version")
            break
        except OSError:
            time.sleep(1)
    trials.run_once(model, 4096, prompt="hi", num_predict=1)
    runner = trials.runner_config(model) or {}
    if runner.get("num_parallel") != 1:
        sys.exit(f"Still -np {runner.get('num_parallel')} after restart. Stopping.")
    print("restarted Ollama with 1 slot.")


def file_mb(model):
    """Size of the TEXT weights - the first FROM in the Modelfile - which is
    what decode streams. Not /api/tags' size: for a multimodal model that also
    counts the vision projector (qwen3.5:4b-bf16: 8,265 MiB text + 644 MiB
    vision), which overstated its bandwidth by ~8%."""
    modelfile = trials._ollama("/api/show", {"model": model}).get("modelfile", "")
    for line in modelfile.splitlines():
        if line.startswith("FROM "):
            path = line.split(None, 1)[1].strip()
            if os.path.exists(path):
                return os.path.getsize(path) / 2**20
            break
    return None


def describe(model):
    show = trials._ollama("/api/show", {"model": model})
    details = show.get("details", {})
    digest = next((m.get("digest") for m in trials._ollama("/api/tags")["models"]
                   if m.get("name") == model or m.get("model") == model), None)
    return {"quantization": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
            "model_digest": digest}


def measure_level(conn, model, meta, run_group, round_no, n_trials, seed_base,
                  version, notes):
    """One level, one round: evict, settle, load, measure. Returns a summary."""
    trials.evict_all()
    hw = sampler_mod.HardwareSampler(interval=0.15, settle=True)
    baseline = hw.baseline_vram_mb()

    # Warm-up loads the model and is discarded. The footprint reported is the
    # peak DURING the trials, minus the baseline: what the model holds while
    # serving, which is the number that decides whether it fits.
    trials.run_once(model, 4096, prompt=DECODE_PROMPT, num_predict=16)
    server_vram = trials.server_vram_mb(model)
    runner = trials.runner_config(model) or {}

    decode, prefill, short, aborted = [], [], 0, 0
    with hw:
        for t in range(n_trials):
            t0 = time.perf_counter()
            d = trials.run_once(model, 4096, prompt=DECODE_PROMPT,
                                num_predict=DECODE_TOKENS)
            t1 = time.perf_counter()
            p_prompt = prefill_prompt(seed_base + t)
            p = trials.run_once(model, 4096, prompt=p_prompt, num_predict=1)
            t2 = time.perf_counter()

            if d["error"]:
                aborted += 1
            elif d["output_tokens"] < DECODE_TOKENS:
                short += 1
            if d["decode_tok_s"] is not None:
                decode.append(d["decode_tok_s"])
            if p["prefill_tok_s"]:
                prefill.append(p["prefill_tok_s"])

            for kind, m, a, b, prompt in (("decode", d, t0, t1, DECODE_PROMPT),
                                          ("prefill", p, t1, t2, p_prompt)):
                s = hw.summary(a, b)
                ram_gb = s.get("system_ram_used_gb_peak")
                storage.record_run(
                    conn, run_group=run_group, trial_index=round_no * 100 + t,
                    backend="ollama", backend_version=version,
                    model=model, model_digest=meta["model_digest"],
                    quantization=meta["quantization"], context_length=4096,
                    temperature=0.0, seed=42, streaming=1,
                    prompt=prompt, response=m["response"],
                    input_tokens=m["input_tokens"],
                    cached_input_tokens=m["cached_input_tokens"],
                    output_tokens=m["output_tokens"], total_tokens=m["total_tokens"],
                    ttft_ms=m["ttft_ms"], client_total_ms=m["client_total_ms"],
                    load_ms=m["load_ms"], prefill_ms=m["prefill_ms"],
                    decode_ms=m["decode_ms"], server_total_ms=m["server_total_ms"],
                    decode_tokens_per_s=m["decode_tok_s"] if kind == "decode" else None,
                    prefill_tokens_per_s=m["prefill_tok_s"] if kind == "prefill" else None,
                    gpu_percent=s.get("gpu_percent_mean"),
                    gpu_clock_mhz=s.get("gpu_clock_mhz_mean"),
                    gpu_power_w=s.get("gpu_power_w_mean"),
                    gpu_temp_c=s.get("gpu_temp_c_peak"),
                    gpu_vram_used_mb=s.get("gpu_vram_used_mb_peak"),
                    gpu_vram_baseline_mb=baseline,
                    gpu_vram_baseline_clean=1 if baseline is not None else None,
                    server_vram_mb=server_vram,
                    server_num_parallel=runner.get("num_parallel"),
                    cpu_percent=s.get("cpu_percent_mean"),
                    system_ram_used_mb=(ram_gb * 1024) if ram_gb else None,
                    hardware_id=HARDWARE_ID,
                    notes=f"phase6 {kind} | round {round_no + 1}"
                          + (f" | ABORTED: {m['error']}" if m["error"] else "")
                          + (f" | {notes}" if notes else ""),
                )

    peak = hw.summary().get("gpu_vram_used_mb_peak")
    return {
        "footprint": (peak - baseline) if peak is not None and baseline is not None else None,
        "server_vram": server_vram,
        "decode": statistics.median(decode) if decode else None,
        "prefill": statistics.median(prefill) if prefill else None,
        "short": short,
        "aborted": aborted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(LADDER_3B))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3,
                        help="decode + prefill pairs per level per round")
    parser.add_argument("--seed", type=int, default=6,
                        help="seeds the round order, recorded so it can be replayed")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    available = {m["name"] for m in trials._ollama("/api/tags")["models"]}
    missing = [m for m in models if m not in available and f"{m}:latest" not in available]
    if missing:
        sys.exit(f"not pulled yet: {', '.join(missing)}")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=1.5):
            sys.exit("vLLM is running and holds VRAM. Stop it first.")
    except OSError:
        pass
    if sampler_mod.read_sensors() is None:
        sys.exit("LibreHardwareMonitor is not reachable - footprints would be NULL. "
                 "Start it elevated with the web server on 8085.")

    ensure_one_slot(models[0])
    conn = storage.connect()
    version = trials._ollama("/api/version").get("version")
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    metas = {m: describe(m) for m in models}
    sizes = {m: file_mb(m) for m in models}

    rng = random.Random(args.seed)
    results = {m: [] for m in models}
    for r in range(args.rounds):
        order = models[:]
        rng.shuffle(order)
        print(f"\nround {r + 1}/{args.rounds}: {', '.join(metas[m]['quantization'] or m for m in order)}")
        for m in order:
            run_group = f"quant|{m}|{stamp}"
            res = measure_level(conn, m, metas[m], run_group, r, args.trials,
                                seed_base=1000 * r + 10 * models.index(m),
                                version=version, notes=args.notes)
            results[m].append(res)
            fp = "-" if res["footprint"] is None else f"{res['footprint']:,.0f} MB"
            dec = "     -" if res["decode"] is None else f"{res['decode']:6.1f}"
            print(f"  {m:<30} {fp:>10}  decode {dec} tok/s  "
                  f"prefill {res['prefill'] or 0:7.0f} tok/s"
                  + (f"  ({res['short']} SHORT outputs)" if res["short"] else "")
                  + (f"  ({res['aborted']} ABORTED by Ollama)" if res["aborted"] else ""))

    print(f"\n{'level':<13} {'file MB':>8} {'VRAM MB':>8} {'claimed':>8} "
          f"{'decode':>8} {'spread':>7} {'prefill':>8} {'GB/s':>6} {'% peak':>7}")
    print("-" * 87)
    ordered = sorted(models, key=lambda m: sizes[m] or 0)
    for m in ordered:
        rs = results[m]
        dec = [x["decode"] for x in rs if x["decode"] is not None]
        if not dec:
            print(f"{metas[m]['parameter_size']} {metas[m]['quantization']:<8} "
                  f"no valid decode - every generation aborted by Ollama")
            continue
        med = statistics.median(dec)
        spread = (max(dec) - min(dec)) / med if med else 0
        fps = [x["footprint"] for x in rs if x["footprint"] is not None]
        pre = [x["prefill"] for x in rs if x["prefill"]]
        bw = (sizes[m] / 1024) * med if sizes[m] else None
        label = f"{metas[m]['parameter_size'] or '?'} {metas[m]['quantization'] or '?'}"
        print(f"{label:<13} {sizes[m] or 0:>8,.0f} "
              f"{statistics.median(fps) if fps else 0:>8,.0f} "
              f"{rs[0]['server_vram'] or 0:>8,.0f} {med:>8.1f} {spread:>6.1%} "
              f"{statistics.median(pre) if pre else 0:>8,.0f} "
              f"{bw or 0:>6.0f} {(bw or 0) / PEAK_BANDWIDTH_GBS:>6.0%}")
    print(f"\nspread = range of the per-round medians. round order seed {args.seed}.")
    print("SHORT = the level ended its answer before num_predict. Decode rate is "
          "per token, so it\nstays comparable; the shorter answer itself is a "
          "behavioural difference worth noting.")
    print(f"stored under quant|<model>|{stamp}")


if __name__ == "__main__":
    main()

"""
Phase 7 helper: measure one (model, placement) configuration on a side server.

Shared by files 03-05 so CPU vs GPU, the offload sweep and the backend swap
are measured identically. Each call:

1. evicts, waits for VRAM to settle, loads with the requested options;
2. reads placement from the server's own log ("offloaded N/M layers to GPU")
   and /api/ps (size vs size_vram) - requested and actual are both kept;
3. runs decode and prefill trials under the hardware sampler, so GPU load
   during a "CPU" run is visible rather than assumed to be zero;
4. stores every trial in inference_runs.

Decode uses a long-form prompt with num_predict pinned. A short answer cannot
be timed: "Count to five." -> "Five." is 3 tokens, and Ollama reported 75 tok/s
for what the client clock measured at 36 (Phase 7, file 3 notes).
"""

import importlib.util
import os
import random
import statistics
import time


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
MAIN_OLLAMA = "http://127.0.0.1:11434"

# LibreHardwareMonitor refreshes its sensors once a second (measured: values
# changed at 1.02, 2.07, 3.02, ... s). A shorter window reads one or two stale
# values: a 0.5 s GPU decode showed "3% GPU load" where Phase 1's sustained
# runs read 100%. Load, clock and power are stored only for windows long
# enough to hold ~3 fresh readings; otherwise NULL, which is the truth.
MIN_RATE_WINDOW_S = 3.0

DECODE_PROMPT = "Write a 500-word description of a coastal lagoon at dawn."

VOCAB = ("the quick brown fox jumps over a lazy dog while morning light falls "
         "across quiet fields and rivers carry water toward distant hills where "
         "old stone walls divide green pasture from darker woodland").split()


def prefill_prompt(seed, words):
    """Unique per trial: a repeated prompt hits the prefix cache (Phase 5)."""
    rng = random.Random(seed)
    filler = " ".join(rng.choice(VOCAB) for _ in range(words))
    return f"{filler}\n\nSummarise the text above in one sentence."


def use_server(base):
    """Point the shared trials module at a server (side or main)."""
    trials.OLLAMA_URL = base + "/api/generate"
    trials.OLLAMA_BASE = base


def evict_main():
    """The user's main Ollama must not hold VRAM during a measurement."""
    use_server(MAIN_OLLAMA)
    try:
        trials.evict_all()
    except OSError:
        pass        # not running is fine


def latest_ceilings(conn):
    """Measured ceilings from 02_ceilings.py, for efficiency figures. Falls back
    to None (printed as '-') rather than to a datasheet number."""
    def best(probe, where):
        row = conn.execute(
            f"SELECT MAX(best) FROM hardware_ceilings WHERE probe = ? AND {where}",
            (probe,)).fetchone()
        return row[0] if row else None
    return {"vram": best("vram_read", "bytes >= 134217728"),
            "ram": best("cpu_read", "environment LIKE '%threads=6%'")}


def weights_gb(model):
    """Text-weights file size (Phase 6 file 01's method): what decode streams."""
    modelfile = trials._ollama("/api/show", {"model": model}).get("modelfile", "")
    for line in modelfile.splitlines():
        if line.startswith("FROM "):
            path = line.split(None, 1)[1].strip()
            if os.path.exists(path):
                return os.path.getsize(path) / 2**30
            break
    return None


def measure(conn, server, model, *, label, run_group, round_no, n_trials,
            decode_tokens, prefill_words, seed_base, num_gpu=None, num_thread=None,
            version=None, notes=None):
    """Measure one configuration. Returns a summary dict."""
    use_server(server.base)
    trials.evict_all()
    hw = sampler_mod.HardwareSampler(interval=0.2, settle=True)
    baseline = hw.baseline_vram_mb()

    options = {}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    if num_thread is not None:
        options["num_thread"] = num_thread

    before = server.offload_count()
    t_load = time.perf_counter()
    trials.run_once(model, 4096, prompt=DECODE_PROMPT, num_predict=8,
                    extra_options=options)
    load_s = time.perf_counter() - t_load
    placed = server.offload_line(since=before)  # (on_gpu, total), this load only
    ps = next((m for m in trials._ollama("/api/ps")["models"]
               if m.get("name") == model or m.get("model") == model), {})
    size_mb = ps.get("size", 0) / 2**20 or None
    vram_mb = ps.get("size_vram", 0) / 2**20 if ps else None
    # No placement line on the CPU build, and none when -ngl is 0: both mean
    # zero layers on the GPU. Anything else missing stays NULL.
    if placed is None and (server.library == "cpu" or num_gpu == 0):
        placed = (0, None)
    layers = placed[0] if placed else None

    decode, prefill, gpu_dec, gpu_pre, cpu_dec, cpu_pre, short = [], [], [], [], [], [], 0
    with hw:
        for t in range(n_trials):
            t0 = time.perf_counter()
            d = trials.run_once(model, 4096, prompt=DECODE_PROMPT,
                                num_predict=decode_tokens, extra_options=options)
            t1 = time.perf_counter()
            p_prompt = prefill_prompt(seed_base + t, prefill_words)
            p = trials.run_once(model, 4096, prompt=p_prompt, num_predict=1,
                                extra_options=options)
            t2 = time.perf_counter()

            if d["output_tokens"] < decode_tokens:
                short += 1
            if d["decode_tok_s"] is not None:
                decode.append(d["decode_tok_s"])
            if p["prefill_tok_s"]:
                prefill.append(p["prefill_tok_s"])

            for kind, m, a, b, prompt in (("decode", d, t0, t1, DECODE_PROMPT),
                                          ("prefill", p, t1, t2, p_prompt)):
                s = hw.summary(a, b)
                if b - a < MIN_RATE_WINDOW_S:
                    for k in ("gpu_percent_mean", "gpu_clock_mhz_mean",
                              "gpu_power_w_mean", "cpu_percent_mean"):
                        s.pop(k, None)
                (gpu_dec if kind == "decode" else gpu_pre).append(s.get("gpu_percent_mean"))
                (cpu_dec if kind == "decode" else cpu_pre).append(s.get("cpu_percent_mean"))
                ram_gb = s.get("system_ram_used_gb_peak")
                storage.record_run(
                    conn, run_group=run_group, trial_index=round_no * 100 + t,
                    backend="ollama", backend_version=version, model=model,
                    context_length=4096, temperature=0.0, seed=42, streaming=1,
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
                    server_vram_mb=vram_mb, server_size_mb=size_mb,
                    server_num_parallel=server.num_parallel,
                    num_gpu_layers=layers, gpu_library=server.library,
                    num_thread=num_thread,
                    cpu_percent=s.get("cpu_percent_mean"),
                    system_ram_used_mb=(ram_gb * 1024) if ram_gb else None,
                    hardware_id=HARDWARE_ID,
                    notes=f"phase7 {label} {kind} | round {round_no + 1}"
                          + (f" | ERROR: {m['error']}" if m["error"] else "")
                          + (f" | {notes}" if notes else ""),
                )

    peak = hw.summary().get("gpu_vram_used_mb_peak")

    def med(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    return {"decode": med(decode), "prefill": med(prefill),
            "gpu_load_decode": med(gpu_dec), "gpu_load_prefill": med(gpu_pre),
            "cpu_load_decode": med(cpu_dec), "cpu_load_prefill": med(cpu_pre),
            "layers": placed, "size_mb": size_mb, "vram_mb": vram_mb,
            "footprint": (peak - baseline) if peak is not None and baseline is not None else None,
            "load_s": load_s, "short": short}

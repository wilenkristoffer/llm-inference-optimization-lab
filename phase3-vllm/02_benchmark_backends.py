"""
Phase 3, file 2: benchmark both backends and persist the results.

File 1 printed to a terminal and the numbers vanished. This runs the same
comparison through the Phase 1 harness so every trial becomes a row in lab.db
with hardware telemetry attached - which is what Phase 10's cross-backend
comparison will query.

Two things are read from the servers rather than typed in, for the same reason
the Ollama harness reads model metadata: a mislabelled row makes a comparison
wrong in a way that is very hard to spot later.

    context_length   Ollama: /api/ps    vLLM: /v1/models max_model_len
    version          Ollama: /api/version
                     vLLM:   system_fingerprint on a response

One field genuinely cannot be read: vLLM does not report the precision it
loaded. It comes from --quantization, defaulting to FP16 because that is what
the server was started with. Operator-supplied, and flagged as such.

IMPORTANT: run both backends in the same session, interleaved if possible.
Between-session drift on this machine is ~4%, which is larger than some of the
differences we are trying to measure.

Usage:
    python 02_benchmark_backends.py
    python 02_benchmark_backends.py --backend vllm --trials 7
    python 02_benchmark_backends.py --history
"""

import argparse
import importlib.util
import json
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


client = _load("client", "01_openai_client.py")
storage = _load("storage", "../phase1-ollama-baseline/04_storage.py")
sampler_mod = _load("sampler", "../phase1-ollama-baseline/06_hardware_sampler.py")
trials_mod = _load("trials", "../phase1-ollama-baseline/03_repeated_trials.py")

HARDWARE_ID = "Ryzen 5600X / RX 7800 XT / 32GB DDR4"


def get_json(url, payload=None, timeout=10):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


# Backends spell the same format differently: Ollama says "F16", vLLM is told
# "float16" and we label it "FP16". Left alone, a GROUP BY quantization in
# Phase 10 splits one format across several rows and misreports it. Normalise
# on write to a canonical label.
QUANT_ALIASES = {
    "f16": "FP16", "fp16": "FP16", "float16": "FP16",
    "bf16": "BF16", "bfloat16": "BF16",
    "f32": "FP32", "fp32": "FP32", "float32": "FP32",
}


def normalise_quant(value):
    if not value:
        return None
    return QUANT_ALIASES.get(value.strip().lower(), value.strip())


def describe(name, cfg, quantization_hint):
    """
    Whatever each server will actually tell us about itself. Anything it will
    not tell us stays None rather than being guessed.
    """
    meta = {"backend_version": None, "model_digest": None,
            "context_length": None, "quantization": quantization_hint}

    if name == "ollama":
        root = cfg["base_url"].rsplit("/v1", 1)[0]
        try:
            meta["backend_version"] = get_json(root + "/api/version").get("version")
        except Exception:
            pass
        try:
            show = get_json(root + "/api/show", {"model": cfg["model"]})
            # Ollama DOES report precision, so prefer it over the hint.
            meta["quantization"] = (show.get("details", {})
                                    .get("quantization_level") or quantization_hint)
        except Exception:
            pass
        try:
            for entry in get_json(root + "/api/tags").get("models", []):
                if entry.get("name") == cfg["model"]:
                    meta["model_digest"] = entry.get("digest")
        except Exception:
            pass
        try:
            # Only populated once the model is resident, so call after warm-up.
            for m in get_json(root + "/api/ps").get("models", []):
                if m.get("name") == cfg["model"]:
                    meta["context_length"] = m.get("context_length")
        except Exception:
            pass

    else:  # vllm
        try:
            for m in get_json(cfg["base_url"] + "/models").get("data", []):
                if m.get("id") == cfg["model"]:
                    meta["context_length"] = m.get("max_model_len")
                    meta["model_digest"] = m.get("root")   # the HF repo id
        except Exception:
            pass
        try:
            # vLLM has no version endpoint, but every response carries a
            # system_fingerprint like "vllm-0.29.0-20ef6fec". One tiny
            # non-streaming request is the cheapest way to read it.
            probe = get_json(cfg["base_url"] + "/chat/completions", {
                "model": cfg["model"],
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1, "temperature": 0,
            })
            meta["backend_version"] = probe.get("system_fingerprint")
        except Exception:
            pass

    meta["quantization"] = normalise_quant(meta["quantization"])
    return meta


def bench(name, cfg, n_trials, quantization_hint, notes, conn):
    print(f"\n{'=' * 78}")
    print(f"{name}  ({cfg['model']} @ {cfg['base_url']})")
    print("=" * 78)

    try:
        warm = client.run_once(cfg["base_url"], cfg["model"])
    except Exception as exc:
        print(f"  UNREACHABLE: {exc}")
        return None
    print(f"warm-up: {warm['input_tokens']} in / {warm['output_tokens']} out")

    meta = describe(name, cfg, quantization_hint)
    run_group = (f"{name}|{cfg['model']}|"
                 f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}")
    print(f"version={meta['backend_version']}  quant={meta['quantization']}  "
          f"ctx={meta['context_length']}")

    hw = sampler_mod.HardwareSampler(interval=0.15)
    if not hw.available:
        # Warn, do not abort: a speed-only benchmark is still valid, it just
        # stores NULL for every hardware column. But say exactly how to fix it.
        print("WARNING: LibreHardwareMonitor unreachable - hardware columns "
              "will be NULL.")
        print("  Start it elevated (it needs admin for sensor access):")
        print("    Start-Process tools/LibreHardwareMonitor/"
              "LibreHardwareMonitor.exe -Verb RunAs")
        print("  The web server starts automatically; the config is committed.")

    results = []
    print("trials: ", end="", flush=True)
    with hw:
        for i in range(n_trials):
            t0 = time.perf_counter()
            m = client.run_once(cfg["base_url"], cfg["model"])
            t1 = time.perf_counter()
            results.append(m)

            s = hw.summary(t0, t1)
            ram_gb = s.get("system_ram_used_gb_peak")

            storage.record_run(
                conn,
                run_group=run_group,
                trial_index=i,
                backend=name,
                backend_version=meta["backend_version"],
                model=cfg["model"],
                model_digest=meta["model_digest"],
                quantization=meta["quantization"],
                context_length=meta["context_length"],
                temperature=0.0,
                seed=42,
                streaming=1,
                prompt=client.PROMPT,
                response=m["response"],
                input_tokens=m["input_tokens"],
                cached_input_tokens=m["cached_input_tokens"],
                output_tokens=m["output_tokens"],
                total_tokens=m["total_tokens"],
                ttft_ms=m["ttft_ms"],
                client_total_ms=m["client_total_ms"],
                # Server-side timings do not exist over the OpenAI interface.
                load_ms=None, prefill_ms=None, decode_ms=None,
                server_total_ms=None, prefill_tokens_per_s=None,
                decode_tokens_per_s=m["decode_tok_s"],
                gpu_percent=s.get("gpu_percent_mean"),
                gpu_clock_mhz=s.get("gpu_clock_mhz_mean"),
                gpu_power_w=s.get("gpu_power_w_mean"),
                gpu_temp_c=s.get("gpu_temp_c_peak"),
                gpu_vram_used_mb=s.get("gpu_vram_used_mb_peak"),
                cpu_percent=s.get("cpu_percent_mean"),
                cpu_temp_c=s.get("cpu_temp_c_peak"),
                system_ram_used_mb=(ram_gb * 1024) if ram_gb else None,
                hardware_id=HARDWARE_ID,
                notes=notes,
            )
            print(f"{i + 1} ", end="", flush=True)
    print()

    trials_mod.summarize("time to first token",
                         [r["ttft_ms"] for r in results], "ms")
    trials_mod.summarize("decode rate",
                         [r["decode_tok_s"] for r in results], "tok/s")
    trials_mod.summarize("total wall clock",
                         [r["client_total_ms"] for r in results], "ms")
    print(f"stored {len(results)} rows in {run_group}")

    return {
        "backend": name,
        "decode": statistics.median(r["decode_tok_s"] for r in results),
        "ttft": statistics.median(r["ttft_ms"] for r in results),
        "total": statistics.median(r["client_total_ms"] for r in results),
    }


def show_history(conn):
    rows = conn.execute("""
        SELECT backend, model, quantization, COUNT(*) AS n,
               ROUND(AVG(decode_tokens_per_s), 1) AS decode,
               ROUND(AVG(ttft_ms), 1)             AS ttft,
               ROUND(AVG(gpu_percent), 0)         AS gpu,
               MIN(timestamp_utc)                 AS started
        FROM inference_runs GROUP BY run_group ORDER BY started DESC LIMIT 15
    """).fetchall()
    print(f"\n{'backend':<9}{'model':<28}{'quant':<9}{'n':>3}"
          f"{'decode':>9}{'ttft':>8}{'gpu%':>6}  started")
    print("-" * 88)
    for r in rows:
        dash = lambda v: "-" if v is None else v
        print(f"{r['backend']:<9}{r['model'][:27]:<28}{str(r['quantization']):<9}"
              f"{r['n']:>3}{dash(r['decode']):>9}{dash(r['ttft']):>8}"
              f"{dash(r['gpu']):>6}  {r['started'][:19]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="both",
                        choices=["ollama", "vllm", "both"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quantization", default="FP16",
                        help="what vLLM was started with; Ollama reports its own")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()

    conn = storage.connect()

    if args.history:
        show_history(conn)
        sys.exit(0)

    names = list(client.BACKENDS) if args.backend == "both" else [args.backend]
    summary = [bench(n, client.BACKENDS[n], args.trials,
                     args.quantization, args.notes, conn) for n in names]
    summary = [s for s in summary if s]

    if len(summary) > 1:
        print(f"\n{'=' * 78}")
        print(f"{'backend':<10}{'decode t/s':>12}{'ttft ms':>10}{'total ms':>11}")
        print("-" * 78)
        for s in summary:
            print(f"{s['backend']:<10}{s['decode']:>12.1f}"
                  f"{s['ttft']:>10.1f}{s['total']:>11.1f}")
        best = max(summary, key=lambda s: s["decode"])
        worst = min(summary, key=lambda s: s["decode"])
        print(f"\n{best['backend']} decodes "
              f"{best['decode'] / worst['decode']:.2f}x faster than "
              f"{worst['backend']}")
        print("NOTE: TTFT is not comparable across these backends - vLLM sits")
        print("behind WSL2's virtual network and pays transport the native")
        print("Windows Ollama does not. Decode rate is largely immune.")

    show_history(conn)

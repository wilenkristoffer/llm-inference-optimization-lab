# LLM Inference Optimization Lab

A hands-on lab for understanding local LLM serving: inference, model serving,
GPU memory, quantization, batching and observability — built one small file at
a time, with every claim measured rather than assumed.

This is **learning notes and experiments**, not a product. The code is
deliberately small and dependency-free so nothing important hides behind an
abstraction.

Unusually, it runs on **AMD** hardware: a Radeon RX 7800 XT (gfx1101, RDNA3)
via ROCm, rather than the NVIDIA/CUDA setups most such write-ups assume. A fair
amount of what's documented here is about what does and doesn't work on
consumer AMD under Windows and WSL2.

---

## Hardware and stack

```
CPU     AMD Ryzen 5 5600X
GPU     AMD Radeon RX 7800 XT, 16 GB, ~624 GB/s memory bandwidth
RAM     32 GB DDR4-3200
OS      Windows 11 + WSL2 (Ubuntu 24.04)
```

Serving runtimes: **Ollama 0.34** (native Windows, ROCm) and **vLLM 0.29**
(WSL2, ROCm 7.2.4). Python 3.11 on the Windows side, standard library only —
no `requests`, no `numpy`, no framework.

---

## Selected findings

All measured on the hardware above. Method and caveats are in `docs/`.

**Decode speed tracks memory bandwidth, and bigger models use the GPU better.**
Same model family, Q4_K_M, single stream:

| model | size | decode | effective bandwidth | % of 624 GB/s peak |
|---|---|---|---|---|
| qwen2.5:1.5b | 0.99 GB | 222 tok/s | 219 GB/s | 35% |
| qwen2.5:3b | 1.9 GB | 129 tok/s | 246 GB/s | 39% |
| qwen2.5:7b | 4.7 GB | 83 tok/s | 392 GB/s | 63% |
| qwen2.5:14b | 9.0 GB | 44 tok/s | 399 GB/s | 64% |

The 14b is 9.1x the weights of the 1.5b but only 5.0x slower — fixed per-token
costs amortise better over a longer forward pass.

**Ollama queues by default.** `OLLAMA_NUM_PARALLEL` defaults to 1, so
concurrent requests run one at a time: aggregate throughput stayed flat at
~197 tok/s across concurrency 1/2/4/8 while latency grew linearly. Set to 4,
batching gives 1.75x on a 1.5B and 1.90x on a 14B — sublinear, because only the
weight-streaming part of each token is shared between sequences.

**vLLM extracts more bandwidth than llama.cpp on the same card.** Identical
weights, FP16, same chat template, same OpenAI-compatible interface:

| backend | decode | effective bandwidth | % of peak |
|---|---|---|---|
| Ollama | 121 tok/s | 375 GB/s | 60% |
| vLLM | 153 tok/s | 472 GB/s | 76% |

**The KV cache is not free.** For qwen2.5:1.5b it costs ~28 KB per token, so a
32k context needs ~917 MB — about the size of the model itself. Ollama
preallocates `num_ctx x num_parallel` at load time whether you use it or not.

**GPU utilisation is unreadable through Windows' built-in counters on AMD.**
During sustained ROCm inference the WDDM engine counters report ~0% compute.
AMD's driver API (read via LibreHardwareMonitor) reports the same workload at
100%. One API returning zeros is not evidence the metric doesn't exist.

---

## Layout

```
phase1-ollama-baseline/   raw request -> streaming/TTFT -> repeated trials
                          -> SQLite storage -> benchmark harness
                          -> hardware sampler
phase2-model-serving/     API surface and chat templates; concurrency/batching
phase3-vllm/              backend-agnostic OpenAI-compatible client
docs/                     per-phase technical notes
```

Files are numbered in the order they were built. Each one exists to teach a
specific thing and is meant to be read top to bottom.

---

## Running it

**Requirements:** Python 3.11+, [Ollama](https://ollama.com) running locally,
and a model (`ollama pull qwen2.5:1.5b`). No pip installs needed.

```bash
python phase1-ollama-baseline/01_raw_request.py        # what a request really is
python phase1-ollama-baseline/05_benchmark.py --model qwen2.5:1.5b --trials 7
python phase1-ollama-baseline/05_benchmark.py --history
python phase2-model-serving/02_concurrency.py --levels 1,2,4,8
```

Results persist to `lab.db` (SQLite), which is gitignored — regenerate it by
running the benchmark.

### Hardware telemetry (optional)

The sampler reads GPU load, clock, power, temperature and VRAM from
[LibreHardwareMonitor](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor).
It is not vendored here. Download the portable build into
`tools/LibreHardwareMonitor/`, run it **as administrator**, and enable
*Options → Remote Web Server → Run* on port 8085. Without it the hardware
columns are stored as `NULL`, which is the correct answer rather than zero.

### vLLM (AMD, via WSL2)

vLLM does not run natively on Windows. `docs/` covers the WSL2 + ROCm 7.2.4 +
ROCDXG path that works on gfx1101, including the rocprofiler crash workaround
that WSL requires. The prebuilt ROCm wheel works — no source build needed.

---

## Notes on method

A few rules the experiments follow, learned the hard way:

- **Warm up, then measure.** A cold start put 84% of a request's wall time into
  model loading. The first run measures your SSD, not your GPU.
- **Report spread, not a single number.** Within-session variance is ~1-2%;
  between sessions it is ~4%. Comparisons that matter are run interleaved in
  one session.
- **Pin `temperature=0` and a seed**, or the same prompt returns different
  token counts and nothing is comparable.
- **Store `NULL` for what you can't measure.** A zero in `gpu_percent` is a lie
  that averages into a chart later.
- **Check numbers against physical limits.** Decode at 222 tok/s on a 1 GB
  model is above what DDR4 could deliver, which proved GPU execution before any
  tool confirmed it.

The docs also record what turned out to be **wrong** — an inflated prefill
figure caused by counting cached tokens, a "quantization is degrading your
model" claim that a controlled test disproved, and several over-optimistic
scaling predictions. Those are kept deliberately; they are the useful part.

---

## Status

Phases 1 and 2 are complete and documented. Phase 3 (vLLM) is in progress —
the backend comparison runs, the concurrency comparison does not yet.
Quantization, GPU environments, Docker, optimization and a final benchmark are
planned but not built.

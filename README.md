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
RAM     32 GB DDR4-3600 (XMP; earlier phases assumed 3200)
OS      Windows 11 + WSL2 (Ubuntu 24.04)
```

Serving runtimes: **Ollama 0.34**, later 0.40 (native Windows, ROCm) and **vLLM 0.29**
(WSL2, ROCm 7.2.4). Python 3.11 on the Windows side, standard library only —
no `requests`, no `numpy`, no framework.

---

## Findings

The full findings, phase by phase, with a checklist for apps that run on
Ollama, are in **[docs/summary.md](docs/summary.md)**. The headlines, all
measured on the hardware above:

- **The model generation matters more than anything else the lab measured.**
  On 60 code-scored mail-triage agent episodes, a current 4B (qwen3.5:4b,
  82%) beats the previous generation's 14B (qwen2.5:14b, 25%) by 57 points.
  No serving setting came close to that.
- **A one-hour LoRA fine-tune took that 4B to 100%**, matching the 27B
  model it learned from (98%), with no loss on the other suites. The best
  training data turned out to be scripted perfect runs, not the 27B's.
- **Decode speed is memory bandwidth.** Fit the model fully on the GPU: a
  model that spills over PCIe on Windows runs ~10x slower per agent turn.
- **Ollama's defaults hide traps.** It queues concurrent requests by
  default, forces some architectures to one slot whatever the setting, and
  truncates an overflowing prompt to half the window without an error.
- **vLLM batches far better on this AMD card** (2,052 tok/s at 32 requests
  on a 1.5B) once its GEMMs are tuned, and it serves a LoRA adapter
  unmerged, but takes WSL2 and more setup.

---

## Layout

```
phase1-ollama-baseline/   raw request -> streaming/TTFT -> repeated trials
                          -> SQLite storage -> benchmark harness
                          -> hardware sampler
phase2-model-serving/     API surface and chat templates; concurrency/batching
phase3-vllm/              OpenAI-compatible client -> matched backend benchmark
                          -> concurrency sweep -> vLLM /metrics diagnostics
phase4-observability/     read-only JSON API over lab.db, backend process
                          control, and a React + TypeScript UI
phase5-inference-fundamentals/
                          tokenization -> prefill scaling -> context truncation
                          -> prefix cache -> inter-token latency
phase6-quantization/      VRAM/speed ladder -> fidelity vs full precision
                          -> checkable questions -> Q2_K collapse investigation
                          -> locally built ladders
phase7-gpu-environments/  stack inventory -> measured hardware ceilings (HIP)
                          -> CPU vs GPU -> layer offload and a model that
                          does not fit -> ROCm vs Vulkan
phase8-docker/            Ollama and vLLM in containers; Compose
phase9-optimization/      speculative decoding -> context and KV cache
                          -> prefix caching -> tuned GEMMs -> concurrency
phase10-final-benchmark/  code-checked suites and mailbox agent episodes
                          -> cost per episode -> Pareto choice
phase11-lora-finetuning/  training data -> baseline -> LoRA in WSL2
                          -> merge into Ollama's GGUF -> unmerged on
                          llama-server and vLLM
docs/                     per-phase technical notes, the project summary,
                          and the WSL2 + ROCm setup that gets vLLM running
                          on consumer AMD
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

### Evaluating a new model

Phase 10's main result is that the model generation matters more than any
serving setting. So the lab ends in a tool that scores a new Ollama tag on the
same suites and the same footing, and places it on the leaderboard:

```bash
python phase10-final-benchmark/evaluate.py gemma4:e4b --quick   # 10 items per suite, ~2 min
python phase10-final-benchmark/evaluate.py qwen3.6:8b --pull    # full run, ~1 h on a 4-8B model
python phase10-final-benchmark/evaluate.py --report             # leaderboard only
```

The full run has five steps, all handled by the tool:
- **detect:** reads `ollama show` to find whether the model takes tools, whether think is a bool or only takes levels, whether it bundles a draft model, and its Modelfile penalties. The result goes to `candidates.json`, so nothing needs typing in.
- **preflight:** one tool call and a think check.
- **suites:** all five, A-E.
- **repeat:** suite D runs a second time.
- **cost:** 20 episodes one at a time, then 8 at `NUM_PARALLEL` 4.

The report includes the leaderboard, a McNemar test against the leader, the Pareto front and a chart. Every step resumes after an interruption.

Each model also gets a list of warnings. These cover the traps Phase 10 found by hand:
- a tool call Ollama's parser left as text;
- think that cannot be turned off;
- a bundled draft model;
- Ollama forcing one slot;
- spilling into system RAM;
- answers that change under batching or between identical runs.

`--quick` rows are kept out of the leaderboard.

### The observability UI

```bash
python phase4-observability/api.py              # 127.0.0.1:8010
cd phase4-observability/ui && npm install && npm run dev   # localhost:5173
```

Dashboard, a filterable/sortable runs table with drill-down, comparison charts,
and start/stop control for both backends with readiness polling — so swapping
Ollama for vLLM is one click instead of three terminals.

Open the UI on `localhost:5173`, not `127.0.0.1:5173`: Vite binds IPv6-only,
which is the exact mirror of Ollama binding IPv4-only. Both facts cost time to
discover and are documented in `docs/`.

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

**Complete.** All eleven phases are done and documented in `docs/`:

| phase | topic | notes |
|---|---|---|
| 1 | Ollama baseline: streaming, TTFT, repeated trials, SQLite, hardware sampler | [phase1.md](docs/phase1.md) |
| 2 | model serving: API surface, chat templates, concurrency and batching | [phase2.md](docs/phase2.md) |
| 3 | Ollama vs vLLM, matched; the batch-size cliff down to the kernel | [phase3.md](docs/phase3.md), [setup](docs/phase3-vllm-setup.md) |
| 4 | observability: JSON API and React UI over `lab.db` | [phase4.md](docs/phase4.md) |
| 5 | inference fundamentals: tokenization, prefill, truncation, prefix cache, latency | [phase5.md](docs/phase5.md) |
| 6 | quantization: GGUF ladders, token-level fidelity, the Q2_K collapse | [phase6.md](docs/phase6.md) |
| 7 | GPU environments: ceilings, CPU vs GPU, offload, ROCm vs Vulkan | [phase7.md](docs/phase7.md) |
| 8 | Docker and Compose: vLLM in a container within 2% of native | [phase8.md](docs/phase8.md) |
| 9 | optimization: speculative decoding, KV cache, prefix caching, tuned GEMMs | [phase9.md](docs/phase9.md) |
| 10 | final benchmark: picking the agent's model from five vendors | [phase10.md](docs/phase10.md) |
| 11 | LoRA fine-tuning of the picked model, and serving the adapter | [phase11.md](docs/phase11.md) |

What it all adds up to, and a checklist for applying it, is in
[docs/summary.md](docs/summary.md). Known gaps are listed at the end of each
phase document; the largest is that kernel profiling is unavailable here,
because `rocprof` had to be disabled to get vLLM running under WSL. Disk
used by the lab's models, and when each can go, is in
`docs/cleanup-plan.md`.

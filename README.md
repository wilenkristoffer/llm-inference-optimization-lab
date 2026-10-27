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

## Selected findings

All measured on the hardware above. Method and caveats are in `docs/`.

**The model generation matters more than anything else the lab measured.**
On 60 multi-turn mail-triage episodes scored by code (the agent must handle
every message and then stop), one at a time on Ollama:

| model | episodes fully right | s/episode | VRAM | correct/hour |
|---|---|---|---|---|
| bonsai-2:27b (ternary, via evaluate.py) | 98% | 19.3 | 9.1 GB | 184 |
| gemma4:e4b (via evaluate.py) | 81% | 3.5 | 5.4 GB | 839 |
| qwen3.5:4b (BF16) | 82% | 12.5 | 9.6 GB | 236 |
| nemotron-3.5-bolt:e4b | 72% | 4.9 | 9.1 GB | 528 |
| gpt-oss:20b, qwen3:14b, gemma4:12b | 59-62% | 7.1-11.9 | 8.6-12.4 GB | 181-301 |
| qwen2.5:14b | 25% | 11.0 | 10.0 GB | 79 |
| qwen2.5 1.5B-7B | 0-18% | | | |

A current 4B beats the previous generation's 14B by 57 points. qwen3.5 vs
nemotron is not significant (p = 0.18, paired); both beat the 60% group. And
running episodes in parallel gained nothing on Ollama for the good models:
qwen3.5 is silently forced to one slot, nemotron's bundled draft model blocks
batching, and gemma4's answers break under batched decode. The first two
rows came from `evaluate.py` on the same day, and they show the point of
the tool. bonsai-2:27b is a 27B in ternary weights that fits on the GPU,
and it fails 2 episodes in 120. gemma4:e4b had been on disk all along and
beats its own NVIDIA tune (nemotron). See `docs/phase10.md`.

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

**vLLM extracts more bandwidth than llama.cpp on the same card** - thanks to
one kernel. Identical weights, FP16, same chat template, same
OpenAI-compatible interface:

| backend | decode | effective bandwidth | % of peak |
|---|---|---|---|
| Ollama | 121 tok/s | 375 GB/s | 60% |
| vLLM | 153 tok/s | 472 GB/s | 76% |

**vLLM's batching falls off a cliff at 6 sequences - then keeps scaling.**
Aggregate throughput, 300-token requests, a CUDA graph for every level:

| concurrency | vLLM aggregate | vLLM per request |
|---|---|---|
| 1 | 151 tok/s | 152 tok/s |
| 4 | 556 | 145 |
| 5 | **695** | 144 |
| 6 | 438 | **74** |
| 8 | 570 | 72 |
| 16 | 1,091 | 69 |
| 32 | **2,052** | 66 |

vLLM's ROCm path uses a hand-written "skinny" matrix-multiply kernel only when
a decode step has 1-5 tokens, and the generic one from 6. Per-sequence speed
halves at the switch and barely moves after it. Turning the kernel off
(`VLLM_ROCM_USE_SKINNY_GEMM=0`) removes the cliff and drops single-stream vLLM
to 76 tok/s - so its lead over Ollama is that kernel. The first sweep stopped at
batch 8, just past the cliff, and read it as a hard ceiling at 4; the
correction is kept in `docs/phase3.md`. Ollama (measured to 8) overtakes vLLM
at concurrency 8, 648 vs 570 tok/s.

**CUDA graph padding can change the kernel.** vLLM pads a batch up to the next
captured graph size, and its default sizes go 4, 8. A batch of 5 is padded to
8, lands on the generic kernel, and runs at 78 tok/s per request instead of
144. Capturing 5 explicitly fixes it.

**A model that almost fits runs far slower than one that fits.** Decode time
is the sum of each device's share, so splitting layers between GPU and CPU is
harmonic: with 5 of the 14B's 49 layers on the CPU, decode halves (52.8 ->
26.5 tok/s), and a two-parameter model predicts every split within 9%. The
32B (19.9 GB) is worse than split. On Windows the ROCm runtime reports ~2 GB
more free VRAM than exists (the desktop holds it), so Ollama's default
placement oversubscribes and WDDM silently pages over PCIe. Forced fully onto
the GPU it reports "100% GPU" and 99% load while decoding at 4 tok/s. Per agent
turn it is ~10x slower than the 14B.

**The vendor library is worth 2-4x at prefill, not at decode.** Same Ollama,
same files, same card: ROCm prefills 2.3-3.7x faster than Vulkan (tuned GEMM
libraries), but decode is mixed, and Vulkan wins by 8% on the 1.5B while using
300-800 MB less VRAM. Measured ceilings: VRAM read 602 GB/s (97% of the
datasheet), PCIe 28 GB/s, FP16 matmul 60 TFLOPS (80%).

**Quantization damage shows up long before answers change.** Comparing each
level's next-token predictions with full precision, token by token, over
~1,500 positions (top-1 agreement / KL divergence):

| level | Qwen2.5-3B | Qwen3.5-4B |
|---|---|---|
| Q8_0 | 98.7% / 0.002 | 99.3% / 0.003 |
| Q4_K_M (Ollama default) | 93.3% / 0.058 | 94.7% / 0.030 |
| Q3_K_M | 75.0% / 0.677 | 89.4% / 0.108 |
| Q2_K | 0.7% / 4.83 - broken | 66.0% / 0.885 |

A 60-question checkable test could not tell FP16 from Q4_K_M on either model.
Qwen2.5-3B falls off a cliff below Q4; Qwen3.5-4B is still sound at Q3_K_M.
Qwen2.5-3B's Q2_K collapse was traced to one tensor type: `ffn_gate` at 2
bits. Raising it alone to 3 bits restores partly coherent output.

**Ollama truncates silently, oldest-first, and an overflow costs half the
window.** Past `num_ctx` it keeps roughly `num_ctx / 2` - the system prompt goes
first - with no error. Asked about a planted code it no longer had, the model
confidently answered with a different one.

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
docs/                     per-phase technical notes, including the WSL2 + ROCm
                          setup that gets vLLM running on consumer AMD
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

Phases 1-7 are complete and documented: Ollama baseline and instrumentation,
model serving and batching, a matched Ollama-vs-vLLM comparison including the
concurrency sweep and a diagnosis of vLLM's batch-size cliff down to the kernel, an
observability API and UI over the results, inference fundamentals -
tokenization, prefill, truncation, prefix caching and inter-token latency -
and quantization: full GGUF ladders on Qwen2.5-3B/7B and Qwen3.5-4B, measured
with a token-level fidelity method, plus a root cause for Qwen2.5-3B's Q2_K
collapse - and GPU environments: the stack layer by layer on this machine,
measured hardware ceilings, CPU vs GPU, layer offload, a model that does not
fit, and ROCm vs Vulkan. Phases 8-10 are complete too: Docker (vLLM in a
container within 2% of native), optimization (speculative decoding, KV
cache, prefix caching, tuned GEMMs, concurrency) and a final benchmark that
picks the model for the user's agent from five vendors' current releases.

### Final summary

What the lab found, for running an agent on a 16 GB consumer AMD card:

- **Pick the model first, and re-pick it with `evaluate.py` when new ones
  ship.** bonsai-2:27b (98% of episodes) or gemma4:e4b (4.6x the correct
  episodes per hour) on Ollama with
  `think: false`, temperature 0, `num_ctx` 8192. Phase 10.
- **Fit it fully on the GPU.** A model that spills pages over PCIe on
  Windows: the 32B is ~10x slower per turn than the 14B, and no layer split
  helps. Phase 7.
- **Keep the prompt prefix stable.** A timestamp at the top of the system
  prompt costs 23x the TTFT on every turn. Phase 9.
- **Check the runner, not the setting.** Ollama silently ignores
  `NUM_PARALLEL` for some architectures, ships draft models in some
  Modelfiles, and truncates overflowing prompts to half the window with no
  error. Phases 5, 9, 10.
- **vLLM is faster here once its GEMMs are tuned**, but only the models it
  was run with (Qwen2.5) could not do the task. Phases 3, 9.

Phase 11 adds LoRA / QLoRA fine-tuning of
a small model and measures what serving the adapter costs - see
`docs/phase11-plan.md`. Known gaps are listed at the end of each phase
document - the largest is that kernel profiling is unavailable in this
environment because `rocprof` had to be disabled to get vLLM running under WSL.
Disk used by the lab's models, and when each can go, is in
`docs/cleanup-plan.md`.

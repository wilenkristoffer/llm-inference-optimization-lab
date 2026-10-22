# Phase 7 plan - GPU environments

Status: **complete 2026-10-22.** Results in `phase7.md`. This is a plan, not results. Predictions
below are written down so the measurements can prove them wrong.

The brief asks for this phase to teach how GPU inference works:

```
Application          the lab's scripts, the user's agent
    |
Inference runtime    Ollama -> llama-server (llama.cpp) / vLLM
    |
GPU libraries        ROCm (HIP, rocBLAS, hipBLASLt) / Vulkan / CUDA
    |
GPU                  RX 7800 XT, gfx1101, 16 GB, ~624 GB/s
```

It should cover VRAM, compute, memory bandwidth, CPU vs GPU, utilisation,
loading into VRAM, what happens when a model does not fit, and offloading. It
should then explain NVIDIA -> CUDA, AMD -> ROCm -> HIP, and where ZLUDA fits.

Phases 1-6 already measured much of the first list. This phase fills the gaps
with experiments on this card, and writes the explanation around those numbers
rather than around diagrams.

---

## What is already known (no new runs needed)

| topic | finding | from |
|---|---|---|
| bandwidth | decode tracks weight size; 35-64% of 624 GB/s, bigger models use more of it | Phase 1 |
| CPU vs GPU, by arithmetic | 222 tok/s on a 1 GB model is above what DDR4-3200 could feed, which proves the GPU was used | Phase 1 |
| loading | cold start was 84% of a request; load 1.68 s vs 0.004 s warm | Phase 1 |
| VRAM | footprint = weights + KV cache (28 KB/token on 1.5B) + buffers; preallocated per slot | Phases 1, 6 |
| utilisation | Windows counters read ~0%; AMD's driver API reads 100% for the same work | Phase 1 |
| clocks | 113 MHz idle to 2592 MHz under load; short runs never reach boost | Phase 1 |
| kernels | vLLM's lead over llama.cpp is one ROCm kernel (skinny GEMM) | Phase 3 |
| compute vs bandwidth | prefill is slower on quantized weights (dequant cost), decode faster | Phase 6 |

Gaps: what the card can actually deliver (624 GB/s is a datasheet number), CPU
inference measured instead of inferred, partial offload, a model that doesn't
fit, and one runtime on two different GPU libraries.

---

## What this machine already shows about the stack

Ollama's install directory ships every backend side by side. The runtime picks
one at startup:

```
lib/ollama/
  cuda_v12/      1.1 GB   NVIDIA, CUDA 12   - unused here
  cuda_v13/      898 MB   NVIDIA, CUDA 13   - unused here
  rocm_v7_1/     980 MB   AMD, ROCm 7.1     - what runs
  vulkan/         40 MB   any vendor, Vulkan compute
  ggml-cpu-*.dll  ~2 MB each, one per CPU generation
```

From the server log:

- `OLLAMA_VULKAN:true` is set, yet `library=ROCm compute=gfx1101` is chosen.
  Both backends see the same card, and Ollama prefers the vendor library.
- The CPU backend loaded is `ggml-cpu-haswell.dll`, not `zen4`. The 5600X is
  Zen 3: it has AVX2 but no AVX-512, and the zen4 build assumes AVX-512.
- The ROCm folder is ~25x the size of the Vulkan one. Most of that is rocBLAS
  and hipBLASLt kernels, precompiled per GPU architecture.

The size gap and the per-CPU builds make the main point of the phase before any
benchmark: "GPU support" means shipping the right compiled code for the exact
chip, and a runtime is a dispatcher over those builds.

WSL has the other half: `hipcc`, `rocminfo`, `rocm-smi` and a ROCm PyTorch
venv (from Phase 3), so a small HIP kernel can be compiled and run.

---

## Experiments

### 1. Stack inventory - `01_stack_inventory.py`

No benchmark. It prints what each layer is on this machine: the Ollama backend
folders and sizes, the backend and CPU variant from the latest server log, the
loaded runner's command line (already parsed in Phase 6), and in WSL,
`rocminfo`'s agent list (gfx1101, compute units, wavefront size 32). A short,
reproducible "what is installed and what was picked".

### 2. What the card can actually do - `02_ceilings`

Every "% of peak" figure so far uses the datasheet's 624 GB/s. That number is
an upper bound, not something a kernel achieves.

- **VRAM bandwidth**: a minimal HIP copy kernel, compiled with `hipcc` in WSL.
  It is about 30 lines and reads like CUDA, which is the point of HIP. It is
  timed across buffer sizes, since small buffers fit in the 64 MB Infinity Cache
  and will report figures above VRAM speed.
- **Compute**: FP16 matrix multiply throughput through PyTorch (rocBLAS) in the
  ROCm venv, against the card's datasheet peak.
- **System RAM bandwidth**: a large-array copy on the CPU, for the CPU-inference
  ceiling (DDR4-3200 dual channel, 51 GB/s on paper).
  *Correction found while running this: the RAM is DDR4-3600 (XMP), 57.6 GB/s
  on paper. The predictions below keep the original figure, as written.*

Prediction: VRAM copy reaches 80-90% of 624 GB/s. If so, Phase 1's 14B at
399 GB/s is nearer 75% of what is achievable than the 64% reported.

Caveat: this runs under WSL's GPU path, not native Windows where Ollama runs.
The hardware is the same but the driver stack is not, and the write-up will say
so.

### 3. CPU vs GPU - `03_cpu_vs_gpu.py`

Same model, same prompt, `num_gpu 0` vs fully on GPU. Interleaved rounds, as in
Phase 6. Models: qwen2.5 1.5b, 7b, 14b (all installed).

Measured per run: prefill tok/s, decode tok/s, and CPU/GPU load from the
sampler.

Predictions:
- Decode on the CPU tracks RAM bandwidth: ~51 GB/s / 4.7 GB = ~11 tok/s ceiling
  for the 7B, so a GPU advantage of roughly 8-10x.
- Prefill is compute-bound, so the gap is much larger, on the order of 50-100x.
  Six Zen 3 cores against a GPU's matrix units.
- The CPU thread count matters for prefill and barely matters for decode once
  memory is saturated. One small `num_thread` sweep on the 1.5B tests this.

### 4. Partial offload - `04_offload_sweep.py`

qwen2.5:14b (49 layers including output, 9 GB) fits entirely, so it can be
swept: `num_gpu` = 0, 8, 16, ..., 49 layers on GPU.

Prediction: time per token = (GPU bytes / GPU bandwidth) + (CPU bytes / CPU
bandwidth). The slow part dominates, so the curve is not linear: moving just
10% of layers off the GPU should cost far more than 10% of speed. The model
gives a predicted curve from experiment 2's ceilings, and the sweep is checked
against it.

Also recorded: VRAM and system RAM footprint per point, and load time.

### 5. The model that does not fit - qwen2.5:32b

19.9 GB on a 16 GB card. Loaded with Ollama's defaults to see how many layers it
places on the GPU, then decode and prefill measured, and compared with the
prediction from experiment 4's formula.

This is the model from the user's agent log (Phase 1's agent analysis), which
noted that offload latency there was never examined. The result is a concrete
number for the agent: what the 32B cost per turn compared with a 14B that fits.

### 6. Same runtime, different GPU library - `05_backend_swap.py`

Ollama on ROCm vs Ollama on Vulkan, same card, same model files. This makes the
"GPU libraries" layer something you can swap and measure.

Feasibility check first: forcing Vulkan needs ROCm hidden from Ollama
(`HIP_VISIBLE_DEVICES=-1` or `OLLAMA_LLM_LIBRARY`), with Ollama restarted
through the lab's server control. If neither works on 0.40.7, this experiment
is dropped and the write-up says why.

Models: 1.5B and 7B Q4_K_M. Prefill and decode, interleaved. No prediction on
the winner. Vulkan llama.cpp has closed much of the gap on AMD in public
reports, and this card is a good place to check.

---

## Explanation sections in `docs/phase7.md`

Written around the numbers above, not as a generic primer:

- **The stack**: what each layer is on this machine, using experiment 1.
- **VRAM, bandwidth, compute**: roofline in plain terms. Decode streams the
  weights once per token (bandwidth); prefill reuses them across the prompt
  (compute). Phases 1, 5 and 6 plus experiment 2.
- **Utilisation**: why "100% GPU" in `ollama ps` (placement) and 100% load
  (busy) are both true while bandwidth sits at 60%. Busy is not efficient.
- **Loading and not fitting**: experiments 4 and 5.
- **NVIDIA -> CUDA -> ecosystem**: the driver, the CUDA runtime, and the
  libraries (cuBLAS, cuDNN, NCCL, TensorRT). Why so much software assumes it:
  15+ years of compiled kernels and tooling. Explained, not measured; there is
  no NVIDIA card here.
- **AMD -> ROCm -> HIP**: HIP as CUDA-shaped source that compiles for AMD or
  NVIDIA, `hipify`, rocBLAS/hipBLASLt as the cuBLAS equivalents, and
  per-architecture builds. That is why gfx1101 support matters and why
  consumer cards lag behind. This lab's own history serves as evidence: the
  WSL setup, the rocprofiler stub, the skinny-GEMM kernel.
- **Vulkan**: the vendor-neutral route, and how it compares (experiment 6).
- **ZLUDA**: a translation layer that runs unmodified CUDA *binaries* on
  non-NVIDIA GPUs. It is not HIP and not AMD's native path. It exists for
  software that ships only CUDA builds. Conceptual only (see below).

---

## Planned files

```
phase7-gpu-environments/
  01_stack_inventory.py   what is installed, what got picked
  02_bandwidth.hip        HIP copy kernel (built and run in WSL)
  02_ceilings.py          runs the HIP probe, torch matmul, CPU copy; stores results
  03_cpu_vs_gpu.py        num_gpu 0 vs all, interleaved
  04_offload_sweep.py     14b layer sweep + 32b default split
  05_backend_swap.py      ROCm vs Vulkan (if feasible)
docs/phase7.md            explanation + results
```

Results go to `lab.db` through the existing harness. The `num_gpu` setting and
backend library become columns on `inference_runs`, so the UI can filter on
them.

No downloads: every model used is already installed.

---

## Out of scope, deliberately

- **Running ZLUDA.** llama.cpp and vLLM have native ROCm builds, so there is
  nothing here that needs CUDA translation, and a result would measure ZLUDA,
  not the card. Explained only.
- **vLLM CPU offload** (`--cpu-offload-gb`). Possible, but the llama.cpp sweep
  already answers the question, and vLLM's WSL setup is fragile. Possible in
  Phase 9 if wanted.
- **Kernel profiling.** `rocprof` is stubbed out in WSL (see the ROCm setup
  notes); still a known gap.
- **Multi-GPU.** One card.

---

## Decisions (2026-10-22)

1. Vulkan comparison: **included**, subject to the feasibility check.
2. ZLUDA: **explained only**, not installed.

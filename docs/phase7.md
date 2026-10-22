# Phase 7 - GPU environments

Technical notes. How GPU inference actually works, layer by layer, on this
machine: what each layer is, what the hardware can really deliver, what
happens on the CPU, between CPU and GPU, and when a model does not fit, and
the same runtime on two different GPU libraries. Then NVIDIA's stack, AMD's
stack, and where Vulkan and ZLUDA fit.

Date: 2026-10-22. Ollama 0.40.7 (llama.cpp runner), ROCm 7.2.4 + PyTorch 2.14
in WSL2. Plan: `phase7-plan.md`. Code: `phase7-gpu-environments/`.

---

## Goal

Phases 1-6 measured much of what this phase is about without naming it:
bandwidth-bound decode, model loading, VRAM footprints, unreadable GPU
counters, one ROCm kernel deciding vLLM's speed. This phase fills the gaps:

1. **The stack** - what "application -> runtime -> GPU library -> GPU"
   actually consists of here, from files and log lines.
2. **The ceilings** - what the card and the CPU can really deliver, measured,
   instead of datasheet figures.
3. **CPU vs GPU** - the same weights on each.
4. **Not fitting** - partial offload, and a model bigger than the card.
5. **The library layer** - Ollama on ROCm vs Ollama on Vulkan.

---

## 01 - The stack on this machine

```
Application          the lab's scripts, the user's agent
    |                HTTP, JSON
Inference runtime    ollama.exe (API, scheduler, model store)
    |                spawns one child process per loaded model
                     llama-server.exe (llama.cpp: the model's maths)
    |                loads one ggml backend DLL
GPU library          rocm_v7_1\ggml-hip.dll -> HIP runtime, rocBLAS, hipBLASLt
    |                kernel launches, memory copies
Driver               AMD Adrenalin (WDDM on Windows)
    |
GPU                  RX 7800 XT, gfx1101, 60 CUs, 16 GB GDDR6
```

`01_stack_inventory.py` reads each layer from the machine. What it showed:

**One runtime, many backends.** Ollama's install ships every backend side by
side and chooses at startup:

| folder | size | what |
|---|---|---|
| `cuda_v12` | 1,113 MB | NVIDIA, CUDA 12 |
| `cuda_v13` | 898 MB | NVIDIA, CUDA 13 |
| `rocm_v7_1` | 977 MB | AMD, ROCm 7.1 - **what runs here** |
| `vulkan` | 40 MB | any vendor |
| `ggml-cpu-*.dll` | ~2 MB each | 14 builds, one per CPU instruction-set generation |

The ROCm and CUDA folders are ~25x the Vulkan one because they carry
precompiled math libraries - rocBLAS, hipBLASLt, cuBLAS - with kernels built
and tuned separately for each GPU architecture. "Supports this GPU" means
"ships compiled kernels for this exact chip". The runtime is a dispatcher over
those builds.

**The ROCm backend is the CUDA backend.** llama.cpp logs
`ggml_cuda_init: found 1 ROCm devices`. Its ROCm support is its CUDA source
compiled through HIP, not a separate implementation. PyTorch does the same:
on a ROCm build, `torch.cuda` *is* the HIP runtime.

**The CPU build is chosen by instruction set.** The Ryzen 5600X loads
`ggml-cpu-haswell` (AVX2), not `zen4`: Zen 3 has no AVX-512, and the zen4 build
assumes it. Vendor is irrelevant; capabilities decide.

**Vulkan is enabled but not used.** `OLLAMA_VULKAN=true` is set, both
libraries see the card, and Ollama prefers the vendor one.

**Placement is a runner flag.** A request's `num_gpu` becomes llama-server's
`-ngl` (layers on the GPU). With no `num_gpu`, Ollama 0.40 passes no `-ngl`
and llama.cpp decides. `/api/ps` reports the result as `size` vs `size_vram`,
and the runner's log says `offloaded N/M layers to GPU`.

**WSL sees the card differently.** PyTorch in WSL reports **30** compute
units for a card sold as having 60. RDNA3 pairs CUs into work-group
processors (WGPs) and the HIP runtime counts WGPs. `rocminfo` fails outright
under WSL. It enumerates GPUs through the Linux `kfd` driver, and WSL reaches
the GPU through `/dev/dxg` (the same reason `rocprof` is stubbed out, see
`phase3-vllm-setup.md`).

---

## 02 - What the hardware can actually do

Every "% of peak" in the lab so far divided by a datasheet number. A datasheet
number is clock x bus width, and no real kernel reaches it.
`02_bandwidth.hip` (a ~30-line HIP program, built with `hipcc` in WSL) and
`02_ceilings.py` (PyTorch probes) measure the ceilings inference actually
runs into.

| ceiling | measured | datasheet | share | what uses it |
|---|---|---|---|---|
| VRAM read, >= 128 MB | **602 GB/s** | 624 | 97% | decode: every weight read once per token |
| VRAM copy (read+write) | 470-490 GB/s | 624 | 76% | - |
| PCIe 4.0 x16, pinned | **28 GB/s** | 31.5 | 89% | loading, offload, paging |
| GPU FP16 matmul | **59.6 TFLOPS** | 74.65 | 80% | prefill |
| GPU FP32 matmul | 10.0 TFLOPS | 37.3 | 27% | - |
| RAM read, 6 threads | **46.6 GB/s** | 57.6 | 81% | CPU decode |
| CPU FP32 matmul | 0.45 TFLOPS | - | - | CPU prefill (roughly) |

Things worth taking from it:

- **The VRAM figure is real.** A plain read kernel gets 97% of the datasheet,
  more than the 80-90% predicted. So Phase 1's 14B at 399 GB/s effective uses
  66% of what the card can do. The remaining third is lost in the runtime, not
  in an inflated spec.
- **Caches look like impossible bandwidth.** Buffers up to 64 MB read at
  ~2,000 GB/s: they live in the 64 MB Infinity Cache. The copy probe drops off
  one step earlier because it touches two buffers. A model's weights are
  gigabytes, so decode never benefits. Only the plateau matters.
- **AMD's FP32 figure assumes dual issue.** RDNA3 can issue two FP32 operations
  per clock in specific instruction pairings. Compilers rarely produce them,
  and a real GEMM got 27%. FP16 goes through the matrix (WMMA) path and gets
  80%.
- **The GPU's lead is far larger in compute than in bandwidth.** 13x on memory
  (602 vs 46.6 GB/s) and roughly 130x on matmul (59.6 FP16 vs 0.45 FP32). That
  predicts the next section: decode (bandwidth) should gain much less from the
  GPU than prefill (compute).

**A correction found here:** the RAM is **DDR4-3600** (an XMP kit,
`ConfiguredClockSpeed 3600`), not the DDR4-3200 assumed since Phase 1. The paper
ceiling is 57.6 GB/s, not 51.2. Phase 1's "GPU proven by arithmetic" argument
still holds (~3.8x margin instead of 4x). The README and `phase1.md` are
corrected.

**Caveat:** these probes run in WSL, Ollama runs natively. The silicon is the
same; the driver path is not. Memory bandwidth is a property of the hardware
and should carry over. The CPU figure comes from inside a VM.

---

## 03 - CPU vs GPU

`03_cpu_vs_gpu.py`. Same weights (Q4_K_M), three placements, each on a side
Ollama (`side_server.py`) so all run under identical settings:

- **gpu** - ROCm build, every layer on the GPU
- **ngl0** - ROCm build, zero layers on the GPU (`num_gpu 0`)
- **cpu** - CPU-only build, no GPU library loaded at all

2 rounds x 2 trials, interleaved and shuffled. Decode is 128 tokens of a
long-form answer. Prefill is ~1,240 tokens, Phase 6's prompt length.

| model | GPU decode | CPU decode | ratio | GPU prefill | CPU prefill | ratio |
|---|---|---|---|---|---|---|
| 1.5B | 236 tok/s | 40.0 | **5.9x** | 8,132 | 270 | **30x** |
| 7B | 96.8 | 9.7 | **10.0x** | 2,324 | 58 | **40x** |
| 14B | 53.1 | 4.9 | **10.9x** | 1,109 | 28 | **39x** |

`ngl0` and the CPU build agree within noise everywhere (14B: 4.8 vs 4.9 decode,
27 vs 28 prefill), with no GPU activity during `ngl0` runs. So zero layers
on the GPU means CPU-only, even with the GPU library loaded. See 04 for why
that is not true at one layer.

**Decode follows the bandwidth ratio, once the GPU is used well.** Effective
bandwidth (weights x tok/s) against the *measured* ceilings:

| model | GPU | of 602 GB/s | CPU | of 46.6 GB/s |
|---|---|---|---|---|
| 1.5B | 217 GB/s | 36% | 37 GB/s | 79% |
| 7B | 422 GB/s | 70% | 42 GB/s | 90% |
| 14B | 444 GB/s | 74% | 41 GB/s | 87% |

The CPU runs close to its memory limit at every size. The GPU does not reach
its limit on small models (Phase 1's finding: fixed per-token costs dominate a
short forward pass). So the GPU's decode lead grows with model size, from 6x
towards the 13x bandwidth ratio. The plan predicted 8-10x. Right for large
models, wrong for small ones, and the reason is GPU efficiency, not CPU.

**Prefill gains more, but less than predicted.** The plan said 50-100x from
the compute ratio; measured 30-40x. The GPU reaches 35 TFLOPS on the 7B
(2,324 tok/s x ~15 GFLOP/token), 59% of its measured FP16 ceiling. The CPU
reaches ~0.9 TFLOPS, *double* PyTorch's FP32 matmul figure. llama.cpp's CPU
path multiplies quantized weights with 8-bit integer dot products, which AVX2
does faster than FP32. Comparing the GPU's FP16 to the CPU's FP32 overstated
the gap.

**Decode on the 7B/14B at Ollama 0.40 is ~15-20% faster than Phase 1's table**
(7B 85.5 -> 96.8, 14B 44.3 -> 53.1). Phase 6's runs on 0.40.7 show the same
rise, so it is the runtime upgrade (0.34 -> 0.40), not anything here. The
README's bandwidth table is from 0.34.

### CPU threads: decode saturates, prefill scales

Same model (1.5B), CPU build, `num_thread` swept:

| threads | decode | prefill |
|---|---|---|
| 1 | 17.6 | 58 |
| 2 | 33.4 | 116 |
| 3 | 39.8 | 166 |
| 4 | **40.1** | 189 |
| 6 | 38.4 | 219 |
| 12 | 29.9 | **262** |

Decode stops improving at 3-4 threads: that is where RAM is saturated, and
more cores just wait on memory. At 12 threads (SMT, two per core) it gets
*worse*, as threads contend for the same cores and caches. Prefill keeps
scaling to 12 because it is compute. llama.cpp's default (6, one per physical
core) is a reasonable compromise.

### A measurement trap found on the way

A first CPU check with "Count to five." reported **82.6 tok/s** - which on
a ~1 GB model would need 81 GB/s from RAM that physically cannot exceed 57.6.
The model had answered "Five." - **3 tokens**. Ollama's rate divides those by
a ~50 ms window and overstated the client-clock rate (36 tok/s) by about 2x.
Over 256 tokens, server and client clocks agree to 0.1 tok/s. Decode rates are
only measured with a long-form prompt and a pinned token count, and short
outputs are flagged.

### GPU load needs a long enough window

LibreHardwareMonitor refreshes once a second (measured). A 0.5 s GPU decode
showed "3% GPU load"; Phase 1's sustained runs read 100%. Load, clock and
power are now stored only for windows of 3 s or more, and NULL otherwise.

---

## 04 - Not fitting

`04_offload_sweep.py`. llama.cpp places whole layers. `-ngl N` puts N on the
GPU, counted from the top: `-ngl 8` is 7 transformer layers **plus the output
layer** (log: `offloading 7 repeating layers`, `offloading output layer`).
The rest run on the CPU from system RAM, and the hidden state crosses PCIe
at each boundary.

### Part A - qwen2.5:14b, every split

The 14B (9 GB) fits, so every split is measurable. The prediction has no
fitted parameters. Each device contributes its share of the per-token time:

```
t(f) = f / r_gpu + (1 - f) / r_cpu        f = fraction of the 49 layers on GPU
```

with `r_gpu` and `r_cpu` the all-GPU and all-CPU rates from the same run.

| layers on GPU | f | decode | predicted | error | prefill | VRAM |
|---|---|---|---|---|---|---|
| 0/49 | 0.00 | 4.9 | 4.9 | - | 28.6 | 0 MB |
| 8/49 | 0.16 | 6.3 | 5.8 | +9% | **465** | 2,028 |
| 16/49 | 0.33 | 7.6 | 7.0 | +8% | 522 | 3,375 |
| 24/49 | 0.49 | 9.5 | 8.9 | +6% | 608 | 4,741 |
| 32/49 | 0.65 | 12.7 | 12.1 | +5% | 712 | 6,107 |
| 40/49 | 0.82 | 19.7 | 19.0 | +4% | 854 | 7,473 |
| 44/49 | 0.90 | **26.5** | 26.5 | 0% | 954 | 8,146 |
| 46/49 | 0.94 | 33.1 | 33.1 | 0% | 1,025 | 8,511 |
| 48/49 | 0.98 | 44.1 | 44.1 | 0% | 1,090 | 8,876 |
| 49/49 | 1.00 | **52.8** | 52.8 | - | 1,137 | 9,041 |

**Decode is exactly the sum of its parts.** Within 0-9% at every point, and
the small overshoot at low f is the output layer: it is in the first 8 and is
larger than an average layer, so `f` by layer count understates the GPU's
share of bytes.

**The curve is harmonic, and that is the practical lesson.** Ten percent of
the layers on the CPU (44/49) halves decode speed. Half the layers on the GPU
gives 9.5 tok/s: half the model's VRAM buys only twice the CPU's rate. A model that "almost
fits" runs at a fraction of the speed of one that fits, and most of the
benefit of a GPU arrives in the last few layers. The time per token is
dominated by whatever runs on the slow device.

**Prefill is a different story: one layer on the GPU changes everything.**
Zero layers: 28.6 tok/s. Eight layers: 465 (16x), with 84% of the weights
still in system RAM. The CPU could not have processed its 41 layers that fast:
at its measured rate those alone take ~35 s, and the whole prompt took 2.7 s.
So the CPU-resident layers' weights are being sent over PCIe for the GPU to
run the large prompt batches, llama.cpp's operation offload. It is a good
trade at prefill: one weight transfer serves a 512-token batch.

Each extra layer left on the CPU adds ~38 ms to this prefill. That is ~185 MB
of weights x 3 batches in 38 ms, about **15 GB/s**: half the 28 GB/s measured
for pinned memory, consistent with copying from ordinary pageable RAM.
(Derived from timings, not observed: these prefill windows are under 3 s, so no
GPU telemetry was stored.)

At zero layers this does not happen, as 03 showed. The cause was not traced:
the GPU backend is loaded in both cases.

Decode cannot use the same trick. One token's worth of work per weight
transfer is far worse than computing on the CPU.

### Part B - qwen2.5:32b, the model that does not fit

19.9 GB of weights, 16 GB card. This is the model from the user's agent log,
whose offload cost was never examined. Ollama's own choice first, then forced
placements around it, and finally all 65 layers forced onto the GPU. 2 rounds
x 2 trials. The prediction scales part A's 14B rates by weight size
(19.85 / 8.99 GB): 2.2 tok/s on the CPU, 23.9 if it all fit.

| layers | runner's VRAM claim | board VRAM peak | GPU load, decode | decode | predicted | prefill | system RAM |
|---|---|---|---|---|---|---|---|
| 33 | 10.3 GB | 13.3 GB | 12% | 4.2 | 4.1 | **285** | 25.6 GB |
| 41 | 12.6 | 15.2 | 18% | 5.3 | 5.2 | 126 | 25.4 |
| **49 (Ollama default)** | 15.0 | 15.5 | 39% | **5.7** | 7.0 | 90 | 28.5 |
| 53 | 16.1 | 15.6 | 55% | 5.2 | 8.5 | 94 | 28.9 |
| 65 (all) | 19.7 | 15.7 | **99%** | 4.0 | 23.9 | 96 | **31.3 of 32** |

The model holds at 33 and 41 layers, then fails exactly where the board runs
out. From 49 layers up, board VRAM is pinned at the card's physical limit
while the runner believes it holds more. The difference lives in system RAM.
**Windows does not fail the allocation, it pages.** WDDM lets an
oversubscribed allocation succeed and moves the excess over PCIe on demand,
which Phase 4 first saw as 8 tok/s with two backends loaded.

The all-layers row shows everything this phase is about in one place:
Ollama reports "100% GPU", the GPU reports **99% load at 2.7 GHz**, and it
produces 4.0 tok/s - slower than 33 layers with two-thirds of the card idle.
The GPU is busy waiting for memory that is on the wrong side of the bus. Busy
is not efficient, and "100% GPU" is a placement claim, not a measurement.

Prefill degrades earlier (126 at 41 layers) because prompt batches need extra
buffers on top of the weights. That headroom is gone first.

**Why Ollama's default oversubscribes.** llama.cpp's placement step logs:

```
common_params_fit_impl: projected to use 16109 MiB of device memory
                        vs. 16211 MiB of free device memory
```

On Windows the ROCm runtime reports **~15.8 GiB free** while the desktop and
browser hold ~2 GB (LibreHardwareMonitor's settled baseline: 1.9-2.3 GB). Every
placement decision assumes ~2 GB that does not exist, so the default lands
in the paging zone. On a headless Linux server the same logic would be right.
Here the default decodes at 5.7 tok/s where the model predicts 7.0, and
prefills at 90 tok/s where 33 layers manage 285.

**What it costs the agent.** Per agent turn of 2,000 new prompt tokens plus a
200-token reply:

| setup | turn time |
|---|---|
| qwen2.5:14b, fully on GPU | **5.5 s** (1.8 prefill + 3.8 decode) |
| qwen2.5:32b, Ollama default | 55.5 s |
| qwen2.5:32b, best measured split (41) | 52.9 s |

The 32B makes every turn **~10x slower** than a 14B that fits, and no layer
split fixes that: the best measured split is 5% faster than the default. Agent runs of dozens of
turns spend minutes per task on the model alone. Whether the 32B's answers are
worth 10x is a quality question this phase does not answer. Phase 10's
benchmark can.

Loading also costs more: 18-23 s for the 32B against ~9 s for the 14B (SSD to
RAM, then RAM to VRAM over PCIe).

---

## 05 - Same runtime, two GPU libraries

`05_backend_swap.py`. Ollama 0.40.7, the same model files, the same card. The
only difference is which ggml backend llama-server loads, ROCm
(`ggml-hip.dll`) or Vulkan. `side_server.py` hides ROCm to get Vulkan. Ollama
does this itself when no ROCm device is found. 3 rounds x 2 trials, library
order alternating between rounds. Decode 256 tokens, prefill ~1,240.

| model | decode ROCm | decode Vulkan | ratio | prefill ROCm | prefill Vulkan | ratio | footprint ROCm / Vulkan |
|---|---|---|---|---|---|---|---|
| 1.5B | 235 | **255** | 1.08x | 7,860 | 2,134 | 0.27x | 1,462 / 1,134 MB |
| 7B | **95.7** | 64.8 | 0.68x | 2,289 | 932 | 0.41x | 5,003 / 4,181 MB |
| 14B | **52.5** | 40.5 | 0.77x | 1,108 | 491 | 0.44x | 9,558 / 8,898 MB |
| 1.5B, 10 of 29 layers | 59.6 | 61.9 | 1.04x | 2,883 | 1,463 | 0.51x | - |

Decode spreads are 0-7% across rounds, so every decode difference above is
real. Prefill spreads reach 30-50% on the 1.5B, whose prompt takes ~0.2 s, but
ROCm's 2.3-3.7x lead is far outside them.

**Prefill is where the vendor library pays off.** Prefill is big matrix
multiplies. ROCm hands those to rocBLAS/hipBLASLt, which use RDNA3's matrix
(WMMA) instructions with kernels tuned per architecture. That is most of the
977 MB. Vulkan runs llama.cpp's own generic shaders. 2.3-3.7x is what tuned
GEMM libraries are worth on this card, and it is the compute-side
counterpart of 02's 80% FP16 efficiency.

**Decode is closer, and on small models Vulkan wins.** Decode is matrix x
vector, bandwidth-bound, with no big GEMMs for rocBLAS to accelerate. It
comes down to llama.cpp's own kernels on each backend and the fixed per-token
overhead. Vulkan's is lower on the 1.5B (+8%). ROCm's kernels stream the
bigger models' weights better (69-73% of measured bandwidth vs 47-56%). The
7B gap (0.68x) is wider than the 14B's (0.77x). Not explained here.

**Vulkan uses 300-800 MB less VRAM** for the same model and context. The
difference is the ROCm runtime and its math libraries' workspaces, which are
resident whenever ROCm is in use. On a card where the 32B already pages, that
is not nothing.

**The libraries produce different text.** At temperature 0 with the same seed,
ROCm and Vulkan outputs agree for 170-450 characters, then diverge for good.
Different kernels sum floating-point values in different orders, a near-tie
between two tokens flips, and from there the text is different. Phase 6's
fidelity method would treat the two as slightly different "quantizations" of
the same model. More surprising: **ROCm alone produced 3-4 distinct outputs
from 6 identical requests** (see open items).

**A smoke-test claim withdrawn.** An early check had Vulkan at 13 tok/s
against ROCm's 120 at 10 layers, which looked like a large partial-offload
penalty for Vulkan. Both were 3-token "Five." answers (see 03). Measured
properly the two are level (59.6 vs 61.9).

**Verdict for this card:** ROCm, for prefill above all - an agent's long
prompts are prefill. Vulkan is a credible fallback that needs no vendor
platform. On small models it decodes slightly faster and saves VRAM.

---

## How GPU inference works, in this lab's numbers

**Loading.** Weights go from SSD to system RAM, then over PCIe (28 GB/s
measured) into VRAM. Ollama hands the model to a llama-server child process,
which loads one GPU backend and places layers: `-ngl` from the request, or its
own estimate. 9 s for a 9 GB model, 20 s for the 32B. Phase 1's first cold
request spent 84% of its time here.

**VRAM holds three things.** Weights (the file size), the KV cache (28 KB per
token on the 1.5B, preallocated for `num_ctx` x slots, Phases 1 and 6), and
the runtime's working buffers (compute buffers, plus ROCm's library workspaces:
300-800 MB more than Vulkan). Whether a model "fits" is the sum of all three
against what is actually free. On Windows the runtime's idea of "free" is ~2
GB wrong.

**Decode is a memory problem.** Each token reads every weight once, so tokens
per second ~ bandwidth / model size. The card can read 602 GB/s, the CPU 46.6.
The CPU uses ~85-90% of its bandwidth at every model size. The GPU uses only
36% on a 1.5B and ~73% on a 14B: fixed per-token overhead dominates a small
model's forward pass. So the GPU's advantage grows with model size (6x ->
11x).

**Prefill is a compute problem.** A prompt's tokens are processed together,
so each weight read serves hundreds of tokens and matrix units become the
limit. The GPU's lead is 30-40x here, and the vendor library matters most:
ROCm's tuned GEMMs prefill 2.3-3.7x faster than Vulkan's generic shaders on
the same card.

**GPU utilisation is three different numbers.** `ollama ps` "100% GPU" is
placement: where the weights were put. Driver "load" is busy time: the 32B
showed 99% load while paging at 4 tok/s. Efficiency is achieved bandwidth or
FLOPs against measured ceilings: decode reaches 36-74%, and that is the one
that predicts speed. Windows' built-in counters show none of them for ROCm
(Phase 1).

**When it does not fit, there are three outcomes:**

1. **Split** (llama.cpp layers). Decode time is the sum of each device's share,
   so it is harmonic: 10% of layers on the CPU halves speed. Prefill copies
   CPU-resident weights to the GPU per batch and stays far faster than CPU-only.
2. **Page** (Windows WDDM). The allocation succeeds, the excess lives in system
   RAM, and every access crosses PCIe. Nothing errors. It reports "100% GPU" and
   99% load and runs slower than a split.
3. **Fail** (out-of-memory). The honest outcome, and the one this machine
   never produced.

**Threads on the CPU.** Decode saturates RAM at 3-4 of 6 cores, and SMT
threads make it worse. Prefill scales with every thread.

---

## The code

```
phase7-gpu-environments/
  01_stack_inventory.py   what each layer is here, and what the runtime picked
  02_bandwidth.hip        HIP probe: VRAM read/copy across sizes, PCIe both ways
  02_ceilings.py          builds/runs the probe in WSL, PyTorch matmul + CPU probes
  03_cpu_vs_gpu.py        gpu / ngl0 / cpu-only, and the CPU thread sweep
  04_offload_sweep.py     14B layer sweep against the harmonic model; the 32B
  05_backend_swap.py      ROCm vs Vulkan, same runtime and files
  side_server.py          an Ollama on :11435 pinned to ROCm, Vulkan or CPU
  harness.py              the shared measurement (placement from the runner log,
                          telemetry, storage)
```

Storage: a `hardware_ceilings` table for the probes. `inference_runs` gained
`num_gpu_layers`, `server_size_mb`, `gpu_library` and `num_thread`.
`run_once()` gained `extra_options` for `num_gpu` / `num_thread`.

---

## NVIDIA: CUDA and its ecosystem

```
NVIDIA GPU
    |
Driver (kernel module + user-mode driver; nvidia-smi)
    |
CUDA runtime / driver API      launch kernels, allocate memory, streams, graphs
    |
CUDA libraries                 cuBLAS / cuBLASLt (matmul), cuDNN (NN primitives),
                               NCCL (multi-GPU), CUTLASS (GEMM templates),
                               TensorRT / TensorRT-LLM (inference compiler)
    |
Frameworks and runtimes        PyTorch, vLLM, llama.cpp, ...
```

CUDA is three things under one name: a **programming model** (C++ with
`__global__` kernels, threads in blocks in grids, warps of 32), a **compiler**
(`nvcc`, emitting PTX - a virtual ISA - and SASS, the real one), and a
**library ecosystem**. The ecosystem is the moat. Nothing in NVIDIA's
hardware is unique in kind. The advantage is close to twenty years of tuned
kernels, profilers (Nsight), and the fact that every framework was written
against CUDA first. When a project says "GPU support" it has usually meant
"CUDA support". In this lab, Phase 3's vLLM comparison came down to one
hand-written ROCm kernel, which is the same story on a small scale.

Ollama's install carries two CUDA builds (12 and 13) that never run on this
machine. They are there because most users have NVIDIA cards.

Not measured: there is no NVIDIA card here.

---

## AMD: ROCm and HIP

```
AMD GPU (gfx1101)
    |
Driver (amdgpu on Linux; Adrenalin/WDDM on Windows; ROCDXG bridge in WSL)
    |
ROCm runtime (HSA / ROCr)       queues, memory, kernel dispatch
    |
HIP                             the CUDA-shaped API and language (hipcc / clang)
    |
ROCm libraries                  rocBLAS / hipBLASLt (cuBLAS), MIOpen (cuDNN),
                                RCCL (NCCL), Composable Kernel (CUTLASS),
                                rocprof (Nsight), amd-smi / rocm-smi (nvidia-smi)
    |
Frameworks and runtimes         PyTorch (ROCm build), vLLM, llama.cpp
```

**ROCm** is the platform: driver interface, runtime, compilers, libraries,
tools. It is AMD's counterpart to the whole CUDA stack, not to one piece of it.

**HIP** is the part developers touch. It is deliberately CUDA-shaped:
`02_bandwidth.hip` uses `__global__`, `blockIdx`, `hipMalloc`, `hipMemcpy`,
`hipEventRecord`. Replace `hip` with `cuda` and it compiles with `nvcc`. AMD's
`hipify` tools do that translation mechanically for existing CUDA code, and
HIP can itself compile for NVIDIA through a thin header layer. llama.cpp's
ROCm backend is its CUDA backend built this way. HIP is source-level
compatibility: you recompile.

What differs in practice, all seen in this lab:

- **Per-architecture builds.** ROCm libraries ship kernels compiled for
  specific `gfx` targets. A card whose target is missing falls back to slow
  paths or does not work. Official support has centred on datacenter
  (Instinct) and top consumer cards. gfx1101 works, but the path there had
  gaps.
- **Wavefront size.** RDNA (consumer) runs waves of 32, like NVIDIA's warps.
  CDNA (datacenter) runs 64. Kernels tuned for one are not optimal on the
  other. `Wave Size: 32` is in llama.cpp's device line.
- **Kernel coverage.** vLLM's single-stream lead on this card is one
  hand-written ROCm kernel (`wvSplitK`, skinny GEMM) that covers batches 1-5
  only (Phase 5).
- **Tooling on Windows/WSL.** `rocprof`, `amd-smi` and `rocminfo` all fail
  under WSL, and the WSL setup needed a rocprofiler stub (Phase 3). GPU
  telemetry comes from LibreHardwareMonitor on the Windows side (Phase 1).

---

## Vulkan: the vendor-neutral route

Vulkan is a cross-vendor graphics and compute API: every modern GPU driver
implements it. llama.cpp's Vulkan backend is its own set of compute shaders,
compiled by the driver at load time. That is why it is 40 MB against ROCm's
977 MB. It needs no vendor platform and runs on AMD, NVIDIA, Intel and mobile
GPUs alike. It cannot call vendor-tuned libraries like rocBLAS. How much that
costs on this card is section 05.

---

## ZLUDA: where it fits

ZLUDA is a **translation layer for CUDA binaries**. It implements the CUDA
driver API on top of a non-NVIDIA GPU, so a program compiled only for CUDA
can run unmodified. It translates the program's PTX for the target GPU, and
maps calls to libraries like cuBLAS onto their ROCm counterparts.

```
CUDA application binary (unchanged)
    |  calls libcuda / cuBLAS / ...
ZLUDA  (pretends to be CUDA; translates PTX, forwards library calls)
    |
ROCm / HIP runtime
    |
AMD GPU
```

It is **not** AMD's native equivalent of CUDA - that is ROCm and HIP. The
difference:

- **HIP** is source compatibility. The software is ported and recompiled, and
  ships a real ROCm build (as llama.cpp, PyTorch and vLLM do).
- **ZLUDA** is binary compatibility. Nothing is recompiled. It exists for
  software that only ships CUDA builds and will not be ported.

It began as an independent project and was funded by AMD for a period. AMD
stopped, and it continues independently. NVIDIA's CUDA licence terms
restrict translating CUDA output to run on other hardware, so its standing is
not settled. It is not something to build on. Here it would add nothing:
every runtime in this lab already has a native ROCm build, so a ZLUDA result
would measure ZLUDA, not the card. Not installed, by decision (see the plan).

---

## Limitations

- GPU probes run in WSL, inference runs on native Windows (see 02).
- No NVIDIA hardware; the CUDA sections are explanation, not measurement.
- No kernel-level profiling: `rocprof` is unavailable under WSL.
- Sensor readings refresh once a second, so load/clock/power exist only for
  runs of 3 s or more. All-GPU decode runs are shorter than that, so their GPU
  load is NULL.
- The CPU figures are Q4_K_M only. Other quantizations change the CPU's
  integer-dot-product path and were not measured.
- The 32B's quality was not measured, only its speed.

---

## Open items

- **Non-deterministic greedy output on ROCm.** 6 identical requests
  (temperature 0, seed 42) gave 3-4 distinct texts per model. Candidate cause:
  the prompt is processed differently depending on what the prefix cache
  holds, and different batch shapes round differently. Phase 6's fidelity
  method assumed determinism within a level. Its FP16 self-test scored 100%,
  so it held there, but this deserves a direct test.
- **Why `-ngl 0` does not op-offload** while `-ngl 8` does. Not traced.
- **Correct placement on Windows.** Ollama could account for the desktop's
  VRAM (LibreHardwareMonitor's board figure shows it). A practical workaround
  to test in Phase 9: set `num_gpu` from a measured free figure, or
  `OLLAMA_GPU_OVERHEAD` to reserve ~2 GB.
- **Vulkan's 7B gap** (0.68x decode, wider than the 14B's) is unexplained.
- **Quantized KV cache and flash attention** move the "fits" line for the
  32B. That belongs to Phase 9's context/KV experiments.

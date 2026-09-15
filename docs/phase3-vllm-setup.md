# Phase 3 setup - vLLM on a Radeon RX 7800 XT (gfx1101) via WSL2

Technical notes. Getting a dedicated LLM serving runtime onto consumer AMD
hardware, what broke, and what it cost.

Date: 2026-09-14
Host: Windows 11, Adrenalin 26.8.1, RX 7800 XT (gfx1101, RDNA3, 16 GB)
Guest: WSL2 Ubuntu 24.04.4, ROCm 7.2.4, Python 3.12.3

---

## Why vLLM at all

Phase 2 measured Ollama batching at **1.90x aggregate throughput at concurrency
4** on a 14B model, with per-request rate falling 46.8 -> 25.0 tok/s. Fitting a
cost model to that data gave a per-sequence overhead of ~7 ms per batch step
that barely parallelises - only the weight-streaming portion of each token is
shared between sequences.

That overhead is an implementation cost in llama.cpp, not physics. vLLM's
design attacks it directly: continuous batching, PagedAttention, and attention
kernels written for the many-sequence case. Phase 3 exists to find out whether
that translates into a real advantage on this hardware.

---

## Feasibility, checked before installing anything

Consumer RDNA3 is not vLLM's target. Two things had to be true:

- **ROCm must support this card under WSL2.** AMD's WSL compatibility matrix
  lists "AMD Radeon RX 7800 XT" explicitly for ROCm 7.2.1+.
- **vLLM must support the architecture.** vLLM's ROCm docs list `gfx1100/1101`
  (labelled "RX 7900 series" - the gfx targets are what matter, and gfx1101 is
  this card).

vLLM does **not** run natively on Windows. WSL2 is the documented route.

A version-lock concern turned out to be largely obsolete: AMD's newer WSL
compute path, **ROCDXG** (`librocdxg`), is documented as evolving independently
of both ROCm releases and the Windows display driver. The host driver was not
changed at any point.

---

## Install sequence that worked

**1. ROCm userspace.** Note that `--usecase=wsl` does *not* exist in the 7.2.4
installer despite AMD's 7.2 docs describing it - the WSL path moved to ROCDXG.

```bash
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt install ./amdgpu-install_7.2.4.70204-1_all.deb
sudo amdgpu-install -y --usecase=rocm --no-dkms
```

`--no-dkms` matters: there is no kernel module under WSL. GPU access goes
through `/dev/dxg` to the Windows driver, not `/dev/kfd`.

**2. ROCDXG**, prebuilt from the librocdxg GitHub releases:

```bash
wget https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
wget https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-amd-smi-lib_1.2.2_amd64.deb
sudo dpkg -i rocdxg-roct_1.2.2_amd64.deb rocdxg-amd-smi-lib_1.2.2_amd64.deb
```

**3. Required environment variable.** ROCm releases below 7.13 need this or no
GPU is detected at all:

```bash
export HSA_ENABLE_DXG_DETECTION=1
rocminfo | grep -i gfx      # expect: gfx1101
```

**4. PyTorch**, in a venv (Ubuntu 24.04 is PEP 668 externally-managed):

```bash
python3 -m venv ~/rocm-env && source ~/rocm-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
```

**5. vLLM**, prebuilt ROCm wheel, in its own venv so a known-good torch
survives if this one breaks:

```bash
python3 -m venv ~/vllm-env && source ~/vllm-env/bin/activate
pip install "vllm==0.29.0+rocm723" --extra-index-url https://wheels.vllm.ai/rocm/
```

**No source build was needed.** The prebuilt wheel contains working gfx1101
kernels, contrary to the expectation that AMD compiles only for Instinct
targets.

---

## The rocprofiler crash

PyTorch aborted on `torch.cuda.device_count()`:

```
W agent.cpp:608] sysfs nodes path '/sys/class/kfd/kfd/topology/nodes' does not exist
F agent.cpp:1093] Found 0 rocprofiler agents and 2 HSA agents.
                  HSA agents contained 2 internal node ids not found by rocprofiler: 0, 1
Aborted (core dumped)
```

ROCm itself was fine - it enumerated both agents. rocprofiler discovers GPUs
through `/sys/class/kfd`, which does not exist under WSL, so it found zero
agents, saw HSA reporting two, and treated the mismatch as fatal. Upstream fix
in flight: ROCm/rocm-systems PR #11423, "emit CPU agents from the WSL topology
enumerator".

**What did not work:** renaming torch's bundled `librocprofiler-sdk.so` (the
loader falls back to the system copy), renaming the system copy,
`ROCPROFILER_LIBRARY_CTOR=0`, `HSA_TOOLS_LIB=0`, `ROCP_TOOL_LIBRARIES=`. It is
a hard `DT_NEEDED` link, not an optional `dlopen`, so removing it breaks the
link rather than skipping the profiler.

**The diagnostic that settled it** was `LD_DEBUG=libs python ... | grep
rocprofiler`, which prints exactly which file the loader chose. That took
seconds; a filesystem-wide `find` was still running when the answer arrived.

**What worked:** `librocprofiler-register.so` is the gatekeeper that hands
HIP's API table to rocprofiler-sdk. A symbol in an `LD_PRELOAD` library wins
over the same symbol anywhere else, so replacing just that one function with a
no-op stub skips the whole chain without modifying any file.

```c
/* norocprof.c - returns ROCPROFILER_REGISTER_SUCCESS so the caller carries on */
int rocprofiler_register_library_api_table(const char *lib_name,
                                           const void *import_funcs,
                                           unsigned int lib_version,
                                           void **tables,
                                           unsigned long num_tables)
{
    (void)lib_name; (void)import_funcs; (void)lib_version;
    (void)tables;   (void)num_tables;
    return 0;
}
```

```bash
gcc -shared -fPIC -o ~/lab/norocprof.so ~/lab/norocprof.c
echo 'export LD_PRELOAD=$HOME/lab/norocprof.so' >> ~/vllm-env/bin/activate
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/vllm-env/bin/activate
```

Scoped to the venv's `activate` rather than `.bashrc`, so it does not attach to
every process on the system.

**Cost of this workaround: no GPU profiling.** `rocprof` and kernel-level
timing are unavailable in this environment. On an Instinct card under native
Linux they would not be.

---

## Running the server

```bash
source ~/vllm-env/bin/activate
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --served-model-name qwen2.5-1.5b-instruct \
  --dtype float16 --max-model-len 4096 \
  --gpu-memory-utilization 0.35 \
  --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` is required for the benchmark client on the Windows side to
  reach it. WSL2 forwards localhost, but only to a service bound beyond
  loopback.
- Scripts using the offline `LLM` class **must** use
  `if __name__ == "__main__":`. vLLM v1 spawns its engine core in a separate
  process which re-imports the script; without the guard, multiprocessing
  aborts before anything loads.
- `--dtype float16` because the model ships bf16 and Ollama's F16 GGUF is fp16 -
  matching the formats is what makes the comparison meaningful.

---

## Two GPU servers, one card

The first backend comparison produced an implausible result: Ollama at 8.1
tok/s against vLLM's 151.4, an 18x gap. No serving runtime is 18x faster than
another on identical weights.

`ollama ps` reported `100% GPU`, 3.3 GB. The Windows GPU counters disagreed:

```
board total     14,524 MB of 16,368
vmwp (vLLM)      9,975 MB
llama-server       456 MB    <- what Ollama actually had resident
```

vLLM had been started with `--gpu-memory-utilization 0.6`, reserving 9.6 GB for
its process lifetime. Ollama's 3.3 GB allocation *succeeded* - WDDM overcommits
GPU memory and pages the excess to system RAM - so Ollama believed it was fully
resident while the driver faulted weights back across PCIe every token.

The arithmetic confirms it:

```
PCIe 4.0 x16  ~25 GB/s effective  /  3.1 GB model  =  ~8 tok/s
measured                                               8.1 tok/s
```

Dropping vLLM to `--gpu-memory-utilization 0.35` (5.7 GB) left room for both,
and Ollama went from 8.1 to 121.4 tok/s.

**Lesson: `ollama ps` reports what was allocated, not what is resident.** On a
card with a competing tenant, a successful allocation is not evidence of
residency. Cross-check against board-level VRAM counters and against physical
throughput limits.

---

## First matched comparison

Identical weights, FP16, `/v1/chat/completions` on both, both reporting 41
prompt tokens for the same message (so the chat templates agree), 5 trials:

| | Ollama | vLLM |
|---|---|---|
| decode | 121.4 tok/s (cv 1.4%) | 152.7 tok/s (cv 1.0%) |
| effective bandwidth | 375 GB/s (60% of peak) | 472 GB/s (76% of peak) |
| TTFT | 16.9 ms | 28.6 ms |
| total wall clock | 709.8 ms | 612.2 ms |

vLLM is **1.26x faster** in single-stream decode and reaches the highest
fraction of memory bandwidth measured anywhere in this project.

**The TTFT figure is confounded** and should not be quoted as a runtime
difference: Ollama runs natively on Windows while vLLM sits behind WSL2's
virtual network, and every request crosses that boundary. Phase 1 measured ~5 ms
of pure HTTP overhead on native loopback; WSL2 adds more. Decode rate is largely
immune (a sustained stream at 6.5 ms/token), TTFT is a single round trip and is
exactly where the penalty lands.

Single-stream decode is also the workload vLLM is *least* optimised for. Its
design targets many concurrent sequences, so the concurrency comparison is the
one that matters - and it is not yet run.

### Greedy decoding is not reproducible across runtimes

With `temperature=0` and the same seed, the two backends produced different
text:

```
ollama  "A KV cache, or key-value cache, is a type of data storage system that
         stores data in key-value pairs, where each key is associated with a
         value."
vllm    "A Key-Value (KV) cache is a type of data storage system that stores
         key-value pairs, where each key maps to a unique value."
```

Within a runtime, greedy decoding is deterministic - all five trials on each
side were identical. Across runtimes it is not. Different kernels and different
reduction orders in the matmuls produce tiny floating-point differences; where
two candidate tokens are nearly tied, rounding tips the choice one way on one
backend and the other way on the other, and the sentences diverge from there.

Consequence for Phase 6: **output quality cannot be compared across backends
token-by-token.** Comparison has to be semantic, or floating-point noise will
read as a quality difference.

---

## Open items

- Concurrency comparison (the actual point of Phase 3).
- Wire the OpenAI client into the benchmark harness so vLLM runs persist to
  `lab.db` with hardware sampling, as the Ollama runs do.
- `gpu_vram_used_mb` is board-wide, not model-attributable, and needs fixing
  before quantization comparisons in Phase 6.

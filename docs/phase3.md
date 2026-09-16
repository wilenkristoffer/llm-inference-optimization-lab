# Phase 3 - Ollama vs vLLM

Technical notes. A matched comparison of two serving runtimes on the same card,
and what happened when the benchmark was made honest.

Date: 2026-09-14 to 2026-09-16
Setup and install notes: `phase3-vllm-setup.md`

Ollama 0.34.1 (native Windows, ROCm) vs vLLM 0.29.0+rocm723 (WSL2, ROCm 7.2.4).
Model: Qwen2.5-1.5B-Instruct, FP16 on both sides, 4096 context,
`/v1/chat/completions` on both, `temperature=0`.

---

## Goal

Phase 2 measured Ollama batching at 1.75x aggregate throughput at concurrency
4, with per-request rate falling 46%. A cost model fitted to that data implied
~7 ms of per-sequence overhead that barely parallelises. vLLM's design
(continuous batching, PagedAttention, batched attention kernels) attacks
exactly that. Phase 3 asks whether it delivers on consumer RDNA3.

---

## What was built

| File | Purpose |
|---|---|
| `01_openai_client.py` | One client speaking `/v1/chat/completions`, so both backends are driven by identical code. Returns the same keys as the Phase 1 client. |
| `02_benchmark_backends.py` | Single-stream comparison, persisted to `lab.db` with hardware telemetry. |
| `03_concurrency_compare.py` | Barrier-synchronised concurrency sweep against either backend. |
| `04_vllm_metrics.py` | Reads vLLM's own `/metrics` (Prometheus text) to see scheduler state during a run. |

---

## Controlling the comparison

Three things had to be equal or the result would be meaningless:

**Precision.** Ollama defaults to Q4_K_M GGUF; vLLM defaults to FP16
safetensors. Left alone, Ollama would stream 0.99 GB per token against vLLM's
3.09 GB and win by ~3x for reasons that have nothing to do with the runtime.
Fixed by pulling `qwen2.5:1.5b-instruct-fp16` and running vLLM with
`--dtype float16`.

**Chat template.** vLLM applies the template from the model's
`tokenizer_config.json`; Ollama applies one baked into its Modelfile. They
could differ. They do not - both report **41 prompt tokens** for the same
message, which is the check that proves it.

**Interface.** `/v1/chat/completions` on both, same client code, only the base
URL changes.

---

## Single-stream decode

Five trials, two separate sessions:

| | Ollama | vLLM |
|---|---|---|
| session 1 | 121.4 tok/s (cv 1.4%) | 152.7 tok/s (cv 1.0%) |
| session 2 | 124.9 tok/s (cv 0.7%) | 153.5 tok/s (cv 0.4%) |
| effective bandwidth | 375-386 GB/s (60-62% of peak) | 472-474 GB/s (76% of peak) |

**vLLM is 1.23-1.26x faster** and reaches the highest fraction of memory
bandwidth measured anywhere in this project. The absolute numbers moved ~3%
between sessions (consistent with the ~4% between-session drift measured
earlier) while the *ratio* held - which is why comparisons are run interleaved.

TTFT at concurrency 1 was 16-17 ms for Ollama and 25-29 ms for vLLM, but this
is **not a runtime difference**: Ollama runs natively on Windows while vLLM
sits behind WSL2's virtual network. Decode rate is a sustained stream and
largely immune; TTFT is a single round trip and is where the transport penalty
lands.

---

## The benchmark was wrong the first time

The initial concurrency sweep used prompts like "Write one sentence about
oceans" with a 100-token cap. The cap never bound - each request generated
**~25 tokens**. At that length, fixed per-request costs (prefill, scheduling,
HTTP round trip) dominate and batching has almost no decode to amortise.

This affected the Phase 2 Ollama sweep too, which used the same prompts. Those
numbers are internally consistent, but both backends were understated.

Fixed by asking for a 500-word description with a 300-token cap, so the cap
binds and **every request generates exactly 300 tokens** - 12x more decode, and
identical work per request.

Verified before re-running rather than assumed: three different prompts, all
returning exactly 300 output tokens.

---

## Concurrency, with 300-token requests

Aggregate = total tokens / batch wall time. Per-request = median decode rate
experienced by one request.

```
conc   vLLM agg   Ollama agg    vLLM/req  Ollama/req   vLLM ttft  Ollama ttft
   1      148.9        120.3       150.4       121.6          25          33
   2      287.0        194.3       145.7        98.2          35          42
   4      563.0        388.8       143.4        99.9          44          92
   8      563.8        647.9        71.2        83.0          50          97
```

Converting per-request rate into batch step time makes the mechanism clear:

```
             batch 1    batch 2    batch 4    batch 8
vLLM         6.65 ms    6.86 ms    6.97 ms   14.04 ms
Ollama       8.22 ms   10.18 ms   10.01 ms   12.05 ms
```

**vLLM serves four sequences in the time it takes to serve one** - 6.65 to
6.97 ms, a 5% cost for 4x the work. That is continuous batching working as
advertised, and the best scaling measured in this project.

**Then at batch 8 its step time doubles** and aggregate throughput stops dead:
563.0 to 563.8 tok/s. A hard ceiling, not a taper.

Ollama's step time grows gradually throughout, so it keeps gaining and
**overtakes vLLM at concurrency 8** (647.9 vs 563.8).

Telemetry during the sweeps:

```
             gpu%   clock      power    vram        cpu%
vllm  c=1      61   1689 MHz   143 W   15762 MB     39.9
vllm  c=4      98   2730 MHz   204 W   15774 MB     48.4
vllm  c=8      99   2690 MHz   203 W   15774 MB     50.3
ollama c=1     88   2540 MHz   218 W    5712 MB     24.0
ollama c=4     89   2632 MHz   232 W    5707 MB     21.6
ollama c=8     81   2561 MHz   197 W    5703 MB     28.7
```

vLLM saturates the GPU earlier (98-99% from concurrency 2) but stops converting
that into throughput past batch 4. Its CPU cost is also roughly double
Ollama's, which is the WSL VM overhead.

---

## Diagnosing the ceiling

Three hypotheses, all testable against vLLM's own `/metrics`:

```
concurrency 8, 300-token requests, 72 samples during the batch:

  num_requests_running   peak 8.00, mean 7.88    all eight, sustained
  num_requests_waiting   peak 0.00               nothing queued
  kv_cache_usage_perc    peak 0.01               cache 1% full
  num_preemptions_total  delta 0.0               nothing evicted
```

All three are dead. The scheduler batched all eight sequences for the entire
run, nothing queued, and the KV cache - 13.6 GB reserved - was **1% used**.
PagedAttention is nowhere near being the constraint at this model size.

So the limit is in the compute path. Neither roofline explains it:

```
batch 4   143.4 steps/s x 3.09 GB = 443 GB/s = 71% of bandwidth peak
batch 8    71.2 steps/s x 3.09 GB = 220 GB/s = 35% of bandwidth peak
```

Bandwidth consumption **halves** from batch 4 to 8 while throughput stays flat,
so it is not memory-bound at batch 8. Raw compute is not it either - 563 tok/s
on a 1.54B model is roughly 1.7 TFLOPS against a card rated for tens.

The decisive argument: **Ollama reached 647.9 tok/s on the same card.** The
hardware is demonstrably capable of more than vLLM's 563 ceiling, so that
ceiling is a software limit - some GEMM or attention kernel in vLLM's ROCm path
that tiles well up to batch 4 and falls off after.

**Further diagnosis is blocked by an earlier decision.** Kernel-level profiling
needs `rocprof`, which was disabled by stubbing rocprofiler to make vLLM run
under WSL at all (see `phase3-vllm-setup.md`). That tradeoff looked free at the
time and is exactly what prevents closing this question now.

---

## Conclusion

> On this hardware, vLLM's continuous batching is excellent up to 4 concurrent
> sequences - 4x the work for 5% more per-user latency, 1.45x Ollama's
> aggregate throughput, and half the TTFT under load. Past that its ROCm
> kernels stop scaling and llama.cpp overtakes it. This is a consumer-RDNA3
> limitation rather than a property of vLLM: the scheduler, KV cache and
> batching logic were all verified working through vLLM's own metrics.

Practical reading: if the workload is a handful of concurrent users, vLLM is
clearly better here. If it is many, this particular GPU and build do not
deliver vLLM's usual advantage, and the simpler stack wins.

---

## Corrections made in this phase

**1. "LibreHardwareMonitor cannot see WSL GPU activity" - wrong.** The first
vLLM sweep recorded `gpu% 0.0` and a 2 MHz clock, and this was attributed to
the paravirtualised `/dev/dxg` path hiding work from AMD's driver API. It was a
**sampling artefact**: those batches finished in 0.2-0.5 s, faster than LHM
refreshes its sensors, so the samples captured idle readings between batches.
With 300-token requests the same sensors report 98-99% GPU and 2700 MHz. Same
class of error as Phase 1's diluted "11% GPU load".

**2. The first backend comparison was invalid.** vLLM had been started with
`--gpu-memory-utilization 0.6`, reserving 9.6 GB. Ollama's allocation still
*succeeded* because WDDM overcommits, so `ollama ps` reported `100% GPU` while
only 456 MB was actually resident - the rest paged over PCIe every token.
Measured 8.1 tok/s against a PCIe-bound prediction of ~8. Detail in
`phase3-vllm-setup.md`.

**3. Requests were too short to measure batching.** Covered above.

---

## Limitations of this environment

- **No kernel profiling.** `rocprof` is unavailable because rocprofiler is
  stubbed out; `amd-smi` fails under WSL (no `amdgpu` kernel module).
- **TTFT is not comparable across backends.** vLLM pays WSL2 network
  transport that native Ollama does not.
- **One model size.** Only the 1.5B fits comfortably in FP16 on 16 GB. A 7B
  FP16 is ~15.2 GB before KV cache. Conclusions may not transfer to the model
  sizes where batching matters most.
- **Short contexts.** 4096 tokens with a 1% full KV cache means PagedAttention
  was never tested under the pressure it exists to handle.

---

## Open items

- Whether vLLM's batch-8 ceiling moves with `--max-num-seqs`,
  `--max-num-batched-tokens`, or a different attention backend.
- Whether the ceiling scales with model size (does a 3B hit it at batch 4 too?).
- Store vLLM's `/metrics` per run rather than only reading them live (Phase 4).
- `gpu_vram_used_mb` is board-wide, not model-attributable - needs fixing
  before Phase 6's quantization comparisons.

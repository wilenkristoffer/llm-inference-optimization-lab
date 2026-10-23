# Phase 9 plan - Inference optimization

Status: **planned 2026-10-23, not started.** This is a plan, not results.
Predictions are written down so the measurements can prove them wrong.

The brief lists context length, concurrency, batching, KV cache and model
size. Phases 1-8 have measured the baseline for each. This phase changes the
settings that move them, and measures what each change buys and costs on
this card. Speculative decoding isn't in the brief, but it's the most
interesting optimization available here, so it goes first.

Every stage reports three things: what it gains (tok/s, TTFT, VRAM), what it
costs (VRAM, quality, latency elsewhere), and what it means for the user's
agent (`lab-findings-for-agentic-work`).

---

## Starting point (measured, no new runs needed)

| topic | baseline | from |
|---|---|---|
| decode, Ollama Q4_K_M | 1.5b 222, 7b 83, 14b 44 tok/s | Phase 1 |
| decode, vLLM FP16 1.5B | 147-153 tok/s single stream; skinny GEMM only at 1-5 tokens per step | Phase 3 |
| batching | Ollama 1.75x at NUM_PARALLEL 4; vLLM 2,052 tok/s at 32 | Phases 2, 3 |
| prefill | Ollama ~8,400 tok/s + 21.7 ms; vLLM ~11,600 tok/s + 10.0 ms | Phase 5 |
| prefix cache | Ollama 42.4 -> 7.5 ms on a cached 40-token prefix | Phase 5 |
| KV cache | 1.5B ~28 KB/token at FP16; preallocated `num_ctx x num_parallel` | Phases 1, 5 |
| fit | 14B fully on GPU 5.5 s per agent turn; 32B pages over PCIe, ~10x slower | Phase 7 |
| quality method | top-20 KL + top-1 agreement vs a reference, 30 prompts x 64 tokens | Phase 6 |

Models on disk: Ollama has `qwen2.5:0.5b/1.5b/3b/7b/14b` (Q4_K_M), the
1.5B in FP16, and the 3B/7B in Q8_0. vLLM's HF cache has only
`Qwen2.5-1.5B-Instruct`. The whole Qwen2.5 family shares one tokenizer, which
speculative decoding requires.

---

## Pre-checks (done 2026-10-23)

1. **llama.cpp speculative decoding is available without installing
   anything.** Ollama's bundled `lib/ollama/llama-server.exe` (llama.cpp
   build a37cc4610) has `-md/--model-draft`, `--spec-draft-n-max` (default 3),
   `--spec-draft-n-min`, `--spec-draft-p-min`, and a separate KV type for the
   draft. Ollama's own API doesn't expose a draft model, so Stage 1 runs
   `llama-server` directly on Ollama's GGUF blobs, on a side port like
   Phase 7's `side_server.py`.
2. **vLLM 0.29 methods:** `draft_model`, `ngram`, `ngram_gpu`, `suffix`,
   `eagle`/`eagle3`, `medusa`, and several MTP variants. Whether they run on
   gfx1101 under ROCm isn't checked. That's Stage 0.
3. **Disk:** C: has 62 GB free (down from 237 GB; Docker images account for
   most of that). The WSL distro has 879 GB free. Stage 1 needs about 8 GB of
   HF downloads in WSL.

---

## Stage 0 - Can each piece run here? (go/no-go, short)

- `llama-server -m <14b blob> -md <0.5b blob>` serves and reports draft
  acceptance.
- vLLM `--speculative-config '{"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":3}'`
  on a 3B target, and `{"method":"ngram",...}` on the 1.5B. Download
  `Qwen2.5-0.5B-Instruct` (1 GB) and `Qwen2.5-3B-Instruct` (6 GB). A 7B at
  FP16 (15 GB) doesn't fit; quantized vLLM formats are a separate question
  (see Optional).
- llama.cpp flash attention plus a `q8_0`/`q4_0` KV cache on ROCm gfx1101.
  With flash attention off, a quantized V cache isn't allowed.
- **Greedy determinism baseline.** Phase 7 got 3-4 distinct texts from 6
  identical temperature-0 requests. To show that speculative decoding is
  lossless, the target's own run-to-run agreement has to be known first.
  Measure it once per backend: 10 identical requests, and how many tokens in
  before the outputs diverge.

Prediction: llama.cpp works as-is. vLLM `ngram` works. vLLM `draft_model`
works but may need `--enforce-eager` or a different attention backend on
ROCm, and EAGLE needs trained heads that don't exist for Qwen2.5-3B (skip
it).

---

## Stage 1 - Speculative decoding (the headline)

A small draft model proposes K tokens, and the target checks all K in one
forward pass. Decode is bandwidth-bound, so checking K+1 tokens costs about
the same as generating 1. The output is the target's own (lossless).

Expected speedup with acceptance rate `a`, K drafted tokens, and draft cost
`c` (draft step time / target step time):

```
speedup = (1 - a^(K+1)) / (1 - a)   /   (1 + K*c)
           tokens per target pass        cost per pass
```

**llama.cpp (Ollama's binaries)**, draft `qwen2.5:0.5b` Q4_K_M:

| target | c (est.) | prediction at a=0.7, K=4 | at a=0.5 |
|---|---|---|---|
| qwen2.5:14b Q4_K_M | ~0.11 | **1.9x** | 1.35x |
| qwen2.5:7b Q4_K_M | ~0.21 | 1.5x | ~1.05x |
| qwen2.5:3b Q4_K_M | ~0.35 | ~1.1x | a loss |

**vLLM**, draft `Qwen2.5-0.5B-Instruct` FP16 -> target `Qwen2.5-3B-Instruct`
FP16, and `ngram` (prompt lookup, no draft model) on the 1.5B.

Workloads, because the speedup depends entirely on acceptance:
- **code**: predictable;
- **extraction**: JSON fields copied out of a document in the prompt, which
  is the agent's tool-result case and the one ngram is built for;
- **open prose**: unpredictable.

Measured per run: tok/s, acceptance rate, the K sweep (1-8), and the outputs
compared with the non-speculative baseline, judged against the Stage 0
determinism figure.

Predictions:
1. **A cliff caused by the lab's own finding.** In vLLM on ROCm the check
   step processes K+1 tokens per sequence, and the skinny GEMM covers only
   1-5. So `num_speculative_tokens` 5 or more pushes every check step onto
   the generic kernel, which runs at about half per-token speed. The best K
   in vLLM will be at most 4, whatever the acceptance.
2. ngram: 1.5-2.5x on extraction, about 1.0x on prose.
3. Speedup shrinks with concurrency and becomes a loss by batch 16-32 in
   vLLM. Batching already fills the bandwidth that speculation exploits, and
   checking multiplies the tokens per step by K+1.
4. Acceptance: code > extraction > prose. Prose below 0.5 at temperature 0
   with a 0.5B draft.
5. VRAM: the draft adds its weights plus its own KV cache, about 0.6 GB for
   the 0.5B in llama.cpp.

---

## Stage 2 - Context length and the KV cache

Each generated token reads all the weights *and* the whole KV cache so far.
So decode should slow down as the context fills, by the same bandwidth
arithmetic as Phase 1:

```
tok/s(ctx) ~= tok/s(0) * weights / (weights + KV_bytes_per_token * ctx)
```

| model | weights | KV at FP16 | predicted decode at 16k | at 32k |
|---|---|---|---|---|
| qwen2.5:1.5b Q4_K_M | 0.99 GB | 28 KB/token | 151 tok/s (-32%) | 115 tok/s (-48%) |
| qwen2.5:14b Q4_K_M | 9.0 GB | ~196 KB/token | 32 tok/s (-26%) | doesn't fit at FP16 |

Part A: decode tok/s at context depths 0.5k / 2k / 8k / 16k / 32k, on Ollama
and on vLLM (1.5B). A depth here is a real prompt of that length, then 128
generated tokens.

Part B: KV cache quantization, Ollama `OLLAMA_FLASH_ATTENTION=1` +
`OLLAMA_KV_CACHE_TYPE` = f16 / q8_0 / q4_0:
- VRAM per token, measured with the Phase 6/7 evict-and-settle method.
- Decode at each depth. Prediction: q8_0 recovers about half of the
  long-context slowdown, because it halves the KV bytes read.
- Quality: the Phase 6 fidelity method, but on long prompts, since KV error
  matters most when there's a lot of it. Prediction: q8_0 barely measurable,
  q4_0 visible.
- Prediction: the 14B at q8_0 fits a full 32k context on the GPU, and at f16
  it stops at about 18k.

---

## Stage 3 - Prefix caching, measured properly

Resolves the open item from Phases 3 and 5: their TTFT figures were
warm-cache numbers, and nobody has measured cached vs uncached TTFT on both
backends in one session.

An agent-shaped prompt: a 2,000-token stable system prompt plus tool
definitions, then a short varying user turn. Cold, warm, and "warm but one
token changed near the start". Both backends, interleaved.

Predictions:
- Cold TTFT about 260 ms on Ollama and about 180 ms on vLLM. Warm is close
  to the fixed costs: about 25 ms on Ollama, about 15 ms on vLLM.
- One changed token at position 10 costs the full cold TTFT. That is the
  lab's explanation for the agent's 23% cache hit rate, made measurable.
- **With `OLLAMA_NUM_PARALLEL` > 1, cache hits drop.** Each slot holds its
  own KV cache, and a request that lands in a different slot from last time
  misses. vLLM's prefix cache is shared across sequences, so it shouldn't
  have this problem.

---

## Stage 4 - Concurrency and batching settings

- **Ollama at `NUM_PARALLEL` 8 / 16 / 32**, the open item from Phase 5, for a
  fair comparison with vLLM past its cliff. Prediction: aggregate peaks
  around 900-1,100 tok/s on the 1.5B, half of vLLM at 32. Ollama overtook
  vLLM at 8 (648 vs 570), and vLLM wins by 16.
- **vLLM `--max-num-seqs` and `--max-num-batched-tokens`**: what they trade
  between throughput and latency.
- **Inter-token latency under mixed load**: steady decodes while a
  4,000-token prompt arrives. Prediction: Ollama's decodes stall for the
  whole prefill (about 0.5 s). vLLM's chunked prefill limits the stall to
  one chunk, at most tens of ms. This is the measurement Phase 5's ITL work
  couldn't make with a single stream.
- **The cliff on a bigger model**: the vLLM batch sweep on the 3B.
  Prediction: still at 6 (it's the kernel's switch, not the model's), but
  smaller in relative terms, because the 3B uses bandwidth better.

---

## Optional (only if time allows)

- **32B placement on Windows**: `OLLAMA_GPU_OVERHEAD` about 2 GB, or
  `num_gpu` set from a measured free figure, to stop WDDM paging (Phase 7).
  Prediction: a correct split runs the 32B at 6-8 tok/s instead of 4.
- **vLLM quantized formats** (AWQ/GPTQ) on gfx1101, which would let vLLM run
  a 7B target for Stage 1. Unknown whether the ROCm kernels support RDNA3.
  FP8 has no hardware support on RDNA3.
- **vLLM `--cpu-offload-gb`**, deferred from Phase 7.

## Out of scope, deliberately

- **EAGLE/Medusa heads**: they need trained heads for the exact target, and
  none exist for Qwen2.5-3B. Training one is a Phase 11-sized job.
- **Kernel-level tuning** (custom GEMMs, attention backends beyond what's
  selectable): no `rocprof` under WSL, so it can't be profiled.
- **Model size as its own experiment**: Phase 1's decode ladder and Phase 7's
  fit results already cover it. Stage 1 and Stage 2 reuse those models
  rather than adding new ones.

---

## Order and size

Stage 0 is short and decides how much of Stage 1 runs on vLLM. Stages 1-4
are independent after that and all run (full scope confirmed 2026-10-23).
Write-up: `docs/phase9.md`, code in `phase9-optimization/`, same
format as earlier phases.

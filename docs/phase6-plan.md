# Phase 6 plan - Quantization

Status: **planned, not started.** This is a plan, not results. Sizes and
timings below are estimates to be checked, not measurements.

Every model run so far except the FP16 comparisons was Q4_K_M, because that is
what Ollama's default tags are. This phase asks what that choice costs and what
it buys, on this card, measured rather than taken from llama.cpp's published
tables.

---

## Questions

1. **Quality.** How far does each quantization level drift from FP16, and at
   which level does the drift start to change answers?
2. **Memory.** What does each level actually occupy in VRAM, against its file
   size?
3. **Speed.** Does decode rate track weight size, as the Phase 1 bandwidth
   model predicts - and does lower precision cost anything in dequantization
   that the bandwidth model misses?
4. **The practical one.** For a 16 GB card, which model size and quantization
   gives the best quality that fits? A bigger model at Q4 or a smaller one at
   Q8?

---

## What is already in place

- **Clean VRAM footprints.** `05_benchmark.py` evicts every Ollama model, takes
  a settled baseline, and marks the row `gpu_vram_baseline_clean = 1`.
  Validated 2026-10-22: Ollama's own figure runs a constant ~270-320 MB below
  the board, so the measurement is sound and the gap is runtime overhead.
- **Slot count recorded.** `server_num_parallel` is read from the llama-server
  command line. Every slot preallocates FP16 KV cache: at 8 slots the 1.5B Q4
  footprint was 2,236 MB, at 1 slot 1,382 MB. **All footprints in this phase
  are measured at 1 slot.** Speed runs are unaffected by the slot count.
- **Sampling pinned.** `temperature 0`, `seed 42`, `repeat_penalty 1.0`.
- **Token probabilities.** Ollama 0.40 returns `logprobs` with
  `top_logprobs`, which makes the fidelity measurement below possible.

---

## Models

All `-instruct` variants, so the only variable is the weights' precision.

**Primary ladder: qwen2.5:3b, every level.** Every level fits with room to
spare, including FP16, so the full curve is measurable on one model.

| tag | approx. size | have it? |
|---|---|---|
| `qwen2.5:3b-instruct-q2_K` | 1.3 GB | no |
| `qwen2.5:3b-instruct-q3_K_M` | 1.6 GB | no |
| `qwen2.5:3b-instruct-q4_0` | 1.8 GB | no |
| `qwen2.5:3b` (= q4_K_M) | 1.9 GB | yes |
| `qwen2.5:3b-instruct-q5_K_M` | 2.2 GB | no |
| `qwen2.5:3b-instruct-q6_K` | 2.5 GB | no |
| `qwen2.5:3b-instruct-q8_0` | 3.3 GB | no |
| `qwen2.5:3b-instruct-fp16` | 6.2 GB | no |

**Secondary: qwen2.5:7b, Q4_K_M to Q8_0.** Answers question 4 - is a 7B at Q4
better than a 3B at Q8, at a similar footprint?

| tag | approx. size | have it? |
|---|---|---|
| `qwen2.5:7b` (= q4_K_M) | 4.7 GB | yes |
| `qwen2.5:7b-instruct-q5_K_M` | 5.4 GB | no |
| `qwen2.5:7b-instruct-q6_K` | 6.3 GB | no |
| `qwen2.5:7b-instruct-q8_0` | 8.1 GB | no |
| `qwen2.5:7b-instruct-fp16` | 15.2 GB | no - optional, see below |

**7B FP16 is optional.** 15.2 GB of weights plus the desktop's 1.4 GB does not
fit in 16 GB. Ollama would split layers between GPU and CPU. One run would
show what "doesn't fit" costs in tok/s - the reason quantization exists - but
it is a 15 GB download for a single data point. Its quality reference is not
needed: the 7B ladder can be compared against 7B Q8_0, which is close to
lossless in llama.cpp's own measurements (to be checked here on the 3B).

Downloads: ~19 GB for the 3B ladder, ~20 GB for the 7B levels, +15 GB if 7B
FP16 is included. C: has 240 GB free.

---

## Measurements

### 1. Fidelity to FP16 - the primary quality measure

Task accuracy on a few dozen questions is a blunt instrument: with 50
questions, two levels have to differ by ~15 points before the difference is
distinguishable from noise, and quantization damage starts long before that.
So the main measure compares each level's **next-token distribution** to
FP16's, token by token, on the same text.

Method (teacher forcing, the same idea as llama.cpp's KL-divergence tool):

1. FP16 generates greedy reference continuations for ~30 varied prompts, ~64
   tokens each.
2. For each quantized level, for each position in each reference: send the
   prompt plus the reference prefix with `raw: true` (the chat template applied
   by hand, so nothing is added - see Phase 2's finding that every endpoint
   templates otherwise) and request **one** token with `top_logprobs: 20`.
3. From that, per level:
   - **top-1 agreement** - how often the quantized model's most likely next
     token is FP16's token
   - **mean log-probability of FP16's token** - equivalently, the perplexity
     of FP16's text under the quantized model
   - **approximate KL divergence** from FP16, over the top-20 tokens

Cost: ~2,000 one-token requests per level. Each request extends the previous
one, so the prefix cache (Phase 5) makes them cheap - estimated ~1 minute per
level. Feasibility will be confirmed on two levels before running the ladder.

Caveat to state in the results: top-20 KL is an approximation; it ignores the
probability mass outside the top 20, which is small for confident tokens and
not small for uncertain ones.

### 2. Task accuracy - the sanity check

~60 questions with machine-checkable answers, written for this phase:

- **arithmetic word problems** - a single numeric answer
- **factual recall** - an exact answer from a short list of accepted forms
- **format following** - "return JSON with these fields", checked by parsing

No code-generation tasks: checking them means executing model-written code,
which is not worth the sandboxing for a sanity check. Greedy decoding, so each
level gets one deterministic attempt per question. Reported with the number of
questions, so a reader can see how coarse it is.

Phase 1's prompt ("explain a KV cache") stays the speed benchmark only - it has
no checkable answer.

### 3. VRAM and speed

Per level, via `05_benchmark.py` with Ollama at 1 slot, `num_ctx` 4096, 7
trials after warm-up:

- clean VRAM footprint, beside the file size and Ollama's own figure
- decode tok/s, and effective bandwidth = weight size x tok/s against the
  624 GB/s peak (Phase 1's ladder: larger models used it better)
- prefill tok/s at a fixed long prompt (Phase 5 file 02's method, one length)

**Interleaved.** Between-session drift is ~4%, about the size of some of the
differences expected here. The ladder is run as rounds - every level once per
round, in a shuffled order - rather than one level at a time.

---

## Planned files

```
phase6-quantization/
  01_quant_ladder.py     pull check, footprint + speed per level, interleaved
  02_fidelity.py         FP16 references, teacher-forced logprobs, KL/agreement
  03_task_accuracy.py    the checkable question set and its scorer
  questions.json         the questions and accepted answers
docs/phase6.md           results
```

Results go to `lab.db` like everything else. Fidelity and accuracy need new
storage - one row per (level, prompt) - so a second table rather than more
columns on `inference_runs`.

---

## Out of scope, deliberately

- **vLLM quantization** (AWQ, GPTQ, FP8). Different formats from GGUF and
  different kernels - Phase 5 found vLLM's speed depends on one ROCm kernel
  for small batches, and quantized models take other code paths. Worth its own
  look, possibly in Phase 9, not mixed into a GGUF ladder.
- **KV cache quantization** (`OLLAMA_KV_CACHE_TYPE=q8_0`). A separate lever with
  its own quality cost; a natural follow-on once the weight ladder exists.
- **Importance-matrix (imatrix) quants.** Ollama's library tags are what most
  people run, so those are what is measured.

---

## Decisions needed before starting

1. Approve the downloads (~39 GB, or ~54 GB with 7B FP16).
2. Include 7B FP16 as the "does not fit" data point, or skip it.

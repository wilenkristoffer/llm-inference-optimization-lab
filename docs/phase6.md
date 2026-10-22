# Phase 6 - Quantization

Technical notes. What each GGUF quantization level costs in quality and buys
in memory and speed, on the 3B and 7B Qwen2.5, measured with a token-level
fidelity method rather than taken from llama.cpp's published tables. Plus a
root-cause investigation of why Q2_K destroys Qwen2.5-3B, and a "current
model" track on Qwen3.5-4B.

Date: 2026-10-22. Ollama 0.40.7, llama.cpp runner. Plan: `phase6-plan.md`.

---

## Goal

Every Ollama run in Phases 1-5 used Q4_K_M, because that is what the default
tags are. Four questions:

1. **Quality** - how far does each level drift from full precision, and where
   does the drift start changing answers?
2. **Memory** - what does each level actually occupy in VRAM?
3. **Speed** - does decode track weight size, as Phase 1's bandwidth model
   predicts?
4. **Practical** - for a 16 GB card, a bigger model at low precision or a
   smaller one at high precision?

---

## Groundwork that had to land first

Four measurement problems were found and fixed before any ladder ran. Each
would have distorted the comparison by more than the differences being
measured.

**VRAM baselines were not clean.** A model left resident by an earlier run was
counted as "desktop", so `peak - baseline` read **0 MB** for a loaded 1.5B.
`05_benchmark.py` now evicts every Ollama model (confirmed via `/api/ps`),
takes a *settled* baseline (median of 6 readings within 64 MB), and marks the
row `gpu_vram_baseline_clean = 1`. Only clean rows are footprints.

**Idle slots hold KV cache.** Each Ollama slot preallocates its own FP16 KV
cache, whatever the weight quantization. At the Backends panel's 8 slots the
1.5B Q4 footprint was 2,236 MB; at 1 slot, 1,382 MB - **854 MB of cache for
slots nothing was using**, ~40% of the footprint. The slot count is now read
from the llama-server command line (`-np`, stored as `server_num_parallel`)
and every footprint here is at 1 slot. The panel has a slots selector.

**Ollama's own VRAM figure runs low.** Recorded beside the board reading as
`server_vram_mb`. Across four models it was a constant ~270-320 MB below the
measured footprint - per-process runtime overhead it does not count.

**Sampling defaults differ by model.** Already pinned: temperature 0, seed,
`repeat_penalty 1.0` (Phase 5). Added for Qwen3.5, whose model file sets
`presence_penalty 1.5` - a penalty changes the winning token even at
temperature 0 - and which **thinks by default**: on a test question it spent
all 120 tokens on hidden reasoning and returned an empty answer. Every client
now sends `presence_penalty 0`, `frequency_penalty 0` and `think: false`.
Qwen2.5's defaults were already 0 and it has no thinking mode, so no earlier
result changes.

---

## Method

### Fidelity - the primary quality measure

Task accuracy on a few dozen questions cannot rank adjacent levels: with 60
questions one answer is 1.7 points. So the main measure compares **next-token
distributions** to full precision, token by token, on the same text
(`02_fidelity.py`, the idea behind llama.cpp's KL-divergence tool):

1. The reference (FP16, or Q8_0 for the 7B) writes greedy continuations of 30
   prompts - explanation, arithmetic, code, structured output, facts,
   instructions, creative, Swedish - 64 tokens each.
2. Every model is shown the prompt plus the first *i* reference tokens and
   asked for one token with its top-20 alternatives. Each is scored on the
   reference's text, never its own, so one early divergence cannot snowball.
3. Per position: top-1 agreement, the probability given to the reference
   token, and KL divergence from the reference over the top-20.

Three checks make it trustworthy:

- **Self-test.** The reference scored against itself must agree ~100% - it
  did, every run (99.5-100%). The script stops below 99%.
- **Alignment is verified**, not assumed: the server's `prompt_eval_count` must
  equal template + *i* tokens. Re-tokenizing a decoded prefix occasionally
  splits differently; those positions (0.1-2.8%) are excluded, not misscored.
- **Noise floor measured.** Two runs of the same 1.5B pair: top-1 93.6% vs
  93.8% (3 of 1,414 tokens), KL identical. Top-1 differences under ~0.5 point
  mean nothing; KL is the steadier figure.

The KL is an approximation - mass outside the top 20 is ignored - and an
underestimate for a model that ends its turn early (no logprobs come back for
those positions; they count as top-1 misses with no KL).

### Task accuracy - the sanity check

60 machine-checked questions (`03_task_accuracy.py`, `questions.json`): 20
arithmetic (number after `Answer:`), 20 factual (accepted forms, word
boundaries), 20 format-following (JSON parsed and checked, exact word counts,
lowercase-only, ...). No judgement calls; every failure in the validation run
was read by eye, which found one checker bug (it rejected "diamonds") before
any real run. Reported with a 95% Wilson interval.

**The validation run showed why it is secondary:** 1.5B Q4_K_M scored 56/60,
FP16 54/60. The quantized model "won" - noise, exactly as predicted.

### Memory and speed

`01_quant_ladder.py`, Ollama at 1 slot, `num_ctx` 4096. Every level runs once
per round, in a freshly shuffled order, for 3 rounds; drift lands on every
level alike. Per level per round: evict, settle, load, then 3 decode trials
(256 tokens) and 3 prefill trials (~1,250-token unique prompts, so the prefix
cache cannot help). Spread between rounds was 0.4-1.9%.

---

## Qwen2.5-3B - the full ladder

Fidelity against FP16; questions out of 60; footprint at 1 slot.

| level | file MB | VRAM MB | top-1 | KL | ppl | questions | decode tok/s | prefill tok/s |
|---|---|---|---|---|---|---|---|---|
| FP16 | 5,892 | 6,475 | 100% | 0 | 1.263 | 57 | 68.2 | 5,101 |
| Q8_0 | 3,133 | 3,676 | 98.7% | 0.0016 | 1.267 | 57 | 109.5 | 5,240 |
| Q6_K | 2,421 | 3,050 | 97.5% | 0.0056 | 1.268 | 57 | 133.1 | 4,648 |
| Q5_K_M | 2,122 | 2,749 | 95.8% | 0.0168 | 1.291 | 57 | 141.4 | 4,842 |
| **Q4_K_M** | 1,841 | 2,444 | 93.3% | 0.0579 | 1.330 | 57 | 152.1 | 4,906 |
| Q4_0 | 1,738 | 2,262 | 91.4% | 0.0952 | 1.409 | 56 | 138.0 | 5,392 |
| Q3_K_M | 1,517 | 2,038 | 75.0% | 0.677 | 1.971 | 51 | 166.7 | 4,699 |
| Q2_K | 1,216 | 1,826 | 0.7% | 4.83 | 89 | **0** | - | 4,462 |

### Quality: a steady decline, then a cliff

- **The two quality measures disagree, and that is the finding.** Questions
  cannot tell FP16 from Q4_K_M - 57/60 all the way down. Fidelity shows a
  steady decline underneath: from Q8_0 to Q4_K_M, KL roughly triples with
  every step. That is what
  quantization damage looks like before it changes answers.
- **The cliff is between Q4 and Q3.** KL jumps sevenfold from Q4_0 to Q3_K_M,
  and that is where answers start breaking - and they break unevenly:
  arithmetic stayed 20/20 while **format-following fell to 12/20**. Following
  exact instructions goes first.
- **Ollama's default, Q4_K_M, is a sound choice**: 93% agreement with FP16,
  no measurable change in answers, at 31% of the size.
- **Q4_0 is worse than Q4_K_M at almost the same size** (1.74 vs 1.84 GB, KL
  0.095 vs 0.058). The K-quant's mixed recipe is worth its 6%.
- **Q2_K collapses completely** - see its own section below.

### Memory: file size plus a near-constant overhead

Footprint is the weights plus ~520-630 MB (KV cache for one 4,096-token
context, compute buffers, ~280 MB runtime). Quantization shrinks only the
weights, so the overhead is a growing share of small levels: 26% of Q3_K_M's
footprint, 9% of FP16's.

### Speed: decode does not simply track size

Phase 1's model says decode streams every weight once per token, so halving
the file should double the rate. Effective bandwidth (file size x decode rate)
shows where it does not:

```
level     decode   GB/s   % of 624 peak
FP16       68.2     392      63%
Q8_0      109.5     334      54%
Q6_K      133.1     315      50%
Q5_K_M    141.4     293      47%
Q4_K_M    152.1     273      44%
Q4_0      138.0     233      37%
Q3_K_M    166.7     247      40%
```

- **Smaller levels use the bandwidth worse.** Going from FP16 to Q3_K_M cuts
  the bytes 3.9x but decode rises only 2.4x: unpacking the weights costs
  compute, and at this model size the per-token fixed cost (Phase 1) grows in
  relative terms. The saving is real but sublinear.
- **Q4_0 decodes slower than Q4_K_M despite being smaller** (138 vs 152 tok/s).
  The format's kernel matters, not only its bytes. Not investigated further.
- **Prefill is roughly flat across levels** (4.5-5.4k tok/s) for the 3B. It is
  compute-bound, so smaller weights do not help it. The 1.5B showed it
  sharply: Q4_K_M prefilled at ~5,950 tok/s against FP16's ~10,570 - quantized
  weights must be unpacked for every matrix multiply, and FP16 needs no
  unpacking. Faster decode, slower prefill.

### Behaviour changes, not only quality

The 1.5B Q4_K_M ended its "500-word" answer at 159 tokens; FP16 and every 3B
level wrote to the 256-token cap. A quantized model can change *how much* it
writes, not just how accurately.

---

## Qwen2.5-7B - Q4_K_M to Q8_0

FP16 (15.2 GB) does not fit in 16 GB, so the reference is Q8_0 - near-lossless
on the 3B (top-1 98.7%, KL 0.0016), measured above rather than assumed.

| level | file MB | VRAM MB | top-1 vs Q8_0 | KL | questions | decode tok/s | % peak |
|---|---|---|---|---|---|---|---|
| Q8_0 | 7,723 | 7,851 | 99.5% | 0 | 58 | 63.4 | 77% |
| Q6_K | 5,964 | 6,400 | 97.9% | 0.0038 | 59 | 79.4 | 74% |
| Q5_K_M | 5,193 | 5,706 | 96.9% | 0.0098 | 58 | 87.3 | 71% |
| **Q4_K_M** | 4,466 | 5,012 | 95.6% | 0.0322 | 58 | 96.0 | 67% |

- **The 7B degrades more gently than the 3B.** Q4_K_M keeps 95.6% agreement
  against the 3B's 93.3%, at roughly half the KL. Larger models carry more
  redundancy per weight. One caveat flatters the 7B slightly: it is measured
  against Q8_0, not FP16, so the small Q8_0-vs-FP16 loss is not in its figures.
- **The 7B uses the bandwidth better** - 67-77% of peak against the 3B's
  40-63%, continuing Phase 1's ladder. 7B Q8_0 at 77% is the highest fraction
  Ollama has reached in this project.
- **7B Q8_0's footprint is barely above its file** (7,851 vs 7,723 MB, +128 MB,
  where other levels add ~500). Likely explanation, not verified: llama.cpp
  keeps the input token-embedding table in system RAM, and the 7B - unlike the
  3B - has a separate output matrix, so its embeddings never need the GPU.
  At Q8_0 that table is ~580 MB.

---

## The practical question: bigger at Q4, or smaller at Q8?

At a similar footprint - **7B Q4_K_M (5.0 GB)** vs **3B Q8_0 (3.7 GB)** or
**3B FP16 (6.5 GB)** - this phase's measures **cannot rank them**, and saying
so is the honest answer:

- Fidelity is relative to each model's own reference; "95.6% of a 7B" and
  "98.7% of a 3B" are agreement with different things.
- The questions are too easy and too few: 58 vs 57 is inside the noise.

What can be said: the 7B's Q4_K_M loses less of the 7B than the 3B's Q4_K_M
loses of the 3B, and the 7B runs at 96 tok/s - fast enough for interactive
use. Ranking across sizes needs an absolute quality measure (a harder,
larger question set), which is a Phase 10 job.

---

## Why Q2_K destroys Qwen2.5-3B

3B Q2_K emits `433333...` or blank lines for every prompt: 0/60 questions,
13 of them aborted by Ollama's loop guard (HTTP 500, "token repeat limit
reached"), and at 26 positions it chose to end its turn mid-sentence where
FP16 had `" because"` or `" that"`. Q3_K_M is still coherent, so this is a
collapse, not a slope.

### Ruling things out

| hypothesis | test | result |
|---|---|---|
| ROCm/GPU kernel bug | same file on CPU (`num_gpu 0`) | same garbage - **ruled out** |
| corrupt download | Ollama verifies a checksum on every pull | **ruled out** |
| bad file on Ollama's hub | built our own Q2_K from FP16 with the `llama-quantize` Ollama ships | byte-identical size, same garbage - **ruled out** |
| embeddings crushed to 2 bits | read both GGUF headers | `token_embd` is **Q6_K** in both - **ruled out** |
| a broken recipe | compared the 1.5B and 3B Q2_K headers | **identical** recipe; the 1.5B is coherent |

So the same recipe is survivable for one model and fatal for another. The
3B itself is more sensitive somewhere.

### Finding the tensors

`04_q2k_collapse.py` builds Q2_K variants from FP16 with `--tensor-type`
overrides, each raising one set of tensors, and scores each against FP16:

| override on top of Q2_K | top-1 | KL | sample answer |
|---|---|---|---|
| none (reproduction) | 0.9% | 4.83 | `4333333...` |
| `ffn_down` -> Q4_K | 0.5% | 5.00 | `\n\n\n...` |
| `attn_q`, `attn_k` -> Q3_K | 0.8% | 4.85 | `43\n43的333...` |
| `attn_output` -> Q4_K | 1.5% | 4.83 | `\n\n43)4)4)...` |
| `attn_v` -> Q6_K | 0.8% | 4.83 | `4333333...` |
| `ffn_up` -> Q3_K | 0.5% | 4.86 | `4□4333...` |
| **`ffn_gate` -> Q3_K** | **44.3%** | 2.18 | `The capital of Sweden is the capital of` |
| **`ffn_gate` + `ffn_up` -> Q3_K** | **68.2%** | 0.90 | `Stockholm` |

**`ffn_gate` at 2 bits is the cause; `ffn_up` is a secondary contributor.**
The first hypothesis - outlier-heavy `ffn_down` layers, from published work
on activation outliers - was **wrong**: raising every `ffn_down` to 4 bits
changed nothing.

A plausible reading, not proven: in Qwen's feed-forward block the gate's
output passes through a SiLU and then multiplies `up`'s. Errors in the gate
switch neurons on and off wholesale; errors in `up` only rescale them. So the
gate alone can break the model, and `up` only matters once the gate is fixed.

The workaround is one flag: `--tensor-type ffn_gate=q3_k --tensor-type
ffn_up=q3_k` on top of Q2_K (1.29 GB against 1.22 GB), or simply Q3_K_M.

### Is it current?

Someone else hit this in March 2025 - llama.cpp Discussion #12378, "Gibberish
output of Qwen2.5-3B-Instruct with Q2_K quantization", on two other machines
and builds. It has no replies and was never diagnosed.

Qwen2.5 is two years old, so the deciding test was a current model. The same
Q2_K recipe on **Qwen3.5-4B** (Feb 2026, a hybrid attention/state-space
architecture):

| model, plain Q2_K | top-1 vs own full precision | KL | sample |
|---|---|---|---|
| Qwen2.5-3B | 0.9% | 4.83 | `4333333...` |
| Qwen2.5-1.5B | - | - | coherent (`The capital of Sweden is Stockholm.`) |
| Qwen3.5-4B | 66.0% | 0.88 | `The capital of Sweden is **Stockholm**.` |
| Qwen3.5-4B, `ffn_gate`+`ffn_up` -> Q3_K | 71.7% | 0.65 | `The capital of Sweden is Stockholm.` |

**Qwen3.5-4B survives** - degraded (29/60 questions, see its ladder below),
but coherent. The gate fix still helps it, so these layers are sensitive in both
generations; only Qwen2.5-3B is sensitive enough to collapse. The collapse is
a quirk of one older model, not a live problem, so it is documented here
rather than reported upstream.

Not tested: whether an importance matrix (`llama-imatrix`, not shipped with
Ollama) would also prevent it, and llama.cpp's standard perplexity numbers.

---

## Qwen3.5-4B - the current-model track

Qwen2.5 is two years old; Qwen3.5 (Feb 2026) is the newest family that fits
this card - Qwen3.6 and 3.8 start at 27B, and Qwen3.7 shipped no open weights.
The 4B is the direct successor to the 3B measured above, and a different
architecture: attention layers interleaved with state-space layers.

Ollama's library has only Q4_K_M, Q8_0 and BF16 for it, and mixing those with
levels made elsewhere would compare tools as well as precisions. So **every
level was built locally from the BF16 weights** with the same `llama-quantize`
(`05_local_ladder.py`), and the method is otherwise identical. Self-test:
100.0% over 1,558 positions.

| level | text MB | VRAM MB | top-1 | KL | ppl | questions | decode tok/s | prefill tok/s | % peak |
|---|---|---|---|---|---|---|---|---|---|
| BF16 | 8,265 | 9,659 | 100% | 0 | 1.271 | 58 | 53.3 | 3,490 | 69% |
| Q8_0 | 4,398 | 4,828 | 99.3% | 0.0033 | 1.272 | 58 | 77.0 | 3,487 | 53% |
| Q6_K | 3,399 | 3,975 | 98.2% | 0.0051 | 1.274 | 58 | 100.3 | 3,156 | 53% |
| Q5_K_M | 3,016 | 3,602 | 96.8% | 0.0112 | 1.280 | 58 | 106.7 | 3,268 | 50% |
| **Q4_K_M** | 2,655 | 3,261 | 94.7% | 0.0304 | 1.303 | 58 | 113.9 | 3,302 | 47% |
| Q4_0 | 2,492 | 2,983 | 92.5% | 0.0568 | 1.353 | 58 | 109.3 | 3,573 | 43% |
| Q3_K_M | 2,212 | 2,710 | 89.4% | 0.1075 | 1.442 | 58 | 123.8 | 3,189 | 43% |
| Q2_K | 1,869 | 2,478 | 66.0% | 0.885 | 2.853 | 29 | 132.1 | 3,135 | 39% |

The BF16 VRAM figure includes the 644 MB vision projector, which Ollama loads
with the model; the locally built levels carry no projector.

### The new model is much more robust to quantization

Same level, side by side (top-1 / KL against each model's own full precision):

```
level      Qwen2.5-3B          Qwen3.5-4B
Q8_0       98.7%  0.0016       99.3%  0.0033
Q6_K       97.5%  0.0056       98.2%  0.0051
Q5_K_M     95.8%  0.0168       96.8%  0.0112
Q4_K_M     93.3%  0.0579       94.7%  0.0304
Q4_0       91.4%  0.0952       92.5%  0.0568
Q3_K_M     75.0%  0.677        89.4%  0.1075
Q2_K        0.7%  4.83         66.0%  0.885
```

- **At Q4_K_M, Qwen3.5 loses about half as much** (KL 0.030 vs 0.058).
- **The cliff moved down a whole level.** Qwen2.5's KL jumped sevenfold from
  Q4_0 to Q3_K_M; Qwen3.5's rises gently through Q3_K_M (89.4%, the same 58/60
  as BF16) and only jumps eightfold at Q2_K.
- **Q2_K is usable-but-damaged, not broken**: 29/60 questions - arithmetic
  8/20, facts 14/20, format 7/20 - against Qwen2.5-3B's 0/60.
- **At the top it is the reverse.** Qwen3.5's Q8_0 KL is twice Qwen2.5's
  (0.0033 vs 0.0016). Tiny either way, but the new model is not uniformly more
  robust - it has a small loss at 8 bits that shrinks less with each step down.

Caveat: the 4B has about 40% more parameters than the 3B (4.3B vs 3.1B), and the earlier
finding that larger models carry more redundancy (the 7B above) applies. Some
of the robustness is size, not generation; this ladder cannot separate the two.

### Speed: the same pattern, and a confirmed oddity

- **Decode gains from quantization match Qwen2.5's.** BF16 to Q4_K_M is 3.1x
  fewer bytes for 2.1x the decode rate (Qwen2.5-3B: 3.2x for 2.2x).
- **Full precision uses the bandwidth best** - 69% of peak at BF16, falling to
  39% at Q2_K - as on Qwen2.5.
- **Q4_0 decodes slower than the larger Q4_K_M again** (109.3 vs 113.9 tok/s).
  Seen on both families, so it is the format's kernel, not a model quirk.
- **Q2_K decodes faster than Q3_K_M** here (132 vs 124); on Qwen2.5-3B Q3_K_M
  was the fastest level. Q3_K_M also had the largest round-to-round spread on
  this ladder (3.8%).
- **Prefill is ~35% slower than Qwen2.5-3B's** at every level (3.1-3.6k vs
  4.5-5.4k tok/s) and, again, flat across levels - compute-bound, and the
  state-space layers are not free.

### What it means

For this card, Qwen3.5-4B at **Q4_K_M** (3.3 GB) keeps 94.7% agreement with
its full precision and the same 58/60 as BF16. **Q3_K_M** (2.7 GB) is also
viable here, which it was not for Qwen2.5-3B: the "never go below Q4" rule of
thumb is model-dependent, and newer models push the floor down.

---

## Bugs fixed along the way

- **Mid-stream aborts were recorded as measurements.** When Ollama's loop
  guard aborted a streamed generation, the stream ended with an error chunk
  instead of the final one, and `run_once` silently stored "0 tokens at
  -1.8 tok/s". It now reports the abort and stores no rate; 9 rows were
  corrected to NULL with a note.
- **A decode rate of 0 for nothing.** `run_once` returned 0 tok/s when it had
  fewer than two tokens - the lab's rule is NULL, never an invented 0.
- **`--history` crashed** on run groups with no context length (since Phase 5).
- **A multimodal model's size included its vision projector.** The ladder took
  file size from `/api/tags`, which for `qwen3.5:4b-bf16` adds a 644 MB vision
  file that decode never reads - BF16's bandwidth read 74% instead of 69%. It
  now measures the text-weights file itself; the Qwen2.5 sizes were unaffected.
- **Two checker/script bugs** caught by reading every failure before trusting
  a score: the plural "diamonds" rejected, and an `Answer:` line with no number
  falling back to the wrong number.

---

## Limitations

- One family per size, one card, Ollama's llama.cpp runner only. vLLM's
  quantization formats (AWQ, GPTQ, FP8) are different code paths - Phase 9.
- KV cache quantization (`OLLAMA_KV_CACHE_TYPE`) not tested; every footprint
  carries a full-precision KV cache.
- Fidelity is top-20 KL over 30 prompts x 64 tokens; good for ranking levels
  of one model, not for comparing different models.
- The question set is easy: FP16 and Q4_K_M are indistinguishable on it.
- No importance-matrix quants; Ollama's library tags are what people run.

---

## Open items

- An absolute quality measure that can rank across model sizes (Phase 10).
- Why Q4_0 decodes slower than the larger Q4_K_M.
- Whether llama.cpp keeps token embeddings in system RAM (the 7B Q8_0
  footprint).
- `--context-shift --keep 4` on Ollama's llama-server is a candidate mechanism
  for Phase 5's "oldest-first, half the window" truncation - a hypothesis to
  test.

# Phase 9 plan - Inference optimization

Status: **planned 2026-10-23; Stage 0 done the same day.** The rest is a
plan, not results.
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

### Stage 0 result (2026-10-23): everything runs

One prompt (palindrome function plus three tests, chat template, greedy, 256
tokens), 3-5 runs per configuration. The scripts were throwaway probes; Stage 1
rebuilds them properly.

| backend | config | decode | acceptance | output vs baseline | distinct texts |
|---|---|---|---|---|---|
| llama.cpp | 14b, f16 KV, flash attn | 52.3 tok/s | | | 1 of 5 |
| llama.cpp | 14b + 0.5b draft, K=3 | 70.3 tok/s (**1.34x**) | 0.79-0.81 | identical | 1 of 5 |
| llama.cpp | 14b, q8_0 KV | 50.4 tok/s (-4%) | | run 0 identical | **3 of 3** |
| llama.cpp | 14b, q4_0 KV | 49.5 tok/s (-5%) | | differs at char 163 | 3 of 3 |
| vLLM | 1.5B FP16 | 150 tok/s | | | 1 of 5 |
| vLLM | 1.5B + ngram, K=3 | 135 tok/s (0.90x) | 0.47 | identical | 1 of 5 |
| vLLM | 3B FP16 | 75.1 tok/s | | | 1 of 5 |
| vLLM | 3B + 0.5B draft, K=3 | 113.1 tok/s (**1.51x**) | 0.83 | identical | 1 of 5 |

vLLM figures are client wall time including a ~30-token prefill, so slightly
low; llama.cpp's are its own decode timings.

What it shows:
1. **Running llama.cpp directly needs two things Ollama normally provides.**
   `GGML_BACKEND_PATH=<ollama>\lib\ollama\rocm_v7_1\ggml-hip.dll` (plus that
   directory on PATH): without it llama-server finds no GPU at all. And
   `--spec-type draft-simple`: in this build `-md` only *loads* the draft, and
   the spec type defaults to `none`. The first run had a draft loaded and
   unused; the giveaway was 254 graph reuses for 256 tokens, one target pass
   per token.
2. **The direct server beats Ollama on the same blob: 52.3 vs 44 tok/s for the
   14b.** Flash attention forced on and no Ollama runner in between. The
   Stage 1 baseline is therefore this server without a draft, never Phase 1's
   44, or speculation would get credit for the 19% that is just the server.
3. **Speculation matched the non-speculative text** on both backends in these
   runs (but see point 4: identical text is not guaranteed even without it). The draft's acceptance counts vary between runs
   (180/222 vs 178/228) while the text doesn't, which is the mechanism
   working as designed.
4. ~~Greedy decoding is deterministic here.~~ **Corrected the same day: it
   isn't, it just looked it on the 14b.** The 14b gave 1 distinct text of 5,
   but the plain 3b on the same server gave 2 of 4 (diverging at character
   1010), with no draft involved. Greedy on this stack flips near-tie tokens
   between identical requests; the 14b had no near-tie in those 256 tokens.
   A quantized KV cache makes it more frequent (q8_0: 3 of 3 distinct).
   Measured on the 3b, first divergence from plain run 0:

   | config | decode | divergence per run |
   |---|---|---|
   | plain | 148.5 tok/s | same, 1010, same, same |
   | draft K=3 | 131.1 tok/s (0.88x) | same, same, 1076, 1010 |
   | draft K=8 | 116.7 tok/s (0.79x) | 513, 127, 127, 502 |

   K=3 diverges where the plain run diverges on its own; K=8 diverges
   earlier, since a longer verify batch rounds differently from a single
   token. So "byte-identical" is not the test for lossless: Stage 1 compares
   the distribution of divergence points against plain-vs-plain, as planned.
   Also found: llama-server ignores per-request `speculative.n_max` (n_max 0
   still drafted), so K is fixed per server start.
5. **The llama.cpp speedup is well below the formula.** At a = 0.8, K = 3,
   c = 0.11 it predicts about 2.2x; measured 1.34x. Either the draft costs much
   more than its share of bytes suggests (small models use bandwidth badly:
   Phase 1's ladder), or the verify pass of 4 tokens costs more than one
   decode step. Stage 1 separates the two by timing the draft alone and the
   verify step.
6. **vLLM draft_model beats the 3B prediction** (1.51x against ~1.1x): that
   estimate used llama.cpp's Q4 sizes. In FP16 the 0.5B is 1 GB against the
   3B's 6 GB. vLLM falls back to its V1 model runner for `draft_model`; no
   `--enforce-eager` was needed.
7. **ngram is a loss on code written from scratch** (0.90x at 47% acceptance),
   as predicted: there is nothing in the prompt to copy. Extraction is its
   real test.
8. **Quantized KV is slower at short context** (-4/-5%): dequantizing costs
   time, and 300 tokens of KV is too little to save bandwidth. Stage 2 tests
   whether that reverses at depth.

Side note: WSL dropped its service connection once (`Wsl/Service/0x8007274c`)
during a 14b run on Windows and recovered by itself; the download running in
the distro survived.

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

**Side experiment: Jacobi decoding** (added 2026-10-23 at the user's request).
Can a plain model produce several tokens per pass with no draft at all? Guess
the next n tokens, run the model once over all of them, replace each guess
with the model's prediction for that position, repeat. The first position is
always right after a pass, and a later one becomes right once the guesses
before it have settled. It converges to exactly the greedy output, so it is
lossless, and the speedup is the number of tokens that settle per pass.
Implemented by hand in PyTorch (HF transformers, KV cache cropped to the
accepted prefix) on the 1.5B in WSL, window n = 2 / 4 / 8 / 16, same
workloads. Compared with HF greedy in the same framework, not with vLLM.

Prediction: 1.05-1.3 tokens per pass for a model never trained for it, and
little or no wall-clock gain, because a pass over n+1 tokens costs more than
one over 1 in eager PyTorch. Most gain on extraction, where copied spans
settle together. This is the "why is decoding one at a time" demonstration:
the model can't predict token 3 well until token 2 is real.

### Stage 1 result, llama.cpp (2026-10-23)

`phase9-optimization/01_spec_llamacpp.py`, run group
`p9-spec-llamacpp-20261023-1233`, 639 rows, 27 min. Draft qwen2.5:0.5b, all
Q4_K_M, llama-server a37cc4610 run directly. Every config ran 9 prompts x 2;
plain ran first and last per target (drift +0.9% / +1.8% / -0.7%). The draft
alone: 260 tok/s, 3.85 ms per token.

Best speedup per workload (K in brackets):

| target | plain | code | extraction | prose |
|---|---|---|---|---|
| 14b | 52.1 tok/s | **1.58x** (8) | **1.99x** (8), 102.6 tok/s | 1.09x (1) |
| 7b | 96.5 tok/s | 1.05x (8) | 1.49x (8) | a loss at every K |
| 3b | 154.8 tok/s | a loss at every K | 1.10x (8) | a loss at every K |

One pass over n new tokens (cached 300-token prefix), ms:

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14b | 19.8 | 22.4 | 25.4 | 29.4 | 34.6 | 38.6 | 44.8 | 48.0 | **36.0** | 36.2 | 36.3 |
| 7b | 10.9 | 12.4 | 13.9 | 16.1 | 18.4 | 20.5 | 24.0 | 24.7 | **18.5** | 18.5 | 18.7 |
| 3b | 6.9 | 8.3 | 8.8 | 9.9 | 10.8 | 11.8 | 13.0 | 14.0 | **13.2** | 13.4 | 13.5 |

Findings:
1. **Where the Stage 0 shortfall came from: checking is not free here.** The
   formula assumes checking K+1 tokens costs one decode step. In llama.cpp
   the cost rises about linearly up to 8 tokens (14b: 4 tokens = 1.5 steps,
   8 = 2.4), then drops at 9, where it switches from a per-token vector kernel
   to a matrix kernel. With the measured check cost, each pass adds up to
   check + draft + 3-7 ms of other overhead. That budget explains the measured
   speed; e.g. 14b K=4 extraction: 34.6 + 15.0 + 5.1 = 54.7 ms per pass for
   4.55 tokens = 83 tok/s, as measured.
2. **K=8 wins, as the curve predicted.** K=8 checks 9 tokens, the first size on
   the cheap kernel, so on the 14b it checks 9 for less than K=6 checks 7.
   It is the best K for code and extraction on every target. This is the
   mirror image of the vLLM prediction (best K at most 4, skinny GEMM covers
   1-5): **the best K is set by the backend's kernel switch points, not only
   by acceptance.**
3. **Acceptance comes in runs, so the textbook formula misjudges long K.** At K=8
   the 14b accepts 56% of drafted code tokens, which as an independent
   per-token rate predicts 2.3 tokens per pass; measured 5.3. A draft is
   either right for a long stretch (boilerplate, copied text) or wrong early.
   The formula column under-predicts code at K=8 (0.87x against 1.58x) for
   this reason. Tokens per pass, not acceptance, is the number to use.
4. **The draft must be cheap relative to the target.** c = 0.20 on the 14b,
   0.37 on the 7b, 0.60 on the 3b. The 0.5b Q4 runs at 260 tok/s, far below
   its size share (Phase 1's ladder: small models use bandwidth badly). On
   the 3b, drafting 8 tokens takes 29.5 ms per pass against 6.5 ms for a
   plain step, so only near-perfect acceptance (extraction, 0.95) breaks even.
5. **Prose is a loss above K=1-2 on every target**: acceptance 0.18-0.63,
   falling with K.
6. **Speculation changed the greedy text in this backend.** All 36 plain runs
   were repeatable: 1 distinct text per prompt, 4 runs each, all 3 targets.
   Speculative runs diverged on most code and prose runs (median character
   200-440 of ~1,170) and almost never on extraction. The divergences are
   near-tie flips: `-> None:` vs `-> None`, "The air is crisp" vs "The sky,
   a canvas". The cause fits finding 1: the check pass computes logits on a
   different kernel from the single-token step, and different rounding can
   flip a near tie. The text stays fluent, but it is not the plain model's
   text, so "lossless" holds for the distribution, not the bytes. This also
   contradicts Stage 0 finding 4 (plain greedy not repeatable). The Stage 0
   probe ran differently (`/completion` with a raw prompt, no
   `--flash-attn on`, no seed), and which of those caused it wasn't
   isolated. Under this harness plain greedy is repeatable.
7. **VRAM**: the draft adds 502 MB (14b, 7b) and 518 MB (3b), per-process
   counter. That's within the 0.6 GB prediction.

Predictions scored: the 14b at K=4 with a = 0.7 was predicted at 1.9x;
code measured 1.43x and extraction 1.61x. The 7b at 1.5x matched only on
extraction. The 3b ~1.1x matched extraction at K=8; code and prose were
losses. The acceptance order was extraction > code > prose, not
code > extraction as predicted.

### Stage 1 result, vLLM (2026-10-23)

`phase9-optimization/02_spec_vllm.py`, run group `p9-spec-vllm-20261023-1300`,
852 rows. vLLM 0.29, FP16, target Qwen2.5-3B-Instruct, draft 0.5B; ngram with
prompt lookup 2-4 and K=4. Capture sizes 1-16 plus larger, so no step was
padded. Plain 3B first and last: drift -0.1%.

Single stream, speedup over the plain server:

| config | code | extraction | prose |
|---|---|---|---|
| 3B plain | 75.6 tok/s | 75.9 tok/s | 75.5 tok/s |
| draft K=1 | 1.14x | 1.00x | 0.97x |
| draft K=2 | 1.38x | 1.24x | 1.05x |
| draft K=3 | 1.56x | 1.53x | **1.09x** |
| draft K=4 | **1.72x**, 129.8 tok/s | **1.65x** | 1.03x |
| draft K=5 | 0.55x | 0.65x | 0.31x |
| draft K=6 | 0.60x | 0.72x | 0.32x |
| draft K=8 | 0.66x | 0.89x | 0.30x |
| 3B ngram | 0.90x | 1.49x | 0.80x |
| 1.5B ngram (vs 1.5B plain, 151.5 tok/s) | 0.90x | 1.51x | 0.81x |

Aggregate tok/s with c requests in flight (all nine prompts mixed), and the
ratio to the plain server at the same c:

| config | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| 3B plain | 73 | 248 | 176 | 331 |
| draft K=3 | 116 (1.59x) | 116 (0.47x) | 288 (1.64x) | 630 (1.90x) |
| draft K=4 | 125 (1.71x) | 122 (0.49x) | 315 (1.79x) | 605 (1.83x) |
| draft K=5 | 44 (0.60x) | 137 (0.55x) | 331 (1.88x) | 593 (1.79x) |
| 3B ngram | 72 (0.99x) | 65 (0.26x) | 155 (0.88x) | 320 (0.96x) |
| 1.5B plain | 149 | 559 | 1053 | 1973 |
| 1.5B ngram | 140 (0.94x) | 273 (0.49x) | 734 (0.70x) | 1239 (0.63x) |

Findings:
1. **The K=5 cliff is real** (prediction 1). K=4 is the best K, and K=5 runs
   at a third of K=4's speed, with more tokens per pass (4.9 vs 4.3 on code).
   A check of 5+1 = 6 tokens leaves the skinny GEMM. llama.cpp's best K is 8
   for the mirror-image reason: the kernel switch points set the best K.
   The switch is the cause, but the cliff's size is inflated by finding 4:
   the generic path K=5-8 lands on includes the mis-tuned gate_up GEMM. The
   Stage 4 tuned-GEMM re-run includes single-stream K=5/6/8 to measure the
   cliff on a properly tuned kernel.
2. **vLLM gains more than llama.cpp on the same-sized target.** 3B: vLLM 1.72x
   code, against llama.cpp's loss on code at every K. In FP16 the 0.5B draft
   is 1 GB against 6 GB. Its share of each pass is smaller than the Q4 0.5b's
   share against a Q4 3b, and the skinny GEMM checks 2-5 tokens for little
   more than 1.
3. **ngram only pays on extraction**: 1.49-1.51x, the low end of the 1.5-2.5x
   prediction. It loses on code and prose (0.80-0.90x, predicted ~1.0x). The
   draft model beats it even on extraction (1.65x).
4. **The concurrency prediction failed for the 3B, and the reason is a
   mis-tuned GEMM, not speculation.** Plain 3B aggregate *drops* from 248
   tok/s at c=4 to 176 at c=16. Root cause, measured in PyTorch with the
   3B's layer shapes (`scratchpad gemm_shapes.py`): the merged gate_up
   projection, 2048 -> 22016, takes 960-1,500 us per call at any row count on
   the generic path (rocBLAS; hipBLASLt 500-740 us). That's 60-180 GB/s on a
   624 GB/s card, and 4-5x slower than two 11008-wide halves. The 1.5B's
   1536 -> 17920 is fine. With `PYTORCH_TUNABLEOP_ENABLED=1` the same GEMM
   takes ~170 us (~530 GB/s), so a good kernel exists and the library's
   default choice is the fault. Above 5 tokens per step every plain 3B step
   is therefore dominated by a GEMM whose cost hardly depends on the row
   count, and speculation's extra check tokens ride along for free. That's
   why speculation "wins" 1.6-1.9x at c=16-32. At c=4 the opposite happens:
   plain (4 tokens) stays on the skinny GEMM while every speculative config
   (4 x (K+1) >= 8 tokens) falls onto the slow path, hence 0.25-0.65x. The
   1.5B, whose generic path scales, shows the predicted shape: ngram loses
   more as c grows (0.94x -> 0.63x). **So the 3B concurrency numbers measure
   the broken kernel and must be re-run with tuned GEMMs before anything is
   concluded from them.** That's added to Stage 4.
5. **In vLLM speculation does not change the text more than plain does.**
   Plain-vs-plain diverged on 1 of 3 code and prose runs (vLLM's greedy isn't
   repeatable, unlike llama-server under the same harness). Speculative runs
   diverged on 0-2 of 6, the same rate.
6. Startup 52-56 s per config once the compile cache was warm (90-94 s for
   the first two).

Predictions scored: 1 (cliff at K>=5) confirmed. 2 (ngram 1.5-2.5x on
extraction, ~1.0x on prose) at the low end, and prose was a loss. 3
(speculation becomes a loss with concurrency) confirmed on the 1.5B; on the
3B it's confounded by finding 4. 4 (acceptance order) was extraction > code >
prose, as in llama.cpp.

### Stage 1 side experiment result, Jacobi decoding (2026-10-23)

`phase9-optimization/03_jacobi.py`, run group `p9-jacobi-20261023-1348`, 90
rows. HF transformers 5.17 eager, Qwen2.5-1.5B-Instruct FP16 in WSL, 9 prompts
x 2 reps, every window interleaved per prompt. w = 0 is plain greedy in the
same loop.

| window | code | extraction | prose | guesses right |
|---|---|---|---|---|
| 0 (plain) | 36.9 tok/s | 36.7 tok/s | 37.1 tok/s | |
| 2 | 1.00x, 1.05 tok/pass | 1.02x, 1.11 | 0.98x, 1.06 | 2.5-5.7% |
| 4 | 1.00x, 1.06 | **1.07x, 1.13** | 0.99x, 1.06 | 1.5-3.2% |
| 8 | 0.99x, 1.06 | 1.06x, 1.14 | 0.97x, 1.06 | 0.7-1.8% |
| 16 | 1.00x, 1.06 | 1.06x, 1.14 | 0.98x, 1.05 | 0.3-0.9% |

Findings:
1. **The model can't guess ahead of itself** (prediction: 1.05-1.3 tokens per
   pass; measured 1.05-1.14). Fewer than 6% of guesses turn out right, and
   a wider window adds nothing past w = 4. Each extra correct token comes
   from a short copied span settling, which is why extraction leads. This is
   the answer to "why one token at a time": a model trained only on next
   tokens doesn't know token 3 until token 2 is real. A draft model is
   another model trained on the same task, which is why it gets 0.8-0.95.
2. **The wall-clock gain is ~0** (best 1.07x), as predicted. Here a pass
   costs 27-30 ms whatever its width. That's HF eager overhead: the same
   1.5B decodes at 151 tok/s in vLLM against 37 here, so the GPU work is
   hidden. In an efficient backend the pass cost grows with width (the
   llama.cpp curve above), and Jacobi would be a loss. 1.14 tokens per pass
   is the ceiling it could reach anywhere.
3. Prose texts diverged on 2-4 of 6 runs against 0 of 3 for plain: the wider
   pass rounds differently, as in llama.cpp (finding 6 there).

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
  *Update after Stage 1:* the 3B's cliff is much bigger, not smaller. Plain
  3B aggregate falls from 248 tok/s at c=4 to 176 at c=16, because its
  gate_up GEMM (2048 -> 22016) gets a bad default kernel (Stage 1 vLLM
  finding 4).
- **Tuned GEMMs (added after Stage 1).** Re-run the 3B plain and
  speculative concurrency sweep with PyTorch TunableOp
  (`PYTORCH_TUNABLEOP_ENABLED=1`, tuned offline first so CUDA graph capture
  only reads the table). In isolation it took the 3B gate_up from
  ~1,000-1,500 us to ~170 us. Prediction: plain 3B at c=32 goes from 331 to
  ~1,000+ tok/s, and speculation's c=16-32 "win" shrinks to a loss, as on
  the 1.5B. Also check whether it lifts the 1.5B past 1,973 tok/s at c=32
  (its gate_up went from ~110-150 us to ~60 us in the same test).

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

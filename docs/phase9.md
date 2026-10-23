# Phase 9 - Inference optimization

Technical notes. What the settings that move decode speed, latency and memory
buy and cost on this machine (Windows 11, RX 7800 XT gfx1101 16 GB, Ryzen 5
5600X, 32 GB RAM): speculative decoding, context length and KV cache
quantization, prefix caching, concurrency and batching. Along the way it found
that vLLM's default GEMM kernels on this card are badly chosen, and fixing
that turned out to be the biggest win of the phase.

Date: 2026-10-23. Ollama 0.40.9, llama.cpp build a37cc4610 (Ollama's bundled
`llama-server.exe`, ROCm 7.1 backend), vLLM 0.29.0 in WSL2 (torch
2.12.0+git6bbd260, HIP 7.2.53211), HF transformers 5.17. Models: the Qwen2.5
family, which shares one tokenizer. Plan, with every stage's full tables:
`phase9-plan.md`. Code: `phase9-optimization/`.

---

## Goal

Change the settings Phases 1-8 measured the baseline for, and measure what
each change gains, what it costs, and what it means for the user's agent:

1. **Speculative decoding** - a small draft model guesses K tokens and the
   target checks them in one pass. Not in the brief, but the most interesting
   optimization available here. Plus a side experiment: Jacobi decoding, where
   the model guesses ahead of itself with no draft.
2. **Context length and the KV cache** - how decode slows as the context
   fills, and what quantizing the KV cache does to speed, VRAM and quality.
3. **Prefix caching** - cold vs warm TTFT on agent-shaped prompts, on both
   backends in one session.
4. **Concurrency and batching** - Ollama at high `NUM_PARALLEL`, vLLM's
   scheduler limits, and inter-token latency while a long prompt arrives.

---

## Results against the predictions

| metric | prediction | measured | verdict |
|---|---|---|---|
| llama.cpp 14b + 0.5b draft, K=4 | 1.9x | code 1.43x, extraction 1.61x; best 1.99x (extraction, K=8) | too high at K=4 |
| best K on vLLM | at most 4 (skinny GEMM covers 1-5 tokens) | 4 with default GEMMs; with tuned GEMMs K=5-8 work, extraction best at K=8 | right for the wrong reason |
| ngram (prompt lookup) | 1.5-2.5x extraction, ~1.0x prose | 1.49-1.51x extraction, 0.80x prose | low end; prose wrong |
| speculation under concurrency | a loss by c=16-32 | 3B (tuned): 0.67x at c=16, 0.52x at c=32 | right |
| acceptance order | code > extraction > prose | extraction > code > prose | wrong |
| draft VRAM (llama.cpp) | ~0.6 GB | 502-518 MB | right |
| Jacobi decoding | 1.05-1.3 tokens per pass, ~no wall-clock gain | 1.05-1.14, best 1.07x | right |
| 1.5b decode at 31.7k context | -47% | -26% | wrong (too pessimistic) |
| 14b decode at 16k context | -26% | -41% | wrong (too optimistic) |
| q8_0 KV speed | recovers about half the slowdown | 1.5b: slower at every depth; 14b: +18% at 16k | wrong on 1.5b |
| KV quality | q8_0 barely measurable, q4_0 visible | q8_0 KL 0.001-0.006; q4_0 **breaks** the 1.5b and 7b | right; q4_0 understated |
| 14b context that fits | q8_0 fits 32k, f16 stops at ~18k | q8_0 fits 32k (12.1 GB); f16 ~22k (extrapolated) | right / close |
| cold TTFT, 2.2k-token prompt | Ollama ~260 ms, vLLM ~180 ms | 288 ms, 275 ms | Ollama right, vLLM wrong |
| warm TTFT | Ollama ~25 ms, vLLM ~15 ms | 42 ms, 48 ms | wrong |
| `NUM_PARALLEL` > 1 for one conversation | loses cache hits | no change | wrong |
| Ollama aggregate at c=32 | ~1,000 tok/s | 1,125 tok/s (np32) | right |
| tuned GEMMs, vLLM 1.5B at c=32 | ~2,250 tok/s | 3,056 tok/s | too low |
| tuned GEMMs, vLLM 3B at c=32 | ~1,000 tok/s | 1,897 tok/s (from 331) | too low |
| `--max-num-seqs 8` at c=32 | c=8 throughput, ~2.5x latency | 833 tok/s, 2.2x | right |
| Ollama while a 4k prompt arrives (first plan) | streams stall for the whole prefill | worst gap 210 ms; it interleaves | wrong |
| worst gap by chunk size (revised) | 60 / 240 / 470 ms at 512 / 2,048 / 8,192 | 69 / 234 / 381 ms | right |

---

## 01 - Can each piece run here?

Everything runs on gfx1101. Two things are needed to run Ollama's bundled
llama-server directly, because Ollama normally provides them:

- `GGML_BACKEND_PATH=<ollama>\lib\ollama\rocm_v7_1\ggml-hip.dll`, with that
  directory on PATH. Without it llama-server finds no GPU.
- `--spec-type draft-simple`. In this build `-md` only *loads* the draft; the
  spec type defaults to `none`. The giveaway in the first run was 254 graph
  reuses for 256 tokens, one target pass per token.

Also: llama-server ignores a per-request `speculative.n_max`, so K is fixed
per server start. vLLM's `draft_model` and `ngram` both run (`draft_model`
falls back to the V1 model runner; no `--enforce-eager` was needed).

**The direct server is faster than Ollama on the same blob:** 52.3 against
44 tok/s on the 14b, with flash attention forced on and no Ollama runner in
between. Every speculation baseline in this phase is therefore the same
server without a draft. Otherwise speculation would get credit for the 19%
that is just the server.

**Greedy decoding is not a reliable test of "lossless".** In the Stage 0
probe the plain 3b gave 2 distinct texts of 4 identical temperature-0
requests, with no draft. Under the Stage 1 harness (`/v1/chat/completions`,
`--flash-attn on`, seed 42) plain llama-server was repeatable (36 of 36 runs),
and vLLM was not (1 of 3 runs diverged). Which difference made the probe
non-repeatable wasn't isolated. Speculation was therefore judged by where
outputs diverge compared with plain-vs-plain, never by identical bytes.

---

## 02 - Speculative decoding on llama.cpp

`01_spec_llamacpp.py`, run group `p9-spec-llamacpp-20261023-1233`, 639 rows.
Draft qwen2.5:0.5b, all Q4_K_M. Nine prompts (three each of code,
extraction and prose, `workloads.py`) x 2, K = 1-8. Plain ran first and last
per target; drift under 2%.

Best speedup per workload (K in brackets):

| target | plain | code | extraction | prose |
|---|---|---|---|---|
| 14b | 52.1 tok/s | **1.58x** (8) | **1.99x** (8), 102.6 tok/s | 1.09x (1) |
| 7b | 96.5 tok/s | 1.05x (8) | 1.49x (8) | a loss at every K |
| 3b | 154.8 tok/s | a loss at every K | 1.10x (8) | a loss at every K |

**Why it falls short of the formula: checking isn't free on this backend.**
The textbook speedup assumes checking K+1 tokens costs one decode step. Timed
directly (one pass over n new tokens on a cached 300-token prefix, ms):

| n | 1 | 2 | 4 | 6 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| 14b | 19.8 | 22.4 | 29.4 | 38.6 | 48.0 | **36.0** | 36.2 | 36.3 |
| 7b | 10.9 | 12.4 | 16.1 | 20.5 | 24.7 | **18.5** | 18.5 | 18.7 |
| 3b | 6.9 | 8.3 | 9.9 | 11.8 | 14.0 | **13.2** | 13.4 | 13.5 |

The cost rises about linearly to 8 tokens on a per-token vector kernel, then
drops at 9, where llama.cpp switches to a matrix kernel. With that check cost,
plus the draft and 3-7 ms of other overhead per pass, the measured speeds come
out: 14b K=4 extraction is 34.6 + 15.0 + 5.1 = 54.7 ms per pass for 4.55
tokens, 83 tok/s, as measured.

Findings:
1. **K=8 wins because it checks 9 tokens,** the first size on the cheap
   kernel. On the 14b, checking 9 costs less than checking 7. **The best K is
   set by the backend's kernel switch points, not only by acceptance.**
2. **Acceptance comes in runs.** At K=8 the 14b accepts 56% of drafted code
   tokens. As an independent per-token rate that predicts 2.3 tokens per
   pass; measured 5.3. A draft is either right for a long stretch (boilerplate,
   copied fields) or wrong early. Use tokens per pass, not acceptance rate, to
   predict speedup.
3. **The draft has to be cheap relative to the target.** Draft cost per target
   step: 0.20 on the 14b, 0.37 on the 7b, 0.60 on the 3b. The 0.5b runs at
   only 260 tok/s, well below what its size suggests (small models use
   bandwidth badly, Phase 1). On the 3b, drafting 8 tokens takes 29.5 ms
   against 6.5 ms for a plain step.
4. **Prose loses above K=1-2 on every target** (acceptance 0.18-0.63, falling
   with K).
5. **Speculation changed the greedy text here.** Plain runs were repeatable;
   speculative code and prose runs diverged at median character 200-440 of
   ~1,170, extraction almost never. The divergences are near-tie flips
   (`-> None:` vs `-> None`). The check pass computes logits on a different
   kernel from the single-token step, and the rounding difference flips near
   ties. The text stays fluent, but it is not the plain model's text: lossless
   holds for the distribution, not the bytes.
6. VRAM: the draft adds 502-518 MB.

---

## 03 - Speculative decoding on vLLM

`02_spec_vllm.py`. Target Qwen2.5-3B-Instruct FP16, draft 0.5B FP16 (a 7B in
FP16 doesn't fit). Run twice: with the default GEMM kernels
(`p9-spec-vllm-20261023-1300`, 852 rows) and with tuned ones (`--tuned`,
`p9-spec-vllm-tuned-20261023-1610`, 426 rows; see section 07).

Single stream, speedup over the plain 3B (~75 tok/s) on the same server:

| config | code (default -> tuned) | extraction | prose |
|---|---|---|---|
| draft K=3 | 1.56x -> 1.61x | 1.53x -> 1.55x | 1.09x -> 1.11x |
| draft K=4 | 1.72x -> **1.77x** | 1.65x -> 1.73x | 1.03x -> 1.05x |
| draft K=5 | **0.55x** -> 1.67x | 0.65x -> 1.78x | 0.31x -> 0.90x |
| draft K=6 | 0.60x -> 1.67x | 0.72x -> 1.94x | 0.32x -> 0.88x |
| draft K=8 | 0.66x -> 1.53x | 0.89x -> **2.02x** | 0.30x -> 0.69x |
| ngram K=4 (default GEMMs) | 0.90x | 1.49x | 0.80x |

Aggregate tok/s with c requests in flight (all nine prompts mixed):

| config | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| 3B plain, default GEMMs | 73 | 248 | 176 | 331 |
| 3B plain, tuned | 75 | 266 | **1,013** | **1,897** |
| + draft K=3, tuned | 120 | 299 | 680 | 982 |
| + draft K=4, tuned | 128 | 313 | 572 | 648 |
| + draft K=5, tuned | 125 | 275 | 493 | 696 |
| 1.5B plain, default GEMMs | 149 | 559 | 1,053 | 1,973 |
| 1.5B ngram, default GEMMs | 140 | 273 | 734 | 1,239 |

Findings:
1. **With default kernels there is a cliff at K=5,** as predicted: a check of
   K+1 = 6 tokens leaves vLLM's skinny GEMM (1-5 tokens per step) for torch's
   generic path. **With tuned kernels the cliff is gone** (K=5 code 0.55x ->
   1.67x), so its size came from the badly chosen default kernel, not from the
   switch itself. K=3-4 barely move, since their 4-5 token checks never left
   the skinny path: tuning moved only what it should.
2. **Tuned, the best K depends on the workload,** as on llama.cpp: K=4 for
   code, K=8 for extraction (2.02x), K=3 at most for prose.
3. **vLLM gains more than llama.cpp on the same-sized target** (3B code 1.77x
   against a loss at every K). In FP16 the draft is 1 GB against the target's
   6 GB, a smaller share than the Q4 pair, and the skinny GEMM checks 2-5
   tokens for little more than the cost of 1.
4. **Speculation is a single-stream optimization.** Tuned, the best draft
   config is 0.67x at c=16 and 0.52x at c=32. Batching already fills the
   bandwidth speculation exploits, and each check multiplies the tokens per
   step by K+1. It still helps a little at c=4 (1.03-1.18x for K=3-6). The
   untuned run's apparent 1.6-1.9x "win" at c=16-32 was an artifact: the
   plain 3B past 5 tokens per step was dominated by a GEMM whose cost hardly
   depends on row count, so the check tokens rode along for free.
5. **ngram (prompt lookup) only pays on extraction** (~1.5x), and the draft
   model beats it even there. It loses more as concurrency grows (1.5B: 0.94x
   at c=1 to 0.63x at c=32).
6. **In vLLM speculation doesn't change the text more than plain does:**
   speculative runs diverged on 0-2 of 6, plain-vs-plain on 1 of 3.

---

## 04 - Jacobi decoding (side experiment)

Can a plain model produce several tokens per pass with no draft at all?
Guess the next n tokens, run the model once over all of them, replace each
guess with the model's prediction for that position, repeat until the guesses
stop changing. It converges to exactly the greedy output.

`03_jacobi.py`, run group `p9-jacobi-20261023-1348`, 90 rows. Hand-written in
PyTorch on HF transformers (eager), Qwen2.5-1.5B-Instruct FP16 in WSL, the
same nine prompts. Window 0 is plain greedy in the same loop.

| window | code | extraction | prose | guesses right |
|---|---|---|---|---|
| 0 (plain) | 36.9 tok/s | 36.7 tok/s | 37.1 tok/s | |
| 2 | 1.00x, 1.05 tok/pass | 1.02x, 1.11 | 0.98x, 1.06 | 2.5-5.7% |
| 4 | 1.00x, 1.06 | **1.07x, 1.13** | 0.99x, 1.06 | 1.5-3.2% |
| 8 | 0.99x, 1.06 | 1.06x, 1.14 | 0.97x, 1.06 | 0.7-1.8% |
| 16 | 1.00x, 1.06 | 1.06x, 1.14 | 0.98x, 1.05 | 0.3-0.9% |

**The model can't guess ahead of itself.** Fewer than 6% of guesses turn out
right, and windows past 4 add nothing. Each extra token comes from a short
copied span settling, which is why extraction leads. That's the answer to "why
one token at a time": a model trained only on next-token prediction doesn't
know token 3 until token 2 is real. A draft model gets 0.8-0.95 acceptance
because it is a second model trained on the same task.

The wall-clock gain is ~0 (best 1.07x). A pass costs 27-30 ms whatever its
width here, which is HF eager overhead hiding the GPU work (the same model
runs at 151 tok/s in vLLM). In an efficient backend the pass cost grows with
width (section 02), and Jacobi would be a loss. 1.14 tokens per pass is the
ceiling it could reach anywhere.

---

## 05 - Context length and the KV cache

`04_context.py` (speed and VRAM, `p9-context-20261023-1401`, 73 rows) and
`05_kv_fidelity.py` (quality, `p9-kvfid-20261023-1434`, 10,240 scored
positions). Run on llama-server directly, not Ollama: same engine, but Ollama
only takes the KV type as a service-wide environment variable. Each (model,
depth, KV type) got its own server with `-c` = depth + 512. Prompts are the
lab's own docs cut to each depth; 128 tokens generated, 2 reps.

### Decode speed against context (tok/s; % vs the 512-token depth)

| config | 512 | 2k | 8k | 16k | 31.7k |
|---|---|---|---|---|---|
| 1.5b f16 | 222 | 217 | 206 | 188 (-15%) | 165 (-26%) |
| 1.5b q8_0 | 208 | 196 | 177 | 154 (-26%) | 125 (-40%) |
| 1.5b q4_0 | 196 | 184 | 156 | 131 (-33%) | 99 (-50%) |
| 14b f16 | 50.2 | 48.6 | 38.7 | 29.8 (-41%) | **6.1, paged** |
| 14b q8_0 | 49.9 | 47.4 | 41.2 | 35.0 (-30%) | 26.1 (-48%) |
| 14b q4_0 | 47.2 | 45.8 | 36.7 | 29.6 (-37%) | 21.3 (-55%) |
| vLLM 1.5B FP16 | 151 | 148 | 136 | 126 (-17%) | 114 (-25%) |

A better model than the byte-ratio formula is step time = fixed part + a cost
per token of context. Fitting decode ms/token against context:

| config | step at 0 ctx | + per 1k ctx | effective KV read rate |
|---|---|---|---|
| 1.5b f16 | 4.45 ms | 0.049 ms | 580 GB/s |
| 1.5b q8_0 | 4.80 ms | 0.098 ms | 155 GB/s |
| 1.5b q4_0 | 5.01 ms | 0.159 ms | 51 GB/s |
| 14b f16 | 18.8 ms | 0.870 ms | 226 GB/s |
| 14b q8_0 | 19.5 ms | 0.576 ms | 181 GB/s |
| vLLM 1.5B f16 | 6.63 ms | 0.068 ms | 422 GB/s |

VRAM per token of context, measured as the slope of the process's dedicated
VRAM over `-c`: 1.5b 29.8 / 16.3 / 9.1 KB (f16 / q8_0 / q4_0), 14b 198.0 /
107.4 / 58.2 KB. That's 1-3 KB above the KV itself, from llama.cpp's compute
buffer growing with `-c`.

### Quality (teacher-forced against the f16-KV greedy text; top-1 / KL)

| model, KV | 2k docs | 2k code | 16k docs | 16k code | 31.7k docs |
|---|---|---|---|---|---|
| 1.5b q8_0 | 97.3% / 0.0044 | 94.9% / 0.0030 | 97.7% / 0.0028 | 96.5% / 0.0036 | 96.5% / 0.0053 |
| 1.5b q8_0 K + q4_0 V | 96.1% / 0.0092 | 96.1% / 0.0084 | 96.5% / 0.0053 | 97.3% / 0.0108 | 95.7% / 0.0110 |
| 1.5b **q4_0** | **2.3% / 3.77** | **1.6% / 3.92** | **2.3% / 4.54** | **1.2% / 3.85** | **1.2% / 4.15** |
| 14b q8_0 | 98.0% / 0.0011 | 98.8% / 0.0008 | 98.4% / 0.0018 | 98.8% / 0.0018 | - |
| 14b q4_0 | 94.5% / 0.0149 | 96.1% / 0.0095 | 94.1% / 0.0159 | 91.8% / 0.0212 | - |

For scale (Phase 6, 3B weights): Q8_0 KL 0.0016, Q6_K 0.0056, Q5_K_M 0.0168,
Q4_K_M 0.058. The f16 self-test scored 100% top-1 in every cell.

Findings:
1. **A q4_0 K cache breaks the small Qwen2.5 models outright.** Top-1 1-3%,
   gibberish even on a six-word prompt. V at q4_0 is harmless, and the 14b
   tolerates q4_0 K. Likely cause: Qwen2.5's K projection has large biases,
   and q4_0 shares one scale per block of 32 values, so one huge channel
   leaves 4 bits no room for its small neighbours. Severity follows each
   model's largest K bias read from the GGUF, and the 7b was predicted to
   break before it was run:

   | model | largest K bias | q4_0 K, short prompt | q4_0 K, 2k prompt |
   |---|---|---|---|
   | 7b | 414 | broken | broken |
   | 1.5b | 316 | broken | broken |
   | 0.5b | 214 | fine | broken |
   | 3b | 92 | fine | degraded |
   | 14b | 33 | fine | fine |

   Consistent with the cause and predictive, but not proven causally.
   **`OLLAMA_KV_CACHE_TYPE=q4_0` silently ruins qwen2.5:1.5b and 7b.** Nothing
   errors.
2. **q8_0 KV costs about as much quality as Q8_0-Q6_K weights** (KL
   0.001-0.006), with no clear growth to 31.7k. q8_0 K + q4_0 V works on every
   model (KL 0.002-0.011).
3. **Quantized KV is slower on the 1.5b at every depth** (-7% to -40%). The
   per-token cost of context doubles at q8_0 and triples at q4_0, although
   the bytes read halve and quarter. Quantized attention is compute-bound
   here; dequantization inside the flash-attention kernel is the likely cost
   (not profiled).
4. **On the 14b, q8_0 wins from 8k on** (+18% at 16k), and at 31.7k it is the
   only config that fits. The 14b's f16 attention reads KV at only 226 GB/s,
   against 580 on the 1.5b with the same kernel, which is why the 14b loses
   speed faster than predicted. Why is open.
5. **Fit: q8_0 fits the 14b at 32k (12.1 GB).** f16 fits 16k (11.95 GB) and
   pages at 31.7k: 1.7 GB went to shared memory, decode fell to 6.1 tok/s
   and prefill to 248 tok/s. The process got at most ~13.3 GB dedicated,
   which puts the f16 limit at ~22k (extrapolated).
6. **vLLM matches the byte formula** (-25% vs -22% at 31.7k), because its FP16
   weights stream near bandwidth. llama.cpp doesn't, in either direction: the
   1.5b's weights stream at only ~220 GB/s, so context adds relatively less,
   and the 14b's attention is slow, so it adds more.
7. **Prefix reuse on llama.cpp is not bit-exact.** Logits depend on how the
   prompt is cut into prefill chunks, and a cached prefix moves the chunk
   boundaries. The last prompt token's logprobs move ~0.04 nats, enough to
   flip greedy near-ties. Same effect as speculation's wider check pass, from
   a different place.

---

## 06 - Prefix caching

`06_prefix_cache.py`, run groups `p9-prefix-20261023-1508` (1.5b, four
backends, 432 rows) and `p9-prefix-20261023-1511` (14b, 108 rows), 6 reps,
backend order rotated per rep. An agent-shaped prompt: a 2,205-token system
prompt (preamble, 8 tool definitions, reference notes) and a short user turn.

**What Ollama 0.40.9 actually runs.** For qwen2.5 it starts llama-server
itself: `-c num_ctx*np -np N -b 1024 -ub 1024 --flash-attn auto
--context-shift --keep 4`, with llama-server's default 8 GiB **RAM prompt
cache**. llama-server picks a slot by longest-common-prefix similarity (over
0.1), else LRU, saves an idle slot to RAM before reusing it, and restores a
saved prompt when it matches better.

### Where the timestamp goes (TTFT, ms; streamed, client-side)

| timestamp position | vLLM | Ollama np 1 | Ollama np 4 | llama-server, no RAM cache | 14b Ollama |
|---|---|---|---|---|---|
| in the user turn (warm) | 48 | 42 | 49 | **18** | 90 |
| top of the system prompt | 275 | 288 | 295 | 248 | **2,051** |
| 1/4 in | 213 | 238 | 253 | 197 | 1,681 |
| 1/2 in | 113 | 188 | 187 | 147 | 1,269 |
| 3/4 in | 86 | 108 | 94 | 77 | 621 |
| end of the system prompt | 55 | 40 | 41 | 20 | 114 |

### Two conversations taking turns (ms; steps after the first pair)

| | vLLM | Ollama np 1 | Ollama np 4 | llama-server, no RAM cache | 14b Ollama |
|---|---|---|---|---|---|
| per switch | 48-56 | 72-80 | 70-77 | 171-184 | 257-295 |

Findings:
1. **Where per-turn data goes is the biggest lever.** On the 14b, a
   timestamp on the first line of the system prompt makes every turn cost
   2.05 s to first token; in the user turn it costs 0.09 s, **23x** less. The
   cost is linear in the tokens after the edit (0.107 ms per uncached token on
   llama-server).
2. **`NUM_PARALLEL` > 1 neither loses nor gains hits for sequential
   requests.** Conversation B shares the 805 tool-definition tokens with A, a
   prefix similarity of 0.33. That clears the 0.1 threshold, so B takes A's
   slot instead of an empty one and A goes to the RAM cache. The other three
   slots stay empty.
3. **The RAM prompt cache does the work.** An evicted conversation comes back
   with all but ~18 tokens cached: ~35-40 ms per swap on the 1.5b (against
   ~140 ms to re-prefill) and ~190 ms on the 14b (against ~1.5 s). It moves at
   3.5-4.5 GB/s counting both copies, a fraction of PCIe 4.0's ~25 GB/s. vLLM
   pays nothing: its cache is shared and stays on the GPU.
4. **Ollama's own layer adds ~20-25 ms per request** (42 ms against 18 ms for
   llama-server with the same flags; identical server-side prompt eval). It
   doesn't grow with prompt length. That locates Phase 5's 21.7 ms Ollama
   fixed cost: it is outside the inference engine.
5. **vLLM is not faster here.** Cold 275 ms against llama-server's 248, warm
   48 ms against a predicted 15, and its warm cost isn't linear in the
   uncached tokens. Not explained.
6. **Multi-turn history is reused on every backend.** llama.cpp re-prefills
   the new turn plus exactly 1 token (the end-of-turn token the template
   rebuilds), vLLM plus up to 16 (its block size).

---

## 07 - Tuned GEMMs: the biggest win of the phase

**Root cause, from Stage 1's broken 3B concurrency numbers.** In vLLM on
ROCm, steps of 1-5 tokens per step use vLLM's own skinny kernels; 6 or more
go to `torch.nn.functional.linear`, which picks a rocBLAS/hipBLASLt kernel by
default heuristics. On gfx1101 those choices are poor for every skinny FP16
shape. The 3B's merged gate_up projection (2048 -> 22016) took 960-1,500 us
per call at any row count, 60-180 GB/s on a 624 GB/s card. Its lm_head took
2.8-4.2 ms. A good kernel exists in the same libraries: PyTorch TunableOp
finds one at ~170 us and 1.1 ms.

**The fix.** `07_tune_gemms.py` tunes offline every GEMM vLLM sends to torch
for the 0.5B, 1.5B and 3B: each layer at every CUDA-graph size of 6+ tokens,
plus lm_head at every row count up to 48 (logits are computed outside the
graph, on the sampled rows only). 483 shapes, 23 minutes, once. vLLM then
starts with `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=0` and
`PYTORCH_TUNABLEOP_FILENAME`, so CUDA-graph capture only reads the table.

One decode step's GEMMs, simulated with distinct weights per layer
(`--step`, ms):

| rows | 1.5B default | 1.5B tuned | 3B default | 3B tuned |
|---|---|---|---|---|
| 8 | 12.4 | 7.9 | 82.7 | 12.6 |
| 32 | 13.2 | 8.0 | 90.0 | 12.8 |

Tuned, the 3B's GEMMs stream 6.2 GB in 12.8 ms (~480 GB/s, 77% of peak) and
the 1.5B's 3.1 GB in 8.0 ms (~390 GB/s). These savings account for the
measured step times: 1.5B at c=32 goes from ~16 to ~10.5 ms per step, 3B from
~97 to ~17 ms.

Results (sections 03 and 08 have the full tables):
- **1.5B: +48-64% at c=8-32,** 3,056 tok/s at c=32 (from 2,004).
- **3B: 5.7x at c=32,** 1,897 tok/s (from 331).
- c=1 and c=4 don't move: those steps never reach torch.
- The batch-6 cliff shrinks but doesn't vanish: the 1.5B's per-stream rate
  goes from 144 tok/s at c=4 to 107 at c=8 (72 untuned).

**Why the prediction was too low:** Stage 1's single-GEMM microbenchmark
reused one weight, which stays in the 64 MB Infinity Cache and hides how slow
the default kernel is on a real step. Simulate full steps with distinct
weights instead.

**Costs and gotchas:**
- Startup is unchanged (44 s against 42 s).
- The table is tied to the exact PyTorch, ROCm, hipBLASLt and rocBLAS
  versions. A venv upgrade means re-tuning.
- The `PYTORCH_TUNABLEOP_FILENAME` reader inserts the device ordinal:
  `qwen25.csv` is opened as `qwen250.csv`. The first tuned run silently used
  no table at all. The tuner therefore writes with
  `set_filename(..., insert_device_ordinal=True)`.
- The env var overrides `torch.cuda.tunable.enable(False)`.

---

## 08 - Concurrency: Ollama against vLLM

`08_concurrency.py --part sweep`, run group `p9-conc-20261023-1557`, 1,107
rows. Qwen2.5-1.5B, 256 tokens out per request, 3 batches per level, each
backend with the card to itself. Ollama serves Q4_K_M and vLLM FP16, so
Ollama reads about a third of the weight bytes per token; that is why it wins
at c=1.

Aggregate tok/s (median of 3):

| backend | c=1 | c=4 | c=8 | c=16 | c=32 |
|---|---|---|---|---|---|
| vLLM, default GEMMs | 151 | 562 | 553 | 1,059 | 2,004 |
| vLLM, tuned GEMMs | 151 | 565 | **820** | **1,733** | **3,056** |
| Ollama NUM_PARALLEL 8 | 227 | 601 | 712 | 720 | 716 |
| Ollama NUM_PARALLEL 16 | 230 | 605 | 713 | 953 | 1,006 |
| Ollama NUM_PARALLEL 32 | 230 | 601 | 710 | 962 | 1,125 |
| vLLM tuned, `--max-num-seqs 8` | | | | | 833 |
| vLLM tuned, `--max-num-seqs 16` | | | | | 1,754 |

Median request latency at c=32: vLLM tuned 2.6 s, vLLM default 3.9 s,
Ollama np32 7.0 s (38 tok/s per stream), `--max-num-seqs 8` 5.8 s,
`--max-num-seqs 16` 3.2 s.

Findings:
1. **Ollama tops out at ~1,125 tok/s.** It beats untuned vLLM at c=8 (712 vs
   553), which is where Phase 5 saw it win. Tuned vLLM leads from c=8 on, by
   2.7x at c=32 with 2.7x lower median latency.
2. **Extra Ollama slots cost nothing but KV memory.** np32 at c=8 equals np8
   at c=8. Since `-c` is num_ctx x np, np32 at 4k preallocates 131,072 tokens
   of KV. Ollama also raises `-b`/`-ub` from 1,024 to 2,048 once the total
   context is large (np16+ here, np9 at num_ctx 8192).
3. **`--max-num-seqs` is a queue, nothing more.** A cap of 8 at c=32 gives
   the c=8 throughput at 2.2x the latency; per-stream speed hardly improves
   (107-116 against 102 tok/s). Cap it only to bound memory.

---

## 09 - Inter-token latency while a long prompt arrives

`08_concurrency.py --part mixed`, run group `p9-conc-20261023-1606`, 108
rows, raw chunk times in `logs/p9-conc-20261023-1606-mixed.json`. Eight
streams decode a 512-token story each; mid-way a 3,865-token prompt arrives
(16 tokens out). vLLM is tuned. Medians of 3 reps:

| backend | prefill chunk | long-prompt TTFT | base ITL | largest gap | excess per stream |
|---|---|---|---|---|---|
| vLLM `--max-num-batched-tokens 512` | 512 | 470 ms | 9.2 ms | **69 ms** | 375 ms |
| Ollama NUM_PARALLEL 9 | 2,048 (`-b`) | 469 ms | 9.8 ms | 210 ms | 409 ms |
| vLLM 2,048 (the default) | 2,048 | 473 ms | 9.2 ms | 234 ms | 420 ms |
| vLLM 8,192 | whole prompt | 431 ms | 9.2 ms | 381 ms | 390 ms |

**The chunk size sets the worst gap, not the total delay.** Every backend
adds 375-420 ms per stream, about the prompt's prefill time: the card has to
do that work either way. What chunking changes is how it's spread. The first
plan's prediction, that Ollama stalls the streams for the whole prefill, was
wrong: llama-server puts the running decodes into every prefill batch, so
Ollama behaves like vLLM at 2,048. A 512-token chunk cuts the worst stall
3.4x and costs the long prompt's TTFT 9%.

---

## What it means for the user's agent

- **Keep the system prompt and tool definitions byte-stable; put anything
  per-turn (time, ids, counters) in the last message.** On the 14b that's
  2.05 s against 0.09 s to first token, every turn.
- **Never set `OLLAMA_KV_CACHE_TYPE=q4_0` with qwen2.5 1.5b or 7b.** q8_0 is
  safe and is what lets the 14b hold 32k on this card, but it's slower than
  f16 at short context on small models.
- **Speculative decoding helps one agent loop on tool-output-heavy turns**
  (1.6-2.0x on extraction), on the 14b in llama.cpp or the 3B in vLLM. It
  hurts once more than about 4 calls run at once. Ollama doesn't expose it.
- **On vLLM on this card, TunableOp is not optional** for parallel calls:
  +50% on the 1.5B, 5.7x on the 3B.
- **On Ollama, set `NUM_PARALLEL` at least to the fan-out.** Extra slots cost
  only VRAM. It won't help a sequential agent's cache, and on 0.40.9 the RAM
  prompt cache already makes subagent switches cheap.
- **The old agent log's 23% cache hit rate isn't explained by this Ollama.**
  0.40.9 hit 99% of the prompt on every sequential pattern tested. The log
  predates 0.40, so look for something in the agent's prompts that varies
  near the start.

---

## The code

| file | what |
|---|---|
| `phase9-optimization/common.py` | servers (llama-server, side Ollama, vLLM with env and log label), streaming clients with per-chunk times, the TunableOp env |
| `phase9-optimization/workloads.py` | the nine prompts: code, extraction, prose |
| `phase9-optimization/01_spec_llamacpp.py` | speculative decoding on llama-server, K sweep, check-pass timing |
| `phase9-optimization/02_spec_vllm.py` | speculative decoding and ngram on vLLM, single stream and concurrency; `--tuned` |
| `phase9-optimization/03_jacobi.py` | Jacobi decoding in HF transformers (runs in WSL) |
| `phase9-optimization/04_context.py` | decode speed and VRAM against context depth and KV type |
| `phase9-optimization/05_kv_fidelity.py` | KV quantization quality, teacher-forced |
| `phase9-optimization/06_prefix_cache.py` | cold/warm/edited-prefix TTFT, turn-taking, multi-turn |
| `phase9-optimization/07_tune_gemms.py` | offline TunableOp tuning (runs in WSL); `--step` simulates a decode step |
| `phase9-optimization/08_concurrency.py` | the Ollama/vLLM sweep, `--max-num-seqs`, the mixed-load ITL test; `--report` prints the tables |

Results are in `lab.db` under the run groups named above. The TunableOp table
lives outside the repo, at `~/tunableop/qwen250.csv` in WSL; regenerate it
with `07_tune_gemms.py`.

---

## Limitations

- **Backends compare different bytes.** Ollama/llama.cpp ran Q4_K_M, vLLM
  FP16 (no GGUF path here). Cross-backend numbers compare stacks as an agent
  would use them, not engines on equal terms.
- **The largest vLLM target is the 3B.** A 7B in FP16 doesn't fit, and
  quantized vLLM formats weren't tried.
- **Nine prompts** decide every speculation result, and acceptance depends
  entirely on the text. The order extraction > code > prose may not hold on
  other code.
- **2-3 reps per configuration,** interleaved where it mattered.
  Between-session drift on this machine is ~4%, so differences under ~5%
  across run groups are noise.
- **The tuned table covers decode shapes only** (6-288 rows, lm_head to 48).
  Prefill GEMMs with more rows still use the default kernels, including the
  prefill in the mixed-load test. Tuned prefill could change its TTFT column.
- **KV-quantization quality was measured on the 1.5b and 14b only;** the
  other sizes only got the q4_0 K short/2k-prompt check.
- **Mixed load used one prompt size and 8 streams.** Prefill-heavy throughput
  at a 512-token chunk was not measured.

---

## Open items

- **Why the 14b's f16 attention reads KV at 226 GB/s** against the 1.5b's 580
  on the same kernel. One untested idea: the 1.5b's per-layer KV fits the
  64 MB Infinity Cache at these depths and the 14b's doesn't, but the 14b is
  already slow at 8k (33 MB per layer).
- **A causal test of the q4_0 K failure:** subtract the K bias before
  quantizing, or quantize K per channel.
- **What makes quantized-KV attention compute-bound** on the 1.5b. No
  `rocprof` under WSL; on Windows, Radeon GPU Profiler might work.
- **vLLM's warm prefix TTFT** (48 ms, non-linear in uncached tokens). A
  suffix prefill on a cached prefix may run outside the CUDA graphs.
- **The RAM prompt cache's 3.5-4.5 GB/s,** well below PCIe 4.0.
- **Tuning prefill shapes too,** and whether that moves vLLM's cold TTFT
  (Stage 3's 275 ms against llama-server's 248).
- **What made greedy non-repeatable in the Stage 0 probe** (raw `/completion`,
  no `--flash-attn on`, no seed): not isolated.
- **From the plan's optional list, not done:** 32B placement on Windows
  (`OLLAMA_GPU_OVERHEAD` or `num_gpu`), AWQ/GPTQ on gfx1101 (would allow a 7B
  vLLM target), `--cpu-offload-gb`.

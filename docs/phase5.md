# Phase 5 - Inference Fundamentals

Technical notes. Five small experiments on what the numbers in Phases 1-4 are
actually made of: tokens, prefill, truncation, the prefix cache, and the
per-token rhythm of decode. Plus a round of fixes to the Phase 4 backend
control that came out of timing vLLM's startup.

Date: 2026-09-17 (files 01-05), 2026-10-22 (rechecks, write-up, startup fixes)

---

## Goal

Every result so far is per-token, and several pieces of advice given along the
way rested on behaviour that had never been measured: which end Ollama drops
when a prompt is too long, how much a prefix-cache hit saves, how much of TTFT
is prefill. This phase measures each one directly, on the 1.5B, so the later
phases build on numbers rather than assumptions.

| File | Question |
|---|---|
| `01_tokenization.py` | What is a token, and what does text cost in them? |
| `02_prefill_scaling.py` | How is TTFT split between fixed cost and prefill? |
| `03_context_truncation.py` | When a prompt exceeds `num_ctx`, what does Ollama drop? |
| `04_prefix_cache.py` | How much does prefix reuse save, and when does it break? |
| `05_inter_token_latency.py` | Is decode steady, or does it stutter? |

---

## Ollama changed version under the experiments

Files 01-05 ran on 2026-09-17 against **Ollama 0.34.1**. By the write-up it had
auto-updated to **0.40.7**. Rather than quote the old numbers as current, 02, 03,
04 and 05 were rerun on 0.40.7:

| Check | 0.34.1 | 0.40.7 |
|---|---|---|
| Truncation (03) | oldest-first, keeps 258 of 1229 | identical |
| Prefix cache (04), warm hit | 1085 of 1086 cached, 5.3 ms | 1090 of 1091, 6.0 ms |
| Prefill (02) | 8,526 tok/s, 22.3 ms fixed | 8,991 tok/s, 25.6 ms fixed |
| Decode, FP16 1.5B | 121 tok/s (Phase 3) | 122 tok/s |

No behavioural change. The +5% prefill is within the ~4% between-session drift
measured in Phase 1, so it is not attributed to the version. From here on,
`backend_version` matters: results from the two versions should not be mixed in
one comparison.

---

## 01 - Tokenization

Uses vLLM's `/tokenize`, which exposes the model's real tokenizer. Qwen2.5
shares one tokenizer across every size, so these hold from 0.5B to 32B.

```
what            chars words tokens chars/tok
plain English      23     6      7       3.3
rare word          40     3      9       4.4
Swedish            40     8     14       2.9
numbers            25     3     22       1.1
code               32     8     12       2.7
json               38     4     16       2.4
a URL              36     1     11       3.3
```

- **Digits are one token each.** `1234567890` is 10 tokens. Deliberate - it
  keeps arithmetic consistent - and expensive in practice.
- **The space belongs to the following token.** `' the'` (279) and `'the'`
  (1782) are different tokens, as are `'The'`, `' The'` and `'THE'`.
- **Swedish costs 1.31x English** for the same sentence (17 vs 13 tokens). The
  vocabulary was fitted mostly to English.
- **JSON runs 2.4 chars/token against prose's 3.3** - about 1.4x the tokens for
  the same characters, before counting the digits inside it.

The last two are why a context window measured in characters lies. An agent's
tool results - JSON full of IDs and numbers - fill `num_ctx` far faster than
their character count suggests.

---

## 02 - What TTFT is made of

Phase 1 measured "prefill" once, at a 41-token prompt, and got 967 tok/s. That
was flagged as unreliable at the time: at that size TTFT is mostly fixed cost.
This sweeps prompt length (77 to 2037 tokens, `max_tokens=1`, a unique prompt
per trial so the prefix cache cannot help) and fits

```
TTFT = intercept + slope x input_tokens
```

Per-trial least-squares fit, 18 trials per run:

| Backend | Fixed cost | Prefill | R^2 | A 2,000-token system prompt adds |
|---|---|---|---|---|
| vLLM (2026-09-17) | 9.8 ms | 11,513 tok/s | 0.966 | 174 ms |
| Ollama 0.34.1 (2026-09-17) | 22.3 ms | 8,526 tok/s | 0.993 | 235 ms |
| Ollama 0.40.7 (2026-10-22) | 25.6 ms | 8,991 tok/s | 0.996 | 222 ms |

The relationship is close to perfectly linear. Prefill overtakes the fixed cost
at roughly 200 input tokens; below that TTFT barely depends on prompt length.

Phase 1's 967 tok/s was the intercept speaking: 41 tokens in 42 ms is almost all
fixed cost. Real prefill is ~9x faster than that.

### This undercuts a Phase 3 explanation

Phase 3 reported TTFT at concurrency 1 as 16-17 ms for Ollama and 25-29 ms for
vLLM, and attributed the gap to WSL2 network transport. Every one of those
trials sent the same 41-token prompt (`lab.db`: one distinct prompt per run
group). Cache hits were not recorded - the OpenAI interface does not report
them - so what follows is inferred, not measured:

- **Ollama's 17 ms is below its own uncached fixed cost** (22-26 ms here). It
  can only have been served from the prefix cache.
- **vLLM's 25.5 ms is above what this fit predicts uncached** (9.8 ms + 41
  tokens = ~13 ms). If it was cached, caching did not make it faster.

With unique prompts the order **reverses**: vLLM's fixed cost is 9.8 ms against
Ollama's 22-26 ms. WSL transport cannot be both the reason vLLM is slower in
Phase 3 and invisible here. The caution "TTFT is not comparable across backends"
stands, but its stated cause is at best incomplete. Not resolved - it needs
cached vs uncached TTFT on both backends, same request shape, one session.

---

## 03 - Which end Ollama drops

Owed since Phase 1. The advice given for the agent (pin the system prompt, trim
the middle) assumed Ollama truncates oldest-first. Never verified until now.

**Method.** Plant `ALPHA7` near the start and `OMEGA3` near the end of a
1,229-token prompt, ask for both. A control at `num_ctx` 4096 must pass first -
if the model cannot repeat both codes when both are present, the truncated
result means nothing. It passed. Then `num_ctx` 512.

**Oldest-first, confirmed.** The start code was gone, the end code survived.
The server's own `prompt_eval_count` confirms it independently of the model's
answer.

**Overflow costs half the window**, not just the excess:

```
num_ctx   tokens processed   ratio
    256                130    0.51
    512                258    0.50
   1024                514    0.50
   2048               1146    fits, no truncation
```

On overflow Ollama keeps roughly `num_ctx / 2`. Exceed 4096 by one token and the
model works from ~2048.

**The model fills the gap with a guess.** Told explicitly to write MISSING for
any code it was not given, it answered `START_CODE=OMEGA3 END_CODE=OMEGA3` -
the nearest available value, stated confidently. 971 tokens dropped with no
error, no warning, and nothing in the response except `prompt_eval_count` to
say so.

Consequence: never rely on server-side truncation. Trim client-side with a
margin - crossing the limit costs half the window, which justifies a generous
one - and detect truncation by comparing what was sent against
`prompt_eval_count`.

---

## 04 - Prefix cache

Token i's keys and values depend only on tokens 0..i, so if a new prompt starts
with the same tokens as the last one, their K/V can be reused. The consequence
is asymmetric, and the asymmetry is the practical point. Ollama's native API
reports `prompt_eval_cached_count` per request, which makes it a direct
instrument (vLLM caches too but does not expose per-request hits).

Ollama 0.40.7, qwen2.5:1.5b, 1,091-token prompt:

```
case         input  cached  hit %  prefill ms  what changed
1 cold       1091      24     2%       223.2  novel text
2 warm       1091    1090   100%         6.0  byte-identical repeat
3 suffix     1091    1083    99%        18.8  same start, different ending
4 prefix     1091      24     2%       114.6  different start, same ending
```

Cases 3 and 4 differ from their predecessor by about the same number of tokens.
One keeps 99% of the cache, the other loses all of it. **Where the change is
matters, not how big it is.** The 24 tokens that survive every case are the chat
template's header.

A full hit cuts prefill from ~115 ms to 6 ms. So: stable content first (system
prompt, tool definitions), varying content last. A timestamp or session id near
the start of a prompt invalidates everything after it, every call.

Unexplained: case 1 takes about twice case 4 (223 vs 115 ms) though both are
~2% cached. Case 4 matches what 02 predicts for ~1,070 uncached tokens; case 1
does not. The same ratio appeared on 0.34.1 (202 vs 116 ms), so it is
consistent, not noise. Possibly a first-allocation cost for a new prompt length.

---

## 05 - Inter-token latency

Every throughput figure in this project is an average. An average cannot tell a
metronome from a server that emits ten tokens instantly and then stalls. This
records the arrival time of every token. 400 tokens, TTFT excluded:

| | Ollama 0.40.7 FP16 | vLLM 0.29 FP16 |
|---|---|---|
| mean | 8.18 ms (122.2 tok/s) | 6.54 ms (152.8 tok/s) |
| p50 | 8.10 ms | 6.54 ms |
| p99 | 8.77 ms | 6.97 ms |
| max | 11.37 ms | 7.23 ms |
| stdev | 0.25 ms | 0.16 ms |

**Decode is steady on both.** p99 is within 8% of the median; the slowest 1% of
gaps account for ~1% of generation time. The averages quoted throughout the
project describe what a reader actually sees. vLLM is the tighter of the two -
its worst gap is 1.1x its median, Ollama's 1.4x.

---

## Fixes to the Phase 4 backend control

Starting vLLM from the UI felt like 2-3 minutes. Timed with a timestamp on every
log line:

| Stage | Time |
|---|---|
| Python/torch import (API server) | ~16-18 s |
| Engine process spawn and second import | ~8-10 s |
| Weight load, 2.88 GiB | 4-11 s |
| torch.compile, from cache | ~5 s |
| CUDA graph capture, 51 sizes, **twice** | ~16 s |
| API setup, two Hugging Face network checks | ~5 s |
| **Total** | **~75 s warm, 68 s after `wsl --shutdown`** |

The 2-3 minutes was not reproduced. Nothing downloads; the model and compiled
kernels come from cache. Graphs are captured twice because vLLM's memory
profiler does one capture to estimate graph memory before the real one.

Changes in `servers.py`:

- **`HF_HUB_OFFLINE=1`**, so startup no longer makes network calls.
- **CUDA graphs for 13 batch sizes** (1-8, 10, 12, 16, 24, 32) instead of 51.
  Capture fell from ~16 s to ~2 s. The first version captured only 1, 2, 4, 8,
  16 - which turned out to halve batch-5 throughput; see the cliff below.
- **`--gpu-memory-utilization` 0.85 -> 0.35.** At 0.85 the KV cache held 364k
  tokens (89 concurrent 4k requests) for a lab that runs 8. At 0.35 it holds
  50,272 (12x).
- **Optional coexistence.** `start(name, keep_other=True)`, exposed as a
  checkbox in the Backends panel, leaves the other server running.
- **Full startup log** in `phase4-observability/logs/<name>.log`, each line
  stamped with seconds since start and a READY line from a 0.5 s health poll.
  The UI keeps only 40 lines, which is why the first slow start could not be
  diagnosed after the fact.

Result: **37-49 s** from click to ready with the compile cache warm, against
~75 s. The spread is in the import phase, before vLLM does any real work. Changing any
compile setting changes vLLM's cache key, so the first start after such a
change recompiles (+14 s) - time the second. A start beside a resident Ollama
took 48 s in one observation.

Not done: `--enforce-eager` would skip compile and graphs entirely, but decode
gets slower, which disqualifies it for a benchmarking lab. Disabling the
memory-estimate capture (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`) would
save another second but removes graph memory from vLLM's budget - the wrong
trade when two servers share the card.

### Verified, not assumed

**vLLM throughput is unchanged** by the graph and memory changes:

```
conc   Phase 3 (2026-09-16)   now (2026-10-22)
   1          148.9                150.1
   2          287.0                293.0
   4          563.0                566.2
   8          563.8                569.2
```

**Coexistence is safe for the 1.5B.** Phase 4 recorded that with vLLM at 0.85,
Ollama's allocation still succeeds and pages to system RAM: 8.1 tok/s while
reporting `100% GPU`. At 0.35 with both resident, interleaved, 3 runs each:

```
Ollama alone           121.7 tok/s
Ollama, vLLM resident  119.0 tok/s
vLLM, Ollama resident  151.8 tok/s  (alone: 152.8)
```

Not paging. About 2% on Ollama - fine for convenience, not for a headline
number. The checkbox is off by default and deliberately not remembered: a 14B
on Ollama beside vLLM would page silently exactly as before.

---

## vLLM's "batch-4 ceiling" is a kernel switch at batch 6

The verification sweep above was extended to 16, which Phase 3 never ran, and
gave 1,082 tok/s - nearly double batch 8. That contradicted Phase 3's central
concurrency finding, so it was chased down.

### The fine sweep

vLLM alone, a CUDA graph captured for **every** level (so no batch is padded to
a larger one), 300-token requests, three sweeps interleaved:

```
conc   aggregate   per request     (median of 3 sweeps, tok/s)
   1       151          152
   2       279          148
   3       422          148
   4       556          145
   5       695          144
   6       438           74    <- halves here
   7       502           73
   8       570           72
  10       703           72
  12       838           71
  16      1091           69
  24      1560           68
  32      2052           66
```

One cliff, between 5 and 6. Either side of it, per-sequence speed barely moves
and aggregate grows almost linearly. Batch 8 - Phase 3's last point - sits just
past the cliff, which is what made 4 -> 8 look like a wall.

### The cause

vLLM's ROCm linear-layer dispatch (`model_executor/layers/utils.py`):

```python
if m > 8 and 0 < n <= 5:
    out = ops.wvSplitK(weight, x_view, cu_count, bias)   # hand-written skinny GEMM
...
# otherwise: the generic torch GEMM
```

`n` is the number of tokens in the step - in decode, one per sequence. Up to 5
sequences vLLM uses a hand-written kernel for thin matrix multiplies; from 6 it
uses the generic GEMM.

**Confirmed by switching it off.** With `VLLM_ROCM_USE_SKINNY_GEMM=0`:

```
conc        1     2     4     5     6     8    16
per-req    76    75    73    73    73    72    69   tok/s
```

The cliff disappears: everything runs at the slow speed, and batches 6+ are
unchanged because they never used the kernel. Phase 3 guessed "some GEMM or
attention kernel... that tiles well up to batch 4 and falls off after", and
noted the question was blocked without `rocprof`. It needed no profiler - the
dispatch condition is plain Python in vLLM's source.

### What this changes

- **vLLM's single-stream lead over Ollama is that one kernel.** 153 tok/s with
  it, 76 without - against Ollama's 121. On the generic path vLLM would lose
  every single-stream comparison in this project.
- **Past the cliff vLLM scales well.** 2,052 tok/s at 32 concurrent requests.
  Phase 3's conclusion that llama.cpp overtakes vLLM holds at concurrency 8
  only. Ollama was never measured above 8 (`OLLAMA_NUM_PARALLEL` is 8 here), so
  there is no fair comparison at 16-32 yet.
- **Graph padding can change the kernel.** vLLM pads a batch up to the next
  captured graph size. With sizes 1, 2, 4, 8, 16 - this phase's first startup
  fix, and also vLLM's own default sequence - batch 5 is padded to 8 and
  measured **78 tok/s per request instead of 144**. In a stock vLLM the cliff
  is effectively at 5, which is why batch 4 looked like the last fast point.
  `servers.py` now captures 5 explicitly.

Phase 3's doc carries a correction pointing here; its original reasoning is
kept, since it was right about the direction and wrong about the shape.

---

## Repetition penalty: the backends disagreed

vLLM's startup log warns that the model's `generation_config.json` overrides
its defaults (`repetition_penalty 1.1`, `top_k 20`, `temperature 0.7`). The
lab's clients pinned `temperature 0` but never set a repetition penalty.
Measured with a prompt that demands repetition (write "banana" 60 times),
greedy, FP16 1.5B on both sides:

```
setting                                 output
ollama native, default                  banana banana banana ...  (41 words)
ollama native, repeat_penalty 1.0       identical to default
ollama native, repeat_penalty 1.05/1.1  HTTP 500 "token repeat limit reached"
ollama native, repeat_penalty 1.3       'banana bananabananabanana... (Note: ...'
ollama /v1, any penalty field           identical to default - field ignored
vllm, default                           'banana bananabananabanabanaba...'
vllm, repetition_penalty 1.0            byte-identical to Ollama
vllm, repetition_penalty 1.1            identical to vLLM default
```

- **Ollama applies 1.0; vLLM applies 1.1.** Every vLLM run before 2026-10-22
  used 1.1, every Ollama run 1.0.
- **At 1.0 the two produce byte-identical greedy output** from the same weights.
  Phase 4 expected cross-runtime output never to match; at least here it does.
- **Ollama's `/v1` endpoint ignores penalty fields.** Only the native API's
  `repeat_penalty` option reaches the sampler.
- **Ollama has a loop guard.** At a mild penalty the model degenerates into a
  repeating `bana` token, and Ollama aborts the generation with HTTP 500 and
  `"prediction aborted, token repeat limit reached"`. A 500 from Ollama can
  mean the model looped, not that the server failed.

**No effect on speed.** The fine sweep was run at both settings, interleaved:
aggregate within 3% at every level, with no consistent direction (batch 4: 550
vs 567; batch 32: 2,065 vs 2,052). Phase 3's throughput numbers stand.

**Pinned to 1.0** in the shared OpenAI client (`phase3-vllm/01_openai_client.py`),
the Ollama native client (`phase1-ollama-baseline/03_repeated_trials.py`), and
files 03 and 05 here. The concurrency harness takes `--repetition-penalty` to
reproduce old runs.

---

## Limitations

- One model (Qwen2.5-1.5B), one card. Prefill rates and the batch-size step
  will move with model size.
- 03 and 04 use Ollama's native API only - `num_ctx` and per-request cache
  counts have no OpenAI-interface equivalent, and vLLM does not report cache
  hits per request.
- 05 is a single 400-token stream per backend. It shows decode is steady when
  nothing else is running; it says nothing about jitter under concurrency.
- The original 2-3 minute vLLM start was never reproduced, so its cause is a
  guess (a cold first-of-day disk read).
- No GPU telemetry on the 2026-10-22 runs: LibreHardwareMonitor was not
  running, and the columns are stored as NULL.

---

## Open items

- Resolve the Phase 3 TTFT explanation: cached vs uncached TTFT on both
  backends, interleaved in one session. (Phase 9.)
- Ollama above concurrency 8 (`OLLAMA_NUM_PARALLEL` 16/32), for a fair
  comparison with vLLM past the cliff.
- Whether the skinny-GEMM cliff is the same size on larger models, and whether
  quantized vLLM models (which use different kernels) have one at all.
- Explain the cold-case prefill in 04 being 2x the prefix-change case.
- Inter-token latency under concurrency, where stalls would actually show.

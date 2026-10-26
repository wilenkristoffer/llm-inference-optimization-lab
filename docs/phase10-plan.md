# Phase 10 plan - The final benchmark

Status: **planned, not started.** This is a plan, not results. Predictions
are written down so the measurements can prove them wrong.

Phases 1-9 measured speed, memory and fidelity one variable at a time. Each
left the same question open: **which model and configuration should actually
run on this card, and what does the choice cost?** This phase answers it with
one benchmark that scores quality on its own absolute scale and measures speed
on the same workload, for every serious candidate, each in its best measured
configuration.

Questions handed to this phase by earlier ones:

- Phase 6: "Ranking across sizes needs an absolute quality measure (a harder,
  larger question set)." At a similar footprint, is **7B Q4_K_M better than
  3B Q8_0**? The 60-question set could not tell FP16 from Q4_K_M.
- Phase 7: the 32B makes every agent turn ~10x slower than the 14B. Whether
  its answers are worth that stays open: the 32B is no longer on disk and
  this phase downloads nothing (see Out of scope).
- Phase 9: which optimizations (speculative decoding, q8_0 KV, TunableOp,
  `NUM_PARALLEL`) hold up on a real task, not only on nine prompts?
- Cleanup plan, Stage 2: the Q8_0 and FP16 references are kept on disk until
  this benchmark is done.

---

## Question

1. **Quality:** on machine-checkable tasks shaped like the user's agent, how
   do the candidates rank, with confidence intervals tight enough to separate
   neighbours?
2. **Cost:** what does each candidate cost per task in wall-clock time and
   VRAM, in its best measured configuration and in Ollama's default?
3. **Choice:** which configurations are on the quality / time / VRAM Pareto
   front, and which one should the agent run?

---

## Starting point (measured, no new runs needed)

| topic | known | from |
|---|---|---|
| decode, Q4_K_M | 1.5b 222, 3b 129, 7b 83-96, 14b 44-52 tok/s | Phases 1, 7, 9 |
| per agent turn (2k prompt + 200 out) | 14b fully on GPU 5.5 s; 32b 52.9-55.5 s | Phase 7 |
| quantization | Q4_K_M sound (93-95% top-1); Q8_0 ~99%; Q2_K broken on 3B | Phase 6 |
| direct llama-server vs Ollama | 52.3 vs 44 tok/s on the 14b; Ollama adds ~20-25 ms/request | Phase 9 |
| speculative decoding | 14b + 0.5b draft 1.6-2.0x single stream on code/extraction; a loss past ~4 concurrent | Phase 9 |
| KV cache | q8_0 safe, holds 14b at 32k; **q4_0 K breaks qwen2.5 1.5b/7b** | Phase 9 |
| prefix cache | stable system prompt: 14b warm TTFT 0.09 s vs 2.05 s cold | Phase 9 |
| fan-out | Ollama np32 1,125 tok/s; vLLM tuned 1.5B 3,056 tok/s at c=32 | Phase 9 |
| the agent's real workload | 381 calls, mail triage with 6 tools, p50 context 2,225 tokens, 4/51 tasks looped on `mail-get-unread` | agent log |
| run-to-run drift | ~4% between sessions | Phase 1 |

---

## Candidates

**Only models already on disk. Nothing is downloaded for this phase.** All
via Ollama unless the backend says otherwise.

| model | quant | weights | why |
|---|---|---|---|
| qwen2.5:1.5b | Q4_K_M | 1.0 GB | smallest that is coherent; vLLM twin in FP16 |
| qwen2.5:3b | Q4_K_M, Q8_0 | 1.9 / 3.3 GB | Phase 6's "smaller at Q8" side |
| qwen2.5:7b | Q4_K_M, Q8_0 | 4.7 / 8.1 GB | Phase 6's "bigger at Q4" side |
| qwen2.5:14b | Q4_K_M | 9.0 GB | the largest Qwen2.5 that fits |
| qwen3:14b | Q4_K_M | 9.3 GB | same size, newer generation; think on/off |
| qwen3.5:4b | BF16 | 9.3 GB | Phase 6's "current model" track; only the BF16 reference is on disk, so it competes at full precision |
| gemma3:12b / gemma4:12b | Q4_K_M | 7.6-8.1 GB | another family, if tool calling works (Stage 0) |
| llama3.1:8b | Q4_K_M | 4.9 GB | another family near the 7B |
| vLLM Qwen2.5-1.5B / 3B | FP16, tuned GEMMs | 3.1 / 6.2 GB | the fan-out backend (already in WSL's HF cache) |

`qwen2.5:0.5b` is the draft model for speculative decoding only. A model that
can't call tools still gets the quality suites that don't need tools.

---

## Stage 0 - Harness and pre-checks (short)

- **Tool-call support per model.** Send one request with `tools` to each
  candidate through Ollama's `/api/chat`. Prediction: all Qwen and llama3.1
  accept it; gemma3 is rejected ("does not support tools") because its
  template has no tools section; gemma4 is unknown.
- **vLLM tool calls.** Start with `--enable-auto-tool-choice
  --tool-call-parser hermes` (the Qwen2.5 format) and check that one call
  parses. Without the parser vLLM returns the call as plain text.
- **Thinking models.** qwen3 and qwen3.5 think by default; Phase 6 saw a
  120-token budget spent entirely on thinking, with an empty answer. Pin
  `think` per request and verify that `think: false` really yields no
  thinking tokens.
- **Pinned settings for every run:** temperature 0, seed 42,
  `repeat_penalty` 1.0, `num_ctx` 8192 (covers 50/51 of the agent's tasks),
  `127.0.0.1`, not `localhost`.
- **No downloads.** Confirm every candidate is still in `ollama list` and in
  WSL's HF cache before starting; anything missing is dropped, not pulled.

### Stage 0 result (2026-10-26)

`01_preflight.py --vllm`, results in `phase10-final-benchmark/logs/preflight.json`.
Ollama is now **0.40.11** (Phase 9 ran 0.40.9).

| model | tools accepted | the call |
|---|---|---|
| qwen2.5:1.5b | yes | `mail-categorize`, category **SALES** (wrong) |
| qwen2.5:3b, 3b Q8_0, 7b, 7b Q8_0, 14b | yes | correct |
| qwen3:14b, qwen3.5:4b-bf16 | yes | correct (think off) |
| gemma3:12b | **refused**: "does not support tools" | - |
| gemma4:12b | yes | correct |
| llama3.1:8b | yes | correct |
| vLLM Qwen2.5-1.5B FP16, hermes parser | yes, parsed into `tool_calls` | category **SALES** (wrong) |

Think control works on both thinking models: `think: false` gives 0
thinking characters and a 4-token answer (~150 ms); `think: true` gives 274-415
tokens (5-8 s) for the same correct answer.

What it means for Stage 1:
- **All 11 Ollama candidates and both vLLM models are present.** Nothing was
  dropped.
- **gemma3:12b takes only suites A, B and E.** Its template has no tools
  section, as predicted. gemma4:12b does call tools.
- **The 1.5B picks the wrong category on both backends**, in the same way. Same
  weights at Q4_K_M and FP16, same mistake, so it's the model, not a backend.
  One item proves nothing, but it fits suite D's prediction for the 1.5b.
- vLLM's startup with the tool parser and TunableOp: 52 s.

---

## Stage 1 - Quality

Four suites, all generated by a deterministic script with a seed, so ground
truth is known and nothing was in any model's training data verbatim. Every
answer is checked by code, as in Phase 6. No LLM judge.

| suite | items | what it scores | why |
|---|---|---|---|
| **A. reasoning** | 200 | multi-step arithmetic, unit conversions, date arithmetic, small logic puzzles; `Answer: <x>` line | the Phase 6 set made harder until FP16 7B stops scoring ~100% |
| **B. extraction** | 150 | JSON from orders, tickets and mails against a schema: parses, validates, per-field exact match | the agent's tool-result case |
| **C. tool choice** | 150 | one turn, 6-8 tools: right tool, right arguments, or correctly no call | single-step tool use |
| **D. agent episodes** | 60 | multi-turn runs against a mock mailbox (below) | the real workload, end to end |

Plus a smaller **long-context check** (suite E, 40 items): a fact planted at
the start, middle and end of 4k / 8k / 16k-token prompts, asked at the end.
This one tests the context window, not truncation; prompts stay under
`num_ctx`.

### Suite D: the mock mailbox

A Python tool server that imitates the agent's mail tools: `mail-get-unread`,
`mail-categorize`, `mail-reply`, `mail-escalate`, `mail-archive`,
`request_human_review`. Each episode gets an inbox of 2-5 synthetic emails,
each with a ground-truth category and expected action (answer, escalate, or
deflect an out-of-scope or jailbreak mail). When the inbox is empty,
`mail-get-unread` returns empty.

The system prompt is **written for the lab and modelled on** the agent's
workflow, not copied. The emails are synthetic. No real customer mail goes
into the repo.

Scored per episode:

- every mail categorized correctly and given the right action
- **terminates** after the inbox is empty (the agent's real bug: 38 calls of
  `mail-get-unread` in a row)
- turn count, tokens in and out, and wall-clock time
- loop detection: the same call with the same arguments 3 times stops the
  episode, and that counts as a failure

An episode passes only if all of these hold. Partial credit is reported
separately.

### Statistics

- 95% Wilson intervals on every score. At n=200 a 70% score is +-6 points,
  so the suites are sized to separate neighbours that differ by ~10 points.
- **Paired** comparisons (McNemar) for the questions that matter:
  7B Q4_K_M against 3B Q8_0, Q4_K_M against Q8_0 of the same model, and
  qwen2.5:14b against qwen3:14b. Same items, so the pairing removes item
  difficulty.
- Greedy isn't bit-repeatable on this stack (Phase 9). Run suite D twice per
  configuration and report how often the two runs disagree, as the noise
  floor.

### Predictions

| comparison | prediction |
|---|---|
| suite A, Qwen2.5 Q4_K_M | 1.5b ~35%, 3b ~50%, 7b ~65%, 14b ~75% |
| **7B Q4_K_M vs 3B Q8_0** | 7B wins by 10+ points on A, C and D; Phase 6's question gets a clear answer |
| Q4_K_M vs Q8_0, same model | within 3 points on every suite; no significant McNemar difference |
| qwen3:14b (no think) vs qwen2.5:14b | qwen3 ahead on A by 5-10 points, about equal on B and C |
| qwen3:14b think on vs off | +10 points on A at 5-10x the output tokens |
| suite D pass rate | 1.5b <20%, 3b ~35%, 7b ~65%, 14b ~80% |
| suite D failure mode | non-termination is the most common failure below 7B; wrong category above it |
| suite E | 14b and 7b near 100% to 16k; 1.5b misses middle facts from 8k |
| vLLM 1.5B FP16 vs Ollama 1.5B Q4_K_M | within noise (Q4_K_M is 93% top-1 agreement, but answers rarely change) |

---

## Stage 2 - Cost of the same work

Speed measured on suite D's episodes and a fixed agent-turn shape (2k-token
cached system prompt, a ~400-token tool result, 200 tokens out), not on a
generic prompt.

Each candidate runs in two configurations, interleaved in one session
(between-session drift is ~4%):

| config | what |
|---|---|
| **default** | Ollama as installed: `NUM_PARALLEL` 1, f16 KV |
| **best measured** | what Phases 7-9 found for that model: direct llama-server with flash attention, q8_0 KV where it helps (14b at long context), 0.5b draft for the 14b/7b single stream, vLLM with TunableOp for 1.5B/3B |

Measured:

- **seconds per episode** and per turn, split into TTFT and decode
- **VRAM**, evicted baseline then settled (the Phase 6 method; the
  board-wide number includes the desktop)
- **fan-out:** episodes per minute with 1, 4 and 16 episodes at once. Speculation
  is switched off above 4 (Phase 9: a loss there).

Predictions:

- 14b best config: ~1.5x faster per episode than default. Speculation gives
  1.6-2.0x on tool-result turns and the direct server ~1.19x; prefill is
  unchanged. Turn time ~5.5 s -> ~3.5-4 s.
- At 16 parallel episodes, vLLM 1.5B tuned finishes ~3x more episodes per
  minute than Ollama 1.5B np16, and **wins on episodes per minute only if
  its pass rate is at least a third of the 14b's**.
- The 14b's q8_0 KV changes nothing at 8k context (Phase 9: it wins only
  from ~8k).

---

## Stage 3 - The choice

One table and one chart, each candidate a point:

- x: seconds per episode (best config), y: suite D pass rate, size: VRAM.
- **Pareto front:** the configurations that nothing else beats on quality,
  time and VRAM.
- **"Correct episodes per hour"** (pass rate x episodes per hour) as one
  number that combines quality and speed, for sequential and fan-out use.

The recommendation for the agent names a model, a quant, a backend and its
settings, and compares its time per turn with Phase 7's measured 52.9-55.5 s
for the 32b the agent ran in the log (time only; the 32b's quality isn't
measured).
The findings go into `lab-findings-for-agentic-work`, and the README gets a
final summary section.

Prediction: **qwen2.5:14b or qwen3:14b at Q4_K_M** is the choice for one
agent loop; **7B Q4_K_M** is the knee of the curve (most of the 14b's
quality for half the time); the 1.5B is only worth it under heavy fan-out on
easy tasks.

---

## Out of scope, deliberately

- **Downloading models.** The candidates are what is on disk now
  (2026-10-26). That leaves out the 32B (so Phase 7's "worth 10x?" stays
  open) and a Q4_K_M build of qwen3.5:4b.
- **Public benchmarks** (MMLU, GSM8K, HumanEval): likely in training data,
  and they don't look like the agent's work. The suites here are generated.
- **LLM-as-judge scoring:** every score is checked by code.
- **New optimizations:** this phase uses what Phases 7-9 measured. Anything
  new found here becomes an open item.
- **Fixing the agent's loop:** that is a termination condition in its code.
  Suite D measures how often each model falls into it; it doesn't fix it.
- **Fine-tuning:** Phase 11, which reuses suite B or C as its task.

---

## Files (planned)

| file | what |
|---|---|
| `phase10-final-benchmark/gen_suites.py` | deterministic generator for suites A-C and E; `--check` verifies answers |
| `phase10-final-benchmark/mailbox.py` | the mock mail tool server and episode generator for suite D |
| `phase10-final-benchmark/01_preflight.py` | Stage 0: tool support, think mode, pinned settings |
| `phase10-final-benchmark/02_quality.py` | Stage 1: runs suites A-E per candidate, stores per-item results |
| `phase10-final-benchmark/03_cost.py` | Stage 2: per-episode time, VRAM, fan-out, default vs best config |
| `phase10-final-benchmark/04_report.py` | Stage 3: tables, intervals, McNemar, Pareto front |

Reuses `phase9-optimization/common.py` for servers, streaming clients and the
TunableOp env. Results go to `lab.db`, one row per item per configuration, so
any comparison can be re-cut later.

---

## Risks and unknowns

- **Suite difficulty.** Too easy and everything scores ~100% (Phase 6's
  problem); too hard and the small models score ~0. Pilot 20 items of each
  suite on the 1.5b and 14b first and adjust until the spread is wide.
- **Tool-call formats differ per model and backend.** A model that writes a
  correct call in the wrong format fails at the parser, not at reasoning.
  Record parse failures separately from wrong calls.
- **Runtime.** ~12 configurations x ~600 items, plus 60 episodes x 2 runs.
  At ~2-10 s per item that is roughly 6-10 hours of GPU time, so run it per
  suite in overnight batches, with checkpoints in `lab.db` so a crash
  resumes.
- **Generated tasks reward the generator's style.** Mix templates and vary
  wording so no model wins on format familiarity alone.

---

## Done when

- Every candidate has scores with intervals on suites A-E and a cost row in
  both configurations.
- The four questions at the top have a measured answer, including
  "can't tell" where the intervals overlap.
- `docs/phase10.md` is written, README's Status and findings are updated, and
  the cleanup plan's Stage 2 can run.

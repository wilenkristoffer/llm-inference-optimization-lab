# Phase 11 - LoRA fine-tuning

Technical notes. Can a small model be taught the user's agent task (mailbox
triage, suite D) well enough to beat the bigger models from Phase 10, on this
card (RX 7800 XT gfx1101 16 GB, Ryzen 5 5600X, 32 GB RAM, Windows 11)? And
what works on AMD along the way: training in WSL2, getting the result back
into Ollama, and serving the adapter unmerged on llama-server and vLLM.

Date: 2026-10-28 to 2026-10-29. Training in WSL2: torch 2.14.0+rocm7.2,
transformers 5.21, peft 0.21.2, flash-linear-attention 0.6 (`~/ft-env`).
Serving: Ollama 0.40.12 and its bundled llama-server; vLLM 0.29.0+rocm723 in
WSL2. Sampling as Phase 10: temperature 0, seed 42, penalties 0, `num_ctx`
8192, `think` off. Plan: `phase11-plan.md`. Code: `phase11-lora-finetuning/`.

---

## Result in one paragraph

**A rank-16 LoRA on qwen3.5:4b, trained for 55 minutes, takes suite D from
82% to 100% and puts a 4B model at the top of Phase 10's leaderboard, level
with the 27B teacher it learned from (98%).** On D', 60 episodes with email templates
the model never saw, it goes from 67% to 92% (p = 0.0015), with no loss on
the other suites. The surprise came from the follow-up: **training on a
scripted solver's runs worked at least as well as training on the teacher's**
(D' 59/60 vs 55/60), and it is the only adapter that clearly beats the
cheaper alternative of a better prompt (p = 0.006). So the 27B teacher, and
its two hours of rollouts, were not needed. Rank 4 does as well as rank 16;
rank 64 is significantly worse. Getting the model served took more work
than training it: Ollama no longer accepts adapters, the merge has to undo a
tensor permutation in the GGUF, and vLLM silently ignores the adapter unless
its keys are renamed.

---

## The task and the test sets

The student is **qwen3.5:4b (BF16)**, Phase 10's most accurate model on D at
82%. The teacher is **bonsai-2:27b**, the best model overall (98%) but 1.5x
slower per episode. The task is Phase 10's suite D: a multi-turn mailbox
episode where the model reads, categorises, replies to, escalates and
archives a small inbox through tool calls, scored by code as fully right or
not.

| set | episodes | made by | purpose |
|---|---|---|---|
| train | 800 scripted, 400 teacher rollouts | `01_episodes.py` (seed 20261101), `02_teacher.py` | training data |
| val | 80 | `01_episodes.py` (seed 20261102) | loss during training (sanity check only) |
| D | 60 | Phase 10, unchanged | the leaderboard score |
| D' | 60 | `01_episodes.py` (seed 20261103), new email templates | did it learn the task or the wording? |

D' is the test that matters. The training episodes come from the same
generator as D, and 44 of D's 159 email bodies appear in the training data
(`logs/episodes.json`). D' shares none of them, and none of its prompt
injections either. A model that gains on D but not on D' has learned the
generator's wording.

Every episode is 1.4k-2.4k tokens (median 2.0k), so one episode is one
training sequence without truncation.

**Two sources of training data:**

- **Scripted:** `01_episodes.py` solves each episode with the rules the
  scorer checks, so every run is perfect by construction. Free to make.
- **Teacher:** bonsai-2:27b ran 400 training episodes through Ollama (2.1 h).
  373 were fully right (93%) and kept. Every mistake was on two kinds of
  email: 14 of 156 jailbreak and 16 of 174 large-refund emails.

The loss is on the assistant turns only (tool calls, replies, closing
summary), never on the system prompt, the user turn or tool results. The
episodes are rendered in exactly the template Ollama serves qwen3.5 with
(`qwen35fmt.py`), so the trained model sees the same tokens at training and
serving time.

---

## The bar: the base model and a better prompt

`03_baseline.py`, qwen3.5:4b-bf16, 60 episodes each:

| prompt | D | D' |
|---|---|---|
| Phase 10 prompt | 49/60 (82%) | 40/60 (67%) |
| tuned prompt (explicit loop rule, one worked example) | 49/60 (82%) | 49/60 (82%) |

The tuned prompt changes nothing on D but lifts D' by 15 points (McNemar 1
vs 10, p = 0.012). That is the cheap alternative a fine-tune has to beat:
prompt work costs no training at all. All fine-tunes below are scored with
the Phase 10 prompt, so the comparison with the base model is like for like,
and are also paired against the tuned-prompt base.

The baseline harness also checked the port: rendering the prompt by hand
(`qwen35fmt`, as training does) gives the same correctness as Ollama's own
renderer on every episode.

---

## Training on gfx1101

`00_preflight.py` decided whether the student was trainable at all, before
any data was made (`logs/preflight.json`).

**What works:**

- The HF checkpoint (`Qwen/Qwen3.5-4B`) loads with no missing or unexpected
  keys. Qwen3.5 is a **hybrid**: 24 of 32 layers are Gated DeltaNet linear
  attention, 8 are full attention.
- **flash-linear-attention's Triton kernels run on gfx1101**: 0.5% relative
  error against the torch reference, 3.5x faster (17.7 vs 62.2 ms forward +
  backward at 2k tokens). transformers picks them up automatically when fla
  is importable.
- LoRA on all 10 projection types (q/k/v/o, the linear-attention `in_proj_qkv`,
  `in_proj_z`, `out_proj`, and the MLP) = 30.5M trainable parameters at r16.

**What does not, or not fully:**

- **SDPA has no flash or memory-efficient kernel at head_dim 256 on gfx1101**,
  only the math path, whose memory grows with the square of the sequence.
- `causal_conv1d` is not installed; its torch fallback costs under 10%.
- **Gradient checkpointing silently does nothing** unless `model.train()` is
  called after `get_peft_model`: the wrapper reports training mode but the
  decoder layers stay in eval mode from `from_pretrained`. 4.8 GB of
  activations at 512 tokens instead of 0.13 GB. `ftcommon.prepare()` handles
  it.
- The 248k-token vocabulary makes full logits cost ~2 MB per token, so the
  loss is computed only at labelled positions, in checkpointed chunks.

**Memory and speed** (batch 1, checkpointing on, r16):

| tokens | train tok/s | peak GB |
|---|---|---|
| 1024 | 517 | 9.8 |
| 2048 | 483 | 10.5 |
| 3072 | 449 | 12.2 |
| 4096 | 372 | 14.3 |
| 5120 | out of memory | |

Without checkpointing it runs out of memory at 1k tokens. 4k tokens is the
ceiling; the episodes fit comfortably. A run of 373 episodes x 2 epochs
(739k tokens, 187 optimizer steps) takes 53-55 min at ~520 tok/s and peaks
at 11.5 GB. The PyTorch allocator logs a couple of out-of-memory retries per
run; they are harmless, and every run trained every episode.

One practical cost: the user's PC froze intermittently during training (WSL
compute holding the shared GPU; no driver resets or hardware errors logged).
Training runs belong in time the PC is not otherwise used.

---

## The first fine-tune: t373-r16

LoRA r16, alpha 32, lr 2e-4, 373 teacher episodes x 2 epochs, merged into
the base weights and served by Ollama as `qwen3.5-mail:t373-r16`.

| | D | D' | s/episode |
|---|---|---|---|
| base qwen3.5:4b | 49/60 (82%) | 40/60 (67%) | 13.1 / 12.4 |
| t373-r16 | **60/60 (100%)** | **55/60 (92%)** | 14.6 / 14.2 |
| McNemar vs base | 0 vs 11, p = 0.001 | 3 vs 18, p = 0.0015 | |

It gained on both D and D', so it learned the task, not just the wording.
The five D' misses: three genuine support emails flagged as out of scope,
and two episodes stopped before the inbox was empty.

### On the Phase 10 leaderboard

Through `evaluate.py` (all five suites, D run twice, cost):

| model | A reasoning | B extraction | C tool choice | D episodes | E long context | s/episode | correct/hour |
|---|---|---|---|---|---|---|---|
| **qwen3.5-mail:t373-r16** | 93% | 87% | 97% | **100%** | 100% | 14.3 | 251 |
| bonsai-2:27b (teacher) | 93% | 99% | 100% | 98% | 100% | 19.3 | 184 |
| qwen3.5:4b-bf16 (base) | 92% | 47% | 97% | 82% | 100% | 12.5 | 236 |
| gemma4:e4b (fastest good) | 80% | 100% | 96% | 82% | 100% | 3.5 | 839 |

- **No forgetting.** A, C and E are unchanged against the base.
- **D: 120/120 over both runs**, the new leader. Against the teacher's
  118/120 the difference is not significant (p = 1); the 4B student matches
  the 27B teacher and is 1.35x faster.
- **B, 47% -> 87%, is one habit, not a new skill.** The base model writes an
  order id like "ORD-872620" as the bare number 872620; the fine-tune copies
  ids verbatim, as the mailbox tools taught it to. Order and correction
  items went from 10/38 and 6/37 to 37/38 and 37/37. Ticket priority ("P3"
  written as 3) stays broken (19/38).
- **Cost:** 14% slower per episode than the base, from ~16% more output
  tokens; same 9.6 GB. It is on the Pareto front, but gemma4:e4b still
  finishes 3.3x as many correct episodes per hour.

---

## Data source and rank: the stick test

The plan called for scripted vs teacher data, then a rank sweep (4/8/16/64)
with 3 seeds. That is most of a day of GPU time, so a stick test came first:
one run each, same seed, epochs and learning rate as t373-r16, scored on D'
only (D is at the ceiling for t373-r16 and cannot show a gain). About 1.3 h
per run: train, merge, score.

| run | data | rank | adapter params | D' | vs t373-r16 | vs tuned-prompt base | s/episode | output tokens |
|---|---|---|---|---|---|---|---|---|
| base, Phase 10 prompt | | | | 40/60 | 18 vs 3, p = 0.0015 | | 12.4 | 574 |
| base, tuned prompt | | | | 49/60 | | | 12.5 | |
| t373-r16 | teacher | 16 | 30.5M | 55/60 | | 5 vs 11, p = 0.21 | 14.2 | 666 |
| **s373-r16** | **scripted** | 16 | 30.5M | **59/60** | 1 vs 5, p = 0.22 | **1 vs 11, p = 0.006** | **13.0** | 595 |
| t373-r4 | teacher | 4 | 7.6M | 54/60 | 4 vs 3, p = 1 | 6 vs 11, p = 0.33 | 14.7 | 669 |
| t373-r64 | teacher | 64 | 121.9M | 46/60 | 11 vs 2, **p = 0.02** | 9 vs 6, p = 0.61 | 14.4 | 665 |

The McNemar columns give the episodes only the reference got right vs only
this run got right.

**Scripted data is at least as good as the teacher's.** 59/60 against 55/60
is not significant on its own, but s373-r16 is the only adapter that beats
the tuned prompt with significance, it no longer makes the teacher's
out-of-scope mistake on genuine support emails, and it is faster: the
scripted runs are terser than bonsai's (221k vs 252k trained tokens for the
same 373 episodes), and the student copies that, at 595 output tokens per
episode against 666. On D, s373-r16 scores **60/60** as well, at 13.1 s per
episode against t373-r16's 14.3.

The teacher's out-of-scope habit is the plausible link: three of t373-r16's
five D' misses are that mistake, r4 makes it six times, and the scripted
model never does. A teacher's fully-right runs still carry its style, and
the student learns the style along with the task.

**Rank is not the limit.** r4 matches r16 (p = 1, the same misses) with a
quarter of the parameters. **r64 is significantly worse**, almost entirely
by stopping before the inbox is empty (10 of its 14 misses). The likely
cause is that alpha = 2r with the same learning rate makes the update
larger at higher rank, but that was not tested. It trained fine otherwise
(peak 13.0 GB, 61 min, a lower final training loss than r16).

---

## Getting it served

### Merging into Ollama's GGUF (`06_merge_export.py`)

No llama.cpp conversion: the merge patches Ollama's own bf16 GGUF of the
student, so tokenizer, metadata, the MTP layer and every untouched tensor
stay byte-identical. Only the targeted tensors change, to W + (alpha/r) B A.

Two things had to be found first:

- **The GGUF is not laid out like the HF checkpoint.** llama.cpp's converter
  reorders the value heads of the linear-attention layers: `in_proj_qkv` and
  `in_proj_z` have their rows permuted, `out_proj` its columns. The merge
  does not assume the permutation; it finds it per tensor by matching the HF
  base weight to the GGUF base weight, stops unless they then match bit for
  bit, and applies the same permutation to the delta.
- **Key names differ**: the adapter says `model.layers.N`, the checkpoint
  `model.language_model.layers.N` (the text model inside the VL wrapper).

**bf16 rounding loses 13.9% of the delta's norm.** The LoRA delta is about
1% of each weight's norm, and bf16 has 8 bits of mantissa, so part of each
small update rounds away when added to a large weight. 84% of the targeted
weights changed at all. It does not matter in practice (next section). The
merge takes 5 min on the CPU in WSL.

### Unmerged, on llama-server (`08_serve_adapter.py`)

**Ollama 0.40.12 refuses adapters entirely** ("LoRA adapters are no longer
supported", for safetensors and GGUF alike), so the plan's `ADAPTER` route
is gone. But Ollama serves qwen3.5 through its bundled `llama-server.exe`,
which accepts `--lora`. 08 runs that binary with the exact arguments Ollama
starts it with, and writes the adapter as an f32 GGUF using the merge's
permutations.

Greedy continuations of 320 assistant turns (20 scripted val episodes):

| | identical to merged | first 40 characters identical | load s | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|
| base | 241/320 | 296 | 7.6 | 2,735 | 53.2 |
| merged | | | 6.6 | 2,778 | 53.3 |
| lora (unmerged, f32) | **318/320** | **320** | 5.6 | 2,186 (-22%) | 46.6 (-13%) |

The unmerged adapter, with nothing rounded away, writes the merged model's
text on 318 of 320 turns, and on D and D' it scored 60/60 and 55/60 with no
episode flipping against the merged model. **The 13.9% rounding loss costs
nothing measurable.** Unmerged serving costs 13% of decode speed: merge to
serve one adapter; keep it unmerged only to swap several on one base.

### Unmerged, on vLLM (`09_vllm_adapter.py`)

vLLM 0.29 runs Qwen3.5-4B on gfx1101 from the existing `~/vllm-env` and the
cached HF weights, including the Gated DeltaNet layers.

**The gotcha: vLLM silently ignores the adapter as PEFT saved it.** vLLM
loads the checkpoint as `Qwen3_5ForConditionalGeneration` (the VL wrapper),
whose text layers are named `model.language_model.layers.N`. The adapter's
`model.layers.N` keys match no module, and vLLM drops them **without a
warning**: the first run produced the base model's text, token for token.
09 copies the adapter with the keys renamed.

Same 320 turns, against llama-server:

| vLLM | identical to llama-server same config | first 40 chars | load s (warm) |
|---|---|---|---|
| base | 307/320 | 318 | 50.5 |
| lora (renamed) | 309/320 | 320 | 54.6 |

Two servers with the same weights differ on ~13 turns in 320 (different
arithmetic order flips near-ties); the adapter adds no difference on top, so
it is applied in full and the D/D' results carry over to vLLM. The first
start took 230-320 s (one-time kernel compilation); later starts 47-57 s.
Speed was not measured; see open items.

---

## What worked on AMD and what did not

| | status |
|---|---|
| LoRA training of a hybrid linear-attention model in WSL2 (fla Triton kernels) | works, 3.5x over the torch fallback |
| SDPA flash / mem-efficient attention at head_dim 256 | no kernel; math path only |
| sequence length with r16, checkpointing | 4k tokens max on 16 GB |
| merging into Ollama's GGUF | works, once the value-head permutation is undone |
| Ollama `ADAPTER` | removed upstream (not an AMD issue) |
| llama-server `--lora` (Ollama's bundled HIP build) | works, -13% decode |
| vLLM `enable_lora` | works after renaming adapter keys |
| QLoRA on gemma4:e4b (stretch) | not attempted |

---

## Recommendation for the agent

- **Fine-tuning works for this task, and the data does not need a big
  teacher.** Scripted perfect runs of the task, made by code, trained the
  best adapter. For the user's real agent this means: write down the rules
  of a correct run, generate examples, train for an hour.
- **A better prompt is the first step.** It got the base model from 67% to
  82% on D' for free; a fine-tune on scripted data got it to 98%.
- **Use rank 4-16 and merge for serving.** Higher rank is not better here,
  and unmerged serving costs 13% of decode speed.
- **The speed trade-off is unchanged.** The fine-tuned 4B is the most
  accurate model on the agent's task, at 13-14 s per episode. gemma4:e4b at
  3.5 s and 81% is still the throughput pick, and a fine-tune of it (the
  stretch goal) is the obvious next experiment.
- The real agent's looping was a missing termination condition in its code.
  Training changes how the model stops, but does not replace that fix.

---

## Limitations

- **One seed per configuration.** The plan's 3 seeds were not run. With 60
  episodes, McNemar only resolves differences of several episodes; the
  r4-vs-r16 and scripted-vs-teacher comparisons are signals, not settled.
- **The stick test scored D' only** (and D for s373-r16). Only t373-r16 went
  through `evaluate.py`, so only it is checked for forgetting and on the
  leaderboard. s373-r16 is the better adapter but has not been checked on
  A, B, C and E.
- The rank sweep skipped r8 and changed rank with alpha = 2r at a fixed
  learning rate, so rank and update size moved together.
- vLLM was checked for correctness, not speed.

## Deviations from the plan

- `05_sweep.py` was not written; the stick test ran 04 -> 06 -> 03 by hand.
- `07_evaluate.py` was not needed: Phase 10's `evaluate.py` took the
  fine-tuned tag as it was.
- `lab.db` got no adapter column; fine-tuned runs are told apart by their
  tag name, `qwen3.5-mail:<run>`.
- File 09 is the vLLM check, not the gemma4 QLoRA stretch goal, which was
  not attempted.

## Open items

- `evaluate.py` on s373-r16 (~2 h) to put the scripted adapter on the
  leaderboard; then possibly s373-r4, likely the cheapest good adapter.
- Why r64 stops early: rsLoRA or a lower learning rate at r64 would tell.
- vLLM speed with the adapter, single stream and concurrent. Ollama forces
  one slot for qwen35 models, so vLLM's batching is where an unmerged
  adapter could pay off.
- QLoRA on gemma4:e4b, if bitsandbytes runs on gfx1101.

---

## The code

| file | does |
|---|---|
| `00_preflight.py` | HF checkpoint, fla kernel check, peak VRAM vs sequence length |
| `01_episodes.py` | train/val/D' episodes from fixed seeds, scripted solutions, overlap and length checks |
| `02_teacher.py` | bonsai-2:27b rollouts, keeping fully-right ones |
| `03_baseline.py` | a model on D/D' with either prompt, paired McNemar; `--model` takes a fine-tuned tag |
| `04_lora_train.py` | one LoRA run (WSL, `~/ft-env`) |
| `06_merge_export.py` | merge into Ollama's bf16 GGUF, `ollama create` |
| `08_serve_adapter.py` | base / merged / unmerged on llama-server: agreement, speed, D/D' |
| `09_vllm_adapter.py` | the adapter on vLLM: key rename, agreement with llama-server |
| `ftcommon.py`, `qwen35fmt.py` | model setup and chunked loss; Ollama's qwen3.5 template |
| `setup_ft_env.sh`, `ft-env.lock.txt` | the training venv |

Adapters (`runs/`) and the regenerated episodes (`data/`) are not committed;
`logs/train/<run>.json` and `logs/episodes.json` record how to remake them.

# Phase 11 plan - LoRA fine-tuning on the agent episodes

Status: **planned, not started.** This is a plan, not results. Numbers below
are estimates to be checked, not measurements.

Revised 2026-10-28 after Phase 10. The first draft (2026-09-28) targeted
qwen2.5 1.5B on a new extraction or tool-call dataset. Phase 10 changed both:
the qwen2.5 models are no longer the ones worth improving, and suite D (the
mailbox agent episodes) is a better task than anything the draft proposed.

Phases 1-10 are about running models. This phase trains one: a LoRA adapter
that teaches a fast model the multi-turn mailbox work, then treats the adapter
as an inference problem like everything else in the lab - what it costs to
serve and on which backend.

---

## What LoRA is, briefly

Full fine-tuning updates every weight and needs optimizer state for all of
them - for a 4B model in bf16 with Adam that is roughly 8 GB weights + 8 GB
gradients + 32 GB optimizer state before activations. It does not fit.

LoRA freezes the base weights and learns a low-rank update for selected
matrices: `W' = W + (alpha / r) * B @ A`, where `A` is `r x d` and `B` is
`d x r` with `r` typically 8-64. Only `A` and `B` are trained, so gradients
and optimizer state shrink to tens of MB. The adapter is a small separate file
that can be merged into the weights or loaded alongside them at serve time.

QLoRA does the same over a base model quantized to 4-bit (NF4), so the frozen
weights take a quarter of the memory. That is what would make an 8B-class
model trainable on a 16 GB card.

Distillation, used here for the training data: a large model that already does
the task well (the teacher) generates the examples, and the small model (the
student) is trained to copy them.

---

## Where Phase 10 left things

Suite D, fully-right episodes, both runs pooled (120):

| model | D | s/episode | correct episodes/hour | notes |
|---|---|---|---|---|
| bonsai-2:27b | **98%** | 19.3 | 184 | Qwen3.5 27B, ternary; one slot only on Ollama |
| qwen3.5:4b-bf16 | 82% | 12.5 | 236 | one slot only on Ollama |
| gemma4:e4b | 81% | 3.5 | **839** | ships a bundled draft model |
| nemotron-3.5-bolt:e4b | 72% | 4.9 | 528 | NVIDIA's fine-tune of gemma4:e4b - *lower* than its base |

The gap is accuracy vs speed: the only near-perfect model is 5.5x slower per
episode than the fastest good one. Closing that gap with a LoRA is the point
of this phase. The nemotron row is a warning: a vendor fine-tune of the same
base made it worse at this task.

---

## Question

1. Can a LoRA on this hardware take a small model from ~81% to near the
   teacher's 98% on D, beating what careful prompting of the same model gets?
2. Does it hold on episodes worded differently from the training data, and
   does it cost anything on the other suites (A, B, C, E)?
3. What does serving the adapter cost - merged vs unmerged, on Ollama and on
   whatever else runs the architecture on this card?

If a tuned prompt gets the base model to the same score, the fine-tune taught
nothing, and that is a valid result.

---

## Models

| role | model | why |
|---|---|---|
| student | **qwen3.5:4b** (bf16) | 82% on D, so there is room to grow. A bf16 LoRA should fit in 16 GB with gradient checkpointing. It is the same family as the teacher, so the teacher's outputs are in a format it already uses. |
| teacher | **bonsai-2:27b** | 98% on D, 0 flips between runs, already set up in Ollama |
| stretch student | **gemma4:e4b** | the throughput pick, and the most valuable model to improve. ~8B raw parameters, so bf16 LoRA likely does not fit; it needs QLoRA, so it depends on bitsandbytes working on gfx1101 |

Ollama holds GGUF files; training needs the Hugging Face weights. Step 0 checks
the student's HF checkpoint (license, size, and whether `transformers` / `peft`
support its architecture) before anything else is built. If qwen3.5's
architecture needs kernels with no ROCm build (for example custom attention
layers), stop and move the plan to gemma4:e4b or another student.

The qwen2.5 models are not used.

---

## Task and data

The task is suite D as it stands: fetch, categorize, reply or escalate,
archive, repeat until the inbox is empty, then stop. `mailbox.py` already has
the generator (`episodes(seed, n)`), the simulator (`Mailbox`) and the scorer
(`run_episode`).

**Test sets (never trained on):**
- **D** - the 60 Phase 10 episodes, seed 20261027, unchanged. The fine-tuned
  model is compared to the same model's base results paired per episode
  (McNemar on the 60 distinct episodes, not the pooled 120).
- **D'** - a new set of episodes from *new* email templates, names and
  products, written for this phase. Training and D come from the same
  generator, so a model could learn the generator's wording instead of the
  task. D' catches that.
- **A, B, C, E** - re-run through `evaluate.py` to catch forgetting.

**Training data**, from `mailbox.episodes()` with seeds that are not D's:
- **Scripted** - a rule-based solver that already knows each email's correct
  category and action writes the perfect sequence of tool calls. It is
  instant and always right, but its replies are templated.
- **Teacher** - bonsai-2:27b runs the same episodes, and only the episodes it
  gets fully right are kept. At ~19 s per episode, 600 episodes take about
  3 hours.

The two data sources are compared at one rank and seed before the sweep. It
is a cheap test of whether teacher-written text beats scripted text.

**Loss on assistant turns only.** System prompt, user turn and tool results
are masked out of the loss. Otherwise the model also learns to predict the
mailbox's own text. Check that the student's chat template marks where the
assistant's turns are, so the masking can find them; write the masking by
hand if it does not.

**The chat template must match serving.** Train with the same template, tool
format and `think: false` rendering that Ollama uses for the model at serving
time, or the fine-tune learns a format it never sees (Phase 2; Phase 10's
`ollama-raw` backend is the reference rendering).

---

## Environment

- WSL2, reusing the ROCm PyTorch setup from Phase 3.
- **A separate venv from vLLM.** Installing `transformers` / `peft` / `trl`
  can pull a different `torch` and silently replace the ROCm wheel. Pin torch
  with a constraints file and check `torch.version.hip` after every install.
- Libraries: `transformers`, `peft`, `trl`, `datasets`; `bitsandbytes` only
  for the QLoRA step. This phase breaks the standard-library-only rule, as
  vLLM already did - it lives on the WSL side.
- Attention via PyTorch SDPA; do not count on flash-attention on gfx1101.
- Teacher rollouts and all evaluation run on the Windows side through
  Ollama, reusing Phase 10's harness. Only training and merging run in WSL.
- Ollama models live on F: (`OLLAMA_MODELS`); the merged GGUF goes there via
  `ollama create`.

---

## Memory estimates (to verify in step 0)

| setup | frozen weights | trainable | expected fit on 16 GB |
|---|---|---|---|
| qwen3.5 4B, LoRA, bf16 | ~8-9 GB | tens of MB | yes with gradient checkpointing; activations decide |
| gemma4 e4b, LoRA, bf16 | ~15-16 GB | tens of MB | no |
| gemma4 e4b, QLoRA, NF4 | ~5 GB | tens of MB | yes, if bitsandbytes works |

Episodes are long multi-turn sequences (system prompt + tool schemas + several
emails), so activations dominate. Measure episode token lengths first; if the
longest do not fit, train on shorter episodes (fewer emails) instead of
truncating, since a truncated episode never shows the stop at the end.
Use the per-process view, not `gpu_vram_used_mb`, which is board-wide.

---

## Files (planned, in `phase11-lora-finetuning/`)

| file | teaches |
|---|---|
| `00_preflight.py` | HF checkpoint for the student, architecture support, one forward/backward step, peak VRAM vs sequence length, `torch.version.hip` |
| `01_episodes.py` | training/validation episodes from new seeds; the D' generator; scripted gold trajectories; token-length histogram |
| `02_teacher.py` | bonsai-2:27b rollouts on the training episodes, keeping only fully-right ones |
| `03_baseline.py` | the student on D and D' with the Phase 10 prompt and with a tuned prompt (explicit loop rule, one worked example) - the bar to beat |
| `04_lora_train.py` | LoRA on the student: loss on assistant turns only, peak VRAM, train tokens/s, loss curve |
| `05_sweep.py` | scripted vs teacher data; then r = 4 / 8 / 16 / 64 vs D score vs adapter size vs train time; 3 seeds at the chosen rank |
| `06_merge_export.py` | merge into base weights, convert to GGUF with llama.cpp, `ollama create`; also the unmerged GGUF adapter for Ollama's `ADAPTER` |
| `07_evaluate.py` | `evaluate.py` on the fine-tuned tag (full A-E, D repeat, cost) plus D', so the result goes onto the Phase 10 leaderboard |
| `08_serve_adapter.py` | decode tok/s and s/episode: base vs merged vs unmerged adapter; vLLM `--enable-lora` if vLLM supports the architecture on ROCm |
| `09_qlora_gemma4.py` | stretch: QLoRA on gemma4:e4b, if bitsandbytes runs on gfx1101 |

Results go into `lab.db`. The runs table needs an adapter column
(name/rank/data source, NULL for base) so adapter runs are not confused with
base runs.

---

## Risks and unknowns

- **HF support for the newer architectures.** qwen3.5 (Ollama architecture
  `qwen35`) and gemma4 are newer than anything trained or served outside Ollama
  in this lab so far. Step 0 decides whether the student is viable.
- **Converting back to GGUF.** llama.cpp's converter has to support the
  architecture, and the template in the `ollama create` Modelfile must match
  the original tag's (including its sampling defaults, e.g. qwen3.5's
  presence_penalty 1.5), or the comparison is not like for like.
- **Ollama `ADAPTER`** support for these architectures is unverified. Merging
  and serving a plain GGUF is the fallback.
- **vLLM on these architectures, on ROCm**, is unverified; Phase 10 only ran
  qwen2.5 on vLLM. If it does not run, the serving comparison is Ollama only.
- **Ollama runs qwen35 models in one slot only** (Phase 10), so a fine-tuned
  qwen3.5 gains accuracy but not concurrency on Ollama.
- **bitsandbytes on ROCm / gfx1101** may not work, or may need a source build.
  If it fails, the gemma4 stretch goal is dropped.
- **Overfitting the generator.** D' is the check; a model that gains on D and
  not on D' learned the wording, not the task.
- **Forgetting.** A narrow fine-tune can cost the other suites. A-E through
  `evaluate.py` is the check.

Not a goal: fixing the user's real agent's looping. That was a missing
termination condition in code. Here the model really does stop early or loop,
which training can change, but the real agent still needs its stop condition.

---

## Done when

- The student's base score (Phase 10 prompt and tuned prompt) and fine-tuned
  score on D and D', paired per episode, with spread across 3 seeds.
- Scripted vs teacher data comparison, and the rank sweep table.
- The fine-tuned model on the `evaluate.py` leaderboard with A-E, cost and
  correct episodes per hour, next to its base and the teacher.
- Serving overhead table for merged vs unmerged, on every backend that runs it.
- A write-up of what worked on AMD and what did not, in the same spirit as
  `docs/phase3-vllm-setup.md`.

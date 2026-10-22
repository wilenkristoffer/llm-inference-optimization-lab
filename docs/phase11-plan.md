# Phase 11 plan - LoRA / QLoRA fine-tuning

Status: **planned, not started.** This is a plan, not results. Numbers below
are estimates to be checked, not measurements.

Phases 1-10 are about running models. This phase trains one: a LoRA adapter on
a small Qwen2.5 model, then treats the adapter as an inference problem like
everything else in the lab - what it costs to serve, on which backend, and
under concurrency.

It comes after Phase 10 so the brief's phases are completed in order, and so it
can reuse Phase 6's quality evaluation and Phase 9's concurrency work.

---

## What LoRA is, briefly

Full fine-tuning updates every weight and needs optimizer state for all of
them - for a 1.5B model in bf16 with Adam that is roughly 3 GB weights + 3 GB
gradients + 12 GB optimizer state before activations. It does not fit.

LoRA freezes the base weights and learns a low-rank update for selected
matrices: `W' = W + (alpha / r) * B @ A`, where `A` is `r x d` and `B` is
`d x r` with `r` typically 8-64. Only `A` and `B` are trained, so gradients
and optimizer state shrink to a few MB. The adapter is a small separate file
that can be merged into the weights or loaded alongside them at serve time.

QLoRA does the same over a base model quantized to 4-bit (NF4), so the frozen
weights take a quarter of the memory. That is what makes a 7B trainable on a
16 GB card.

---

## Question

1. Can a LoRA fine-tune on this hardware make a small model measurably better
   at one narrow task than good prompting of the same model?
2. What does serving the adapter cost - merged vs unmerged, Ollama vs vLLM,
   one adapter vs several under concurrency?

Question 1 needs a baseline. If careful prompting gets the base model to the
same score, the fine-tune taught nothing, and that is a valid result.

---

## Task

Pick one narrow task with **machine-checkable** answers (the same requirement
as Phase 6's quality probes). Candidates:

- **Structured extraction to JSON** against a fixed schema. Scored by: parses,
  validates against schema, per-field exact match.
- **A specific tool-call format** as an agent would emit it. Scored by: valid
  call, correct tool, correct arguments.

Generate the dataset with a deterministic script so ground truth is known and
the test split is guaranteed unseen. Roughly 1-5k training examples,
a few hundred held out.

Not a goal: fixing the agent's looping behaviour. That was a missing
termination condition in code; training cannot substitute for it.

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

---

## Memory estimates (to verify)

| setup | frozen weights | trainable | expected fit on 16 GB |
|---|---|---|---|
| 1.5B, LoRA, bf16 | ~3 GB | a few MB | comfortable |
| 3B, LoRA, bf16 | ~6 GB | a few MB | yes, activations decide |
| 7B, LoRA, bf16 | ~15 GB | a few MB | no |
| 7B, QLoRA, NF4 | ~4-5 GB | a few MB | yes, if bitsandbytes works |

Activations, not weights or optimizer, will dominate at longer sequence
lengths. Gradient checkpointing trades compute for that memory; measure both.
Use the per-process view, not `gpu_vram_used_mb`, which is board-wide.

---

## Files (planned)

| file | teaches |
|---|---|
| `01_dataset.py` | deterministic dataset generator, train/test split |
| `02_eval.py` | scoring against the test split; baseline the base model with a tuned prompt first |
| `03_lora_train.py` | LoRA on qwen2.5 1.5B Instruct: peak VRAM, train tokens/s, loss curve |
| `04_rank_sweep.py` | r = 4 / 8 / 16 / 64 vs score vs adapter size vs train time |
| `05_merge_export.py` | merge into base weights; export for vLLM (safetensors) and Ollama (GGUF adapter via llama.cpp's converter) |
| `06_serve_adapter.py` | decode tok/s: base vs unmerged adapter vs merged, on both backends |
| `07_multi_lora.py` | several adapters on one vLLM base under concurrency - does it move the batch-4 ceiling? |
| `08_qlora_7b.py` | stretch goal: QLoRA on the 7B, if bitsandbytes runs on gfx1101 |

Results go into `lab.db`. The runs table will need an adapter column
(name/rank, NULL for base) so adapter runs are not confused with base runs.

---

## Risks and unknowns

- **bitsandbytes on ROCm / gfx1101** may not work, or may need a source build.
  If it fails, QLoRA is dropped and the phase stays at 1.5B / 3B bf16 LoRA.
- **vLLM LoRA kernels on ROCm** (`--enable-lora`) are unverified on RDNA3.
- **Ollama `ADAPTER`** support depends on architecture and the GGUF adapter
  conversion; verify with Qwen2.5 before relying on it. Merging and serving a
  plain GGUF is the fallback.
- **Base vs Instruct.** Training on the Instruct model must use its chat
  template, or the fine-tune learns to fight the template (see Phase 2).
- **Overfitting a synthetic dataset.** Keep a second, differently-worded test
  set to check the model learned the task, not the generator's phrasing.

---

## Done when

- Base-model baseline and fine-tuned score on the same held-out set, with spread
  across seeds.
- Rank sweep table.
- Serving overhead table for merged vs unmerged on both backends.
- A write-up of what worked on AMD and what did not, in the same spirit as
  `docs/phase3-vllm-setup.md`.

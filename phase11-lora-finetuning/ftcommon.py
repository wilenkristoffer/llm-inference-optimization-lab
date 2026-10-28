"""
Phase 11: pieces shared by the WSL-side training files.

Two memory facts measured in 00_preflight.py shape this:

1. Gradient checkpointing only runs on modules in training mode. A model from
   from_pretrained() comes back in eval mode, and after get_peft_model() the
   wrapper can report training=True while the decoder layers underneath are
   still in eval - so checkpointing silently does nothing (4.8 GB of
   activations at 512 tokens instead of 0.13 GB). prepare() calls
   model.train() on everything, last.

2. Qwen3.5's vocabulary is 248k. Logits for every position cost ~2 MB per
   token once the fp32 upcast and its gradient are counted - 8 GB at 4k
   tokens, more than the activations of the whole trunk. lm_loss() never
   builds the full logits: it takes only the positions that have a label,
   and runs the head + cross-entropy in chunks that are recomputed in the
   backward pass.
"""

import torch
from torch.utils.checkpoint import checkpoint

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# LoRA targets by layer type. in_proj_a / in_proj_b (the per-head decay and
# beta gates of the linear-attention layers) are tiny and left frozen.
TARGETS = {
    "full_attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "linear_attention": ["in_proj_qkv", "in_proj_z", "out_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}
IGNORE = -100


def load_model():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda",
        output_loading_info=True)


def prepare(model, rank, alpha=None, dropout=0.0, grad_ckpt=True):
    """Wrap in LoRA, set up checkpointing, and put every module in train mode."""
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=dropout,
        target_modules=[n for names in TARGETS.values() for n in names], task_type="CAUSAL_LM"))
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    return model


def _chunk_loss(h, weight, y):
    logits = torch.nn.functional.linear(h, weight).float()
    return torch.nn.functional.cross_entropy(logits, y, reduction="sum")


def lm_loss(model, input_ids, labels, chunk=256):
    """Mean next-token loss over labelled positions, without full logits.

    labels has the same shape as input_ids, IGNORE where no loss is wanted
    (system prompt, user turn, tool results). Shifted here, as in HF.
    """
    inner = model.get_base_model()
    hidden = inner.model(input_ids=input_ids).last_hidden_state
    y = labels[:, 1:]
    keep = y != IGNORE
    h = hidden[:, :-1][keep]
    y = y[keep]
    weight = inner.lm_head.weight
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, y.numel(), chunk):
        total = total + checkpoint(_chunk_loss, h[i:i + chunk], weight, y[i:i + chunk],
                                   use_reentrant=False)
    return total / max(y.numel(), 1)

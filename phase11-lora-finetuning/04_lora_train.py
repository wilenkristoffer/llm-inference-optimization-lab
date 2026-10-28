"""
Phase 11, file 4: LoRA on the student, one run.

Trains qwen3.5 4B (bf16, frozen) with LoRA adapters on finished mailbox
episodes. One episode is one sequence (1.4k-2.4k tokens, see
logs/episodes.json), rendered by qwen35fmt.render_train in the format Ollama
serves; the loss is on the assistant turns only (tool calls, replies and the
closing summary, each with its end tag), never on the system prompt, the
user turn or the mailbox's tool results.

    --data scripted   01_episodes.py's solver runs (data/train.jsonl)
    --data teacher    bonsai-2:27b's fully-right runs (data/teacher_train.jsonl)

Validation is the scripted val set for every run, so runs share one yardstick:
loss and next-token accuracy on its assistant tokens. It is a sanity check,
not the result - the result is D and D' through Ollama (07_evaluate.py).

Batch 1 with gradient accumulation, AdamW, linear warmup then cosine decay,
gradient clipping. The adapter goes to runs/<name>/ (not committed), the loss
curve and settings to logs/train/<name>.json.

Runs in WSL, in ~/ft-env. Prints its estimated time before the first step.

Usage:
    python 04_lora_train.py --data scripted --n 400 --epochs 2 --name s400-r16
    python 04_lora_train.py --data teacher --n 400 --epochs 2 --name t400-r16
    python 04_lora_train.py --data scripted --n 8 --epochs 1 --name smoke --val 4
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase10-final-benchmark"))
import mailbox as mb  # noqa: E402
import ftcommon  # noqa: E402

DATA = os.path.join(HERE, "data")
MAX_TOKENS = 4096        # 00_preflight: 5k runs out of memory at r16
TOK_PER_S = 450          # 00_preflight at 2k tokens, r16, checkpointing on
FILES = {"scripted": "train.jsonl", "teacher": "teacher_train.jsonl"}


def load(path, n, only_correct):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if only_correct and not r["correct"]:
                continue
            rows.append(r)
            if n and len(rows) == n:
                break
    return rows


def encode_all(tok, episodes):
    out, skipped = [], 0
    for ep in episodes:
        msgs = [{"role": "system", "content": mb.SYSTEM},
                {"role": "user", "content": mb.USER}] + ep["transcript"]
        ids, labels = ftcommon.encode(tok, msgs, mb.TOOLS)
        if len(ids) > MAX_TOKENS:
            skipped += 1  # never truncate: a cut episode never shows the stop
            continue
        out.append((ep["id"], ids, labels))
    return out, skipped


def lr_at(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def evaluate(model, val):
    model.eval()
    loss = correct = n = 0
    for _, ids, labels in val:
        l, c, k = ftcommon.lm_eval(model, torch.tensor([ids], device="cuda"),
                                   torch.tensor([labels], device="cuda"))
        loss, correct, n = loss + l, correct + c, n + k
    model.train()  # eval() switched checkpointing off; see ftcommon
    return {"val_loss": round(loss / n, 4), "val_acc": round(correct / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=FILES, required=True)
    ap.add_argument("--n", type=int, default=400, help="training episodes (0 = all)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, help="default 2 x rank")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=4, help="episodes per optimizer step")
    ap.add_argument("--warmup", type=float, default=0.05, help="share of optimizer steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=int, default=80, help="scripted val episodes")
    ap.add_argument("--evals", type=int, default=4, help="val passes over the run")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    assert torch.version.hip, "not a ROCm torch - ft-env was broken by an install"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ftcommon.MODEL, revision=ftcommon.REVISION)
    train, skipped = encode_all(tok, load(os.path.join(DATA, FILES[args.data]), args.n,
                                          only_correct=args.data == "teacher"))
    val, _ = encode_all(tok, load(os.path.join(DATA, "val.jsonl"), args.val, False))
    tokens = sum(len(ids) for _, ids, _ in train)
    targets = sum(sum(x != ftcommon.IGNORE for x in lab) for _, _, lab in train)
    steps = args.epochs * len(train)
    opt_steps = math.ceil(steps / args.accum)
    est = args.epochs * tokens / TOK_PER_S + (args.evals + 1) * sum(len(i) for _, i, _ in val) / 1500
    print(f"{len(train)} episodes ({skipped} over {MAX_TOKENS} skipped), {tokens} tokens, "
          f"{targets} targets ({targets / tokens:.0%}); {steps} steps, {opt_steps} updates; "
          f"estimated {est / 60:.0f} min", flush=True)

    model, _ = ftcommon.load_model()
    model = ftcommon.prepare(model, args.rank, args.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, round(args.warmup * opt_steps))
    eval_at = {round(opt_steps * (i + 1) / args.evals) for i in range(args.evals)}

    log = {"name": args.name, "args": vars(args), "episodes": len(train), "skipped": skipped,
           "tokens": tokens, "targets": targets, "estimated_min": round(est / 60),
           "trainable_m": round(sum(p.numel() for p in params) / 1e6, 2),
           "curve": [], "evals": [{"update": 0, **evaluate(model, val)}]}
    print(log["evals"][0], flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step = update = 0
    window = {"loss": 0.0, "n": 0, "tokens": 0, "t": time.perf_counter()}
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        for k in order:
            _, ids, labels = train[k]
            loss = ftcommon.lm_loss(model, torch.tensor([ids], device="cuda"),
                                    torch.tensor([labels], device="cuda"))
            (loss / args.accum).backward()
            window["loss"] += loss.item()
            window["n"] += 1
            window["tokens"] += len(ids)
            step += 1
            if step % args.accum and step != steps:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for g in opt.param_groups:
                g["lr"] = lr_at(update, opt_steps, warmup, args.lr)
            opt.step()
            opt.zero_grad(set_to_none=True)
            update += 1
            if update % 10 == 0 or update == opt_steps:
                dt = time.perf_counter() - window["t"]
                row = {"update": update, "epoch": epoch, "loss": round(window["loss"] / window["n"], 4),
                       "lr": round(opt.param_groups[0]["lr"], 7),
                       "tok_s": round(window["tokens"] / dt),
                       "elapsed_min": round((time.perf_counter() - t0) / 60, 1)}
                log["curve"].append(row)
                print(row, flush=True)
                window = {"loss": 0.0, "n": 0, "tokens": 0, "t": time.perf_counter()}
            if update in eval_at:
                te = time.perf_counter()
                log["evals"].append({"update": update, **evaluate(model, val)})
                print(log["evals"][-1], flush=True)
                window["t"] += time.perf_counter() - te  # eval time is not training speed

    log["train_min"] = round((time.perf_counter() - t0) / 60, 1)
    log["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    out_dir = os.path.join(HERE, "runs", args.name)
    model.save_pretrained(out_dir)
    log["adapter_mb"] = round(sum(os.path.getsize(os.path.join(out_dir, f))
                                  for f in os.listdir(out_dir)) / 1e6, 1)
    os.makedirs(os.path.join(HERE, "logs", "train"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "train", f"{args.name}.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(log, f, indent=1)
    print(f"done in {log['train_min']} min, peak {log['peak_gb']} GB, "
          f"adapter {log['adapter_mb']} MB -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

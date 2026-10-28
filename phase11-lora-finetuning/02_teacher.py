"""
Phase 11, file 2: the teacher's runs on the training episodes.

bonsai-2:27b (98% on D in Phase 10) works through each train and val episode
from 01_episodes.py, with Phase 10's exact settings: Ollama /api/chat, think
off, temperature 0, seed 42, 512-token budget per turn. mailbox.run_episode
scores each run; only fully-right episodes become teacher training data, and
the ones it gets wrong are kept too, marked, to see which kinds it misses.

Results are appended one episode per line, so an interrupted run resumes
where it stopped.

Writes:
    data/teacher_train.jsonl, data/teacher_val.jsonl
    logs/teacher.json        correct share, failure reasons, time per episode

Usage (Windows side, Ollama running):
    python 02_teacher.py --limit 10        # pilot: the first 10 train episodes
    python 02_teacher.py                   # everything, resuming
    python 02_teacher.py --report
"""

import argparse
import collections
import importlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase10-final-benchmark"))
import common  # noqa: E402
import mailbox as mb  # noqa: E402
quality = importlib.import_module("02_quality")

TEACHER = "bonsai-2:27b"
DATA = os.path.join(HERE, "data")
LOG = os.path.join(HERE, "logs", "teacher.json")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_split(backend, split, limit):
    episodes = load_jsonl(os.path.join(DATA, f"{split}.jsonl"))[:limit]
    out_path = os.path.join(DATA, f"teacher_{split}.jsonl")
    done = {r["id"] for r in load_jsonl(out_path)}
    todo = [ep for ep in episodes if ep["id"] not in done]
    print(f"{split}: {len(done)} done, {len(todo)} to run", flush=True)
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for i, ep in enumerate(todo, 1):
            def step(messages, tools):
                return backend.chat(messages, tools, num_predict=quality.NUM_PREDICT["D"])
            t0 = time.perf_counter()
            r = mb.run_episode(step, ep)
            row = {"id": ep["id"], "correct": r["correct"], "score": r["score"],
                   "reason": r["reason"], "turns": r["turns"],
                   "wall_s": round(time.perf_counter() - t0, 1),
                   "emails": ep["emails"], "transcript": r["transcript"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {ep['id']} {'ok ' if r['correct'] else 'BAD'} {row['wall_s']:5.1f}s "
                  f"{'' if r['correct'] else r['reason'][:100]}", flush=True)


def wrong_by_kind(rows):
    """Emails the teacher got wrong / emails seen, per kind. Filtering on fully
    right episodes drops the kinds it misses, so this shows what the teacher
    data is short of (injections, above all)."""
    seen, wrong = collections.Counter(), collections.Counter()
    for r in rows:
        bad = set(re.findall(r"E-\d+", r["reason"])) if not r["correct"] else set()
        for e in r["emails"]:
            seen[e["kind"]] += 1
            wrong[e["kind"]] += e["id"] in bad
    return {k: f"{wrong[k]}/{seen[k]}" for k in sorted(seen)}


def report():
    out = {"teacher": TEACHER, "ollama": common.ollama_version()}
    for split in ("train", "val"):
        rows = load_jsonl(os.path.join(DATA, f"teacher_{split}.jsonl"))
        if not rows:
            continue
        walls = sorted(r["wall_s"] for r in rows)
        out[split] = {
            "episodes": len(rows), "correct": sum(r["correct"] for r in rows),
            "correct_share": round(sum(r["correct"] for r in rows) / len(rows), 3),
            "email_score": round(sum(r["score"] for r in rows) / len(rows), 3),
            "median_s": walls[len(walls) // 2], "hours": round(sum(walls) / 3600, 2),
            "wrong_by_kind": wrong_by_kind(rows),
        }
    print(json.dumps(out, indent=1))
    with open(LOG, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        backend = quality.OllamaBackend(TEACHER, think=False)
        try:
            for split in args.splits:
                run_split(backend, split, args.limit)
        finally:
            backend.close()
    report()


if __name__ == "__main__":
    main()

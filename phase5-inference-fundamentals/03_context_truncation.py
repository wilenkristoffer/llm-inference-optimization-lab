"""
Phase 5, file 3: which end does Ollama discard?

Owed since Phase 1. When a prompt exceeds num_ctx, Ollama does not error - it
silently drops tokens. We ASSUMED oldest-first, which is what makes it
catastrophic in an agent loop: the oldest tokens are the system prompt and the
tool definitions, so the agent loses its instructions while appearing to work.

That assumption drove the recommendation in AGENT_LOOP_INVESTIGATION.md and has
never been verified on this Ollama version. If it truncates from the middle, or
from the newest end, that advice needs correcting.

METHOD

Plant a distinctive code near the START and another near the END of a long
prompt, then ask for both. Whichever the model can still report survived;
whichever it cannot was discarded.

Two things make this trustworthy rather than a vibe:

1. A CONTROL RUN FIRST, at a context large enough to hold everything. If the
   model cannot repeat both codes when they are definitely present, the test
   is broken and the truncated result means nothing.
2. prompt_eval_count, the server's own count of tokens it actually processed.
   If it pins at num_ctx while we sent far more, truncation is confirmed
   independently of whatever the model says.

Uses Ollama's native API because num_ctx is an Ollama option; the OpenAI
interface has no equivalent.

Usage:
    python 03_context_truncation.py
    python 03_context_truncation.py --model qwen2.5:7b --small-ctx 512
"""

import argparse
import json
import random
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA = "http://127.0.0.1:11434"

START_CODE = "ALPHA7"
END_CODE = "OMEGA3"

VOCAB = ("meadow lantern gravel harbour timber ribbon copper meadow thunder "
         "willow anchor marble pebble cavern orchard shadow blanket compass "
         "feather granite hollow juniper kettle lantern mineral nutmeg").split()


def call(path, payload):
    request = urllib.request.Request(
        OLLAMA + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def build_prompt(filler_words, seed=7):
    """
    Codes at both ends, filler between, question last.

    The question sits at the very end on purpose: if truncation dropped the
    NEWEST tokens, the question itself would vanish and the model would respond
    to filler - an unmistakable signal.
    """
    rng = random.Random(seed)
    filler = " ".join(rng.choice(VOCAB) for _ in range(filler_words))
    return (
        f"START_CODE is {START_CODE}. Remember it.\n\n"
        f"{filler}\n\n"
        f"END_CODE is {END_CODE}. Remember it.\n\n"
        "Question: reply with exactly the two codes you were given, "
        "in the form START_CODE=<value> END_CODE=<value>. "
        "If you were not given one of them, write MISSING for that one."
    )


def ask(model, prompt, num_ctx):
    return call("/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx,
                    "repeat_penalty": 1.0, "num_predict": 40},
    })


def report(label, result, sent_tokens, num_ctx):
    text = result["response"].strip().replace("\n", " ")
    processed = result.get("prompt_eval_count")
    saw_start = START_CODE in text
    saw_end = END_CODE in text
    print(f"\n{label}")
    print(f"  num_ctx            {num_ctx}")
    print(f"  tokens we sent     {sent_tokens}")
    print(f"  tokens processed   {processed}"
          f"{'   <- PINNED AT THE LIMIT' if processed and processed >= num_ctx - 1 else ''}")
    print(f"  answer             {text[:110]!r}")
    print(f"  START_CODE visible {'yes' if saw_start else 'NO'}")
    print(f"  END_CODE visible   {'yes' if saw_end else 'NO'}")
    return saw_start, saw_end, processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--small-ctx", type=int, default=512)
    parser.add_argument("--filler-words", type=int, default=900)
    args = parser.parse_args()

    prompt = build_prompt(args.filler_words)

    # How many tokens this really is, measured by the server rather than
    # guessed: send it at a context large enough to hold it all.
    print(f"model {args.model}")
    print("measuring the prompt and running the control...")
    try:
        control = ask(args.model, prompt, 4096)
    except Exception as exc:
        print(f"Ollama not reachable at {OLLAMA}: {exc}")
        print("Start it from the Backends panel in the UI.")
        sys.exit(1)

    sent = control.get("prompt_eval_count")

    print("\n" + "=" * 70)
    print("CONTROL - context large enough to hold everything")
    print("=" * 70)
    c_start, c_end, _ = report("control", control, sent, 4096)

    if not (c_start and c_end):
        print("\nCONTROL FAILED: the model could not repeat both codes even when")
        print("both were present. The test cannot distinguish truncation from")
        print("the model simply being unable to do the task. Try --model")
        print("qwen2.5:7b, or fewer filler words.")
        return

    print("\n  control passed: both codes repeated when both were present.")

    print("\n" + "=" * 70)
    print(f"TRUNCATED - num_ctx {args.small_ctx}, prompt is {sent} tokens")
    print("=" * 70)
    truncated = ask(args.model, prompt, args.small_ctx)
    t_start, t_end, processed = report("truncated", truncated, sent,
                                       args.small_ctx)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if t_start and t_end:
        print("  Both codes survived - no truncation happened. Either num_ctx")
        print("  was not applied, or the prompt fit after all. Raise")
        print("  --filler-words and retry.")
    elif t_end and not t_start:
        print("  OLDEST-FIRST. The start was discarded, the end survived.")
        print("  This is the damaging case for an agent loop: the oldest")
        print("  messages are the system prompt and tool definitions, so the")
        print("  agent silently loses its instructions while still answering.")
        print("  The advice in AGENT_LOOP_INVESTIGATION.md stands: pin the")
        print("  system prompt and trim the MIDDLE of the history instead.")
    elif t_start and not t_end:
        print("  NEWEST-FIRST. The end was discarded - including the question.")
        print("  That contradicts the assumption in")
        print("  AGENT_LOOP_INVESTIGATION.md, which needs correcting.")
    else:
        print("  NEITHER code survived - the middle was kept, or truncation")
        print("  removed both ends. Worth investigating further before")
        print("  trusting any advice about which end is safe.")

    if processed and sent and processed < sent:
        print(f"\n  Independent confirmation: the server processed {processed} of "
              f"{sent} tokens.")
        print(f"  {sent - processed} tokens were dropped with no error and no warning.")


if __name__ == "__main__":
    main()

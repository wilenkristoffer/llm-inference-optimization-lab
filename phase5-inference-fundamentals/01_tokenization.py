"""
Phase 5, file 1: what a token actually is.

Every number in this project is per-token. Tokens/sec, context length, KV cache
size (~28 KB per token), prefill cost, the 29-token chat-template overhead - all
of it. And we have never looked at one.

A token is not a word and not a character. It is an entry in the model's
vocabulary, produced by a byte-pair-encoding tokenizer that learned which
character sequences are common enough to deserve their own id. Common English
words get one token each. Rare words get split. Other languages get split
harder, because the vocabulary was fitted mostly to English.

This uses vLLM's /tokenize endpoint, which exposes the model's REAL tokenizer -
not an approximation. Qwen2.5 shares one tokenizer across every size, so these
counts hold for the 0.5b through the 32b.

Requires vLLM running (use the Backends panel, or see docs/phase3-vllm-setup.md).

Usage:
    python 01_tokenization.py
"""

import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VLLM = "http://127.0.0.1:8000"
MODEL = "qwen2.5-1.5b-instruct"


def post(path, payload):
    request = urllib.request.Request(
        VLLM + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def tokenize(text):
    """
    Returns (ids, strings). add_special_tokens=False so we measure the TEXT,
    not the text plus whatever markers the chat template would add - those we
    already measured separately in Phase 2 (29 tokens).
    """
    r = post("/tokenize", {"model": MODEL, "prompt": text,
                           "add_special_tokens": False,
                           "return_token_strs": True})
    return r["tokens"], r.get("token_strs") or []


SAMPLES = [
    ("plain English", "The cat sat on the mat."),
    ("same, no caps", "the cat sat on the mat."),
    ("rare word", "The antidisestablishmentarianism debate."),
    ("Swedish", "Katten satt pa mattan i det stora huset."),
    ("numbers", "1234567890 and 3.14159265"),
    ("code", "def compute(x): return x * 2 + 1"),
    ("json", '{"model": "qwen2.5", "temperature": 0}'),
    ("repeated spaces", "a     b     c"),
    ("a URL", "https://github.com/vllm-project/vllm"),
]


def main():
    try:
        post("/tokenize", {"model": MODEL, "prompt": "ping"})
    except Exception as exc:
        print(f"vLLM not reachable at {VLLM}: {exc}")
        print("Start it from the Backends panel in the UI, or see")
        print("docs/phase3-vllm-setup.md")
        sys.exit(1)

    # ---------------------------------------------------------------- 1
    print("=" * 76)
    print("A TOKEN IS NOT A WORD")
    print("=" * 76)
    print(f"{'what':<17}{'chars':>6}{'words':>6}{'tokens':>7}{'chars/tok':>10}  text")
    print("-" * 76)
    for label, text in SAMPLES:
        ids, _ = tokenize(text)
        words = len(text.split())
        print(f"{label:<17}{len(text):>6}{words:>6}{len(ids):>7}"
              f"{len(text) / len(ids):>10.1f}  {text[:28]!r}")

    # ---------------------------------------------------------------- 2
    # The space belongs to the token that follows it. This is why you cannot
    # count tokens by splitting on whitespace, and why a leading space changes
    # the tokenization of the word after it.
    print("\n" + "=" * 76)
    print("THE SPACE IS PART OF THE TOKEN")
    print("=" * 76)
    ids, strs = tokenize("The cat sat on the mat.")
    print("'The cat sat on the mat.' ->")
    for i, s in zip(ids, strs):
        # BPE writes a leading space as U+0120 'G with dot above'. Showing both
        # the raw marker and the real string makes the convention obvious.
        real = post("/detokenize", {"model": MODEL, "tokens": [i]})["prompt"]
        print(f"   {i:>6}  {s!r:<12} -> {real!r}")

    # ---------------------------------------------------------------- 3
    # Case matters, because "The" and "the" are different vocabulary entries.
    print("\n" + "=" * 76)
    print("CASE AND POSITION CHANGE THE TOKENS")
    print("=" * 76)
    for text in ["the", " the", "The", " The", "THE"]:
        ids, strs = tokenize(text)
        print(f"   {text!r:<8} -> {ids}  {strs}")

    # ---------------------------------------------------------------- 4
    # The practical consequence: identical meaning, different cost.
    print("\n" + "=" * 76)
    print("THE SAME SENTENCE COSTS DIFFERENT AMOUNTS")
    print("=" * 76)
    pairs = [
        ("English", "The weather is nice today and I am going for a walk."),
        ("Swedish", "Vadret ar fint idag och jag ska ga ut och promenera."),
    ]
    counts = {}
    for label, text in pairs:
        ids, _ = tokenize(text)
        counts[label] = len(ids)
        print(f"   {label:<9} {len(ids):>3} tokens  ({len(text)} chars)")
    if len(counts) == 2:
        a, b = counts.values()
        print(f"\n   Same meaning, {b / a:.2f}x the tokens.")
        print("   The vocabulary was fitted mostly to English, so other")
        print("   languages get split into more, smaller pieces.")

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 76)
    print("WHY THIS MATTERS HERE")
    print("=" * 76)
    print("""
  Every metric in this lab is per-token, so tokenization sets the units:

    KV cache      ~28 KB per token on qwen2.5:1.5b. Text that tokenizes
                  badly fills the context window faster AND costs more VRAM.
    context       num_ctx counts TOKENS, not characters. A 4096-token window
                  holds far less Swedish, code or JSON than English prose.
    prefill       cost scales with input tokens - see file 02.
    throughput    "tokens/sec" is only comparable between runs that tokenize
                  the same way, which is why every benchmark here pins the
                  same prompt and the same model family.

  The agent log analysed in Phase 2 is the practical case: tool results were
  JSON full of IDs and numbers. JSON measured 2.4 characters per token against
  prose's 3.3 above - about 1.4x the tokens for the same characters - and the
  digits inside it cost one token each.
""")


if __name__ == "__main__":
    main()

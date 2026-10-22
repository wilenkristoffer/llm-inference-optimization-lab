"""
Phase 6, file 3: does quantization change answers? The sanity check.

60 questions with machine-checkable answers - arithmetic, factual recall, and
format following - asked once per model, greedy. Every check is mechanical:
a number after "Answer:", an accepted word, JSON that parses and holds the
right values, an exact word count. No judgement calls, so no grader bias.

This is the COARSE measure, deliberately second to file 02. With 60 questions
one answer is 1.7 points, and two models need to differ by roughly 15 points
before the gap beats chance. It exists to catch the level where quantization
stops being a statistic and starts flipping answers people would notice -
not to rank adjacent levels. The summary prints a 95% interval to keep that
honest.

The responses are stored, so a surprising score can be read rather than
trusted.

Usage:
    python 03_task_accuracy.py --models qwen2.5:3b,qwen2.5:3b-instruct-q8_0
    python 03_task_accuracy.py --check-answers      # verify the answer key
    python 03_task_accuracy.py --report
"""

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


storage = _load("storage", "../phase1-ollama-baseline/04_storage.py")
trials = _load("trials", "../phase1-ollama-baseline/03_repeated_trials.py")

OLLAMA = "http://127.0.0.1:11434"
QUESTIONS = os.path.join(os.path.dirname(__file__), "questions.json")
# Penalties pinned to 0 because Qwen3.5's model file defaults presence_penalty
# to 1.5; Qwen2.5's is already 0, so earlier runs are unaffected.
OPTIONS = {"temperature": 0, "seed": 42, "repeat_penalty": 1.0,
           "presence_penalty": 0, "frequency_penalty": 0, "num_ctx": 4096,
           "num_predict": 400}


def ask(model, question):
    """The chat endpoint, so the model sees its normal template - this is a
    test of the model as people use it, unlike file 02's raw prompts."""
    # think=False: Qwen3.5 otherwise spends the whole budget on hidden
    # reasoning and returns an empty answer (measured). No-op on Qwen2.5.
    body = {"model": model, "stream": False, "options": OPTIONS, "think": False,
            "messages": [{"role": "user", "content": question}]}
    request = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            out = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Ollama's loop guard: a generation that degenerates into one repeated
        # token is aborted with HTTP 500. That is the model failing the
        # question, not the server failing - 3B Q2_K did it on 13 of 60.
        detail = exc.read().decode("utf-8", "replace")
        if exc.code == 500 and "repeat limit" in detail:
            return None, None
        raise
    return out["message"]["content"], out.get("eval_count")


# --- checkers: each returns (correct, reason) --------------------------------

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_float(text):
    return float(text.replace(",", ""))


def check_number(response, spec):
    """The number after the last 'Answer:'; failing that, the last number in
    the response. Formatting like '1,000' and '8.00' is accepted."""
    after = NUMBER.findall(response.rsplit("Answer:", 1)[1]) if "Answer:" in response else []
    anywhere = NUMBER.findall(response)
    if not (after or anywhere):
        return False, "no number"
    value = _to_float(after[0] if after else anywhere[-1])
    if math.isclose(value, spec["value"], abs_tol=1e-6):
        return True, None
    return False, f"got {value:g}"


def check_contains_any(response, spec):
    text = response.lower()
    for word in spec["any"]:
        # Word boundaries, so "2" does not match inside "1200".
        if re.search(rf"(?<![\w]){re.escape(word.lower())}(?![\w])", text):
            return True, None
    return False, "none of the accepted answers"


def check_regex_search(response, spec):
    flags = 0 if spec.get("case_sensitive") else re.IGNORECASE
    if re.search(spec["pattern"], response, flags):
        return True, None
    return False, "pattern not found"


def check_regex_full(response, spec):
    """The WHOLE response must match - 'nothing else' means nothing else.
    Surrounding whitespace and quote marks are forgiven."""
    flags = 0 if spec.get("case_sensitive") else re.IGNORECASE
    text = response.strip().strip("\"'`")
    if re.fullmatch(spec["pattern"], text, flags):
        return True, None
    return False, f"response was {text[:40]!r}"


def extract_json(response):
    """First JSON value in the response, tolerating ```json fences."""
    text = re.sub(r"```(?:json)?", "", response)
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                value, _ = json.JSONDecoder().raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    return None


def check_json(response, spec):
    value = extract_json(response)
    if value is None:
        return False, "no parseable JSON"
    if "equals" in spec:
        return (value == spec["equals"],
                None if value == spec["equals"] else f"got {json.dumps(value)[:60]}")
    if not isinstance(value, dict):
        return False, "not an object"
    for key, rule in spec["fields"].items():
        if key not in value:
            return False, f"missing key {key!r}"
        field = value[key]
        if "contains" in rule and rule["contains"] not in str(field).lower():
            return False, f"{key}={field!r}"
        # type() rather than ==, so 36.0 and "36" do not pass for an integer.
        if "equals" in rule and not (field == rule["equals"]
                                     and type(field) is type(rule["equals"])):
            return False, f"{key}={field!r}"
        if "len" in rule and not (isinstance(field, list) and len(field) == rule["len"]):
            return False, f"{key} has {len(field) if isinstance(field, list) else '?'} items"
    return True, None


def check_words(response, spec):
    words = re.findall(r"[A-Za-z0-9'-]+", response)
    n = len(words)
    return (n == spec["n"], None if n == spec["n"] else f"{n} words")


def check_comma_list(response, spec):
    lines = [l for l in response.strip().splitlines() if l.strip()]
    if len(lines) != 1:
        return False, f"{len(lines)} lines"
    items = [i.strip().strip(".").lower() for i in lines[0].split(",")]
    if len(items) != spec["n"]:
        return False, f"{len(items)} items"
    missing = [w for w in spec["must_include"] if w not in items]
    return (not missing, f"missing {missing}" if missing else None)


def check_lowercase_contains(response, spec):
    if any(c.isupper() for c in response):
        return False, "contains uppercase"
    return check_contains_any(response, spec)


def check_bullets(response, spec):
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    bullets = [l for l in lines if re.match(r"^[-*+]\s+\S", l)]
    if len(bullets) != len(lines):
        return False, "text besides the list"
    return (len(bullets) == spec["n"],
            None if len(bullets) == spec["n"] else f"{len(bullets)} bullets")


def check_sentences_start(response, spec):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response.strip()) if s.strip()]
    if len(sentences) != spec["n"]:
        return False, f"{len(sentences)} sentences"
    bad = [s for s in sentences if not s.startswith(spec["word"])]
    return (not bad, f"starts {bad[0][:20]!r}" if bad else None)


CHECKERS = {
    "number": check_number, "contains_any": check_contains_any,
    "regex_search": check_regex_search, "regex_full": check_regex_full,
    "json": check_json, "words": check_words, "comma_list": check_comma_list,
    "lowercase_contains": check_lowercase_contains, "bullets": check_bullets,
    "sentences_start": check_sentences_start,
}


def wilson(k, n, z=1.96):
    """95% interval for a proportion that stays sane near 0% and 100%."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # Clamped: at k=0 rounding leaves the lower bound at -1e-17, printed "-0%".
    return max(0.0, centre - half), min(1.0, centre + half)


def check_answers(data):
    """Recompute every arithmetic answer from its expression, so a typo in the
    key cannot masquerade as a model error. The expressions are this file's
    own, evaluated with no builtins beyond sum and range."""
    bad = 0
    for q in data["questions"]:
        if "expr" in q:
            value = eval(q["expr"], {"__builtins__": {}}, {"sum": sum, "range": range})
            if not math.isclose(value, q["check"]["value"]):
                print(f"  {q['id']}: key says {q['check']['value']}, expression gives {value}")
                bad += 1
    print("answer key OK" if not bad else f"{bad} wrong answers in the key")
    return bad == 0


def report(conn):
    rows = conn.execute("""
        SELECT run_group, model, quantization, category,
               SUM(correct) AS k, COUNT(*) AS n, MIN(timestamp_utc) AS started
        FROM task_results GROUP BY run_group, model, category
        ORDER BY started, model, category
    """).fetchall()
    if not rows:
        print("no task runs recorded yet")
        return
    by_model = {}
    for r in rows:
        key = (r["run_group"], r["model"], r["quantization"])
        by_model.setdefault(key, {})[r["category"]] = (r["k"], r["n"])
    print(f"{'model':<30} {'quant':<8} {'arith':>6} {'fact':>6} {'format':>7} "
          f"{'total':>6}  95% interval")
    print("-" * 82)
    for (group, model, quant), cats in by_model.items():
        k = sum(v[0] for v in cats.values())
        n = sum(v[1] for v in cats.values())
        lo, hi = wilson(k, n)
        cell = lambda c: f"{cats[c][0]}/{cats[c][1]}" if c in cats else "-"
        print(f"{model:<30} {str(quant):<8} {cell('arithmetic'):>6} "
              f"{cell('factual'):>6} {cell('format'):>7} {k:>3}/{n:<3} "
              f"{lo:5.0%}-{hi:4.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen2.5:3b")
    parser.add_argument("--check-answers", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    with open(QUESTIONS, encoding="utf-8") as f:
        data = json.load(f)
    if args.check_answers:
        sys.exit(0 if check_answers(data) else 1)
    conn = storage.connect()
    if args.report:
        report(conn)
        return
    if not check_answers(data):
        sys.exit("fix the answer key first")

    version = trials._ollama("/api/version").get("version")
    run_group = f"tasks|{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        show = trials._ollama("/api/show", {"model": model})
        quant = show.get("details", {}).get("quantization_level")
        digest = next((m.get("digest") for m in trials._ollama("/api/tags")["models"]
                       if m.get("name") == model or m.get("model") == model), None)
        rows, score = [], {}
        for q in data["questions"]:
            prompt = q["q"] + data["suffix"].get(q["category"], "")
            response, n_tokens = ask(model, prompt)
            if response is None:
                ok, reason = False, "aborted by Ollama: repetition loop"
            else:
                ok, reason = CHECKERS[q["check"]["type"]](response, q["check"])
            score.setdefault(q["category"], []).append(ok)
            rows.append({"run_group": run_group, "backend": "ollama",
                         "backend_version": version, "model": model,
                         "model_digest": digest, "quantization": quant,
                         "question_id": q["id"], "category": q["category"],
                         "correct": int(ok), "reason": reason, "response": response,
                         "output_tokens": n_tokens})
            if args.show_failures and not ok:
                print(f"    {q['id']} FAIL ({reason}): {(response or '')[:100]!r}")
        storage.record_rows(conn, "task_results", rows)
        k = sum(r["correct"] for r in rows)
        lo, hi = wilson(k, len(rows))
        per = "  ".join(f"{c} {sum(v)}/{len(v)}" for c, v in score.items())
        print(f"  {model:<32} {str(quant):<7} {k}/{len(rows)}  "
              f"(95% {lo:.0%}-{hi:.0%})   {per}")
    print(f"\nstored in {run_group}")


if __name__ == "__main__":
    main()

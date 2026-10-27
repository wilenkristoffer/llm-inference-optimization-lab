"""
Score a new Ollama model tag end to end, on the same suites and the same
footing as everything Phase 10 measured, and place it on the leaderboard.

Phase 10 showed that the model generation matters more than any serving
setting, so the useful question is "is this new release worth switching
to?". This answers it in one command per tag:

    detect     `ollama show`: architecture, quantization, size, whether it
               takes tools, whether think is a bool or only levels, a bundled
               draft model, Modelfile sampling defaults. Written to
               candidates.json, which every other script here reads.
    preflight  one tool call and the think check from 01_preflight.py
    quality    suites A-E (02_quality.py), run group "eval"
    repeat     suite D again (run group "eval-d-repeat"): is it repeatable?
    cost       20 episodes one at a time, then 8 at NUM_PARALLEL 4 on a side
               Ollama (03_cost.py) -> logs/eval-cost.json
    report     the leaderboard over Phase 10's models and every tag evaluated
               so far, McNemar against the leader, the Pareto front, a chart
               (logs/eval-pareto.svg), and each model's warnings

Every step resumes: rerun the same command after an interruption. Warnings
are the traps Phase 10 fell into, checked automatically: tools refused or
left as text by Ollama's parser, think that cannot be turned off, a bundled
draft model, Ollama forcing one slot, spilling into system RAM, answers that
change under batching or between identical runs, and Modelfile penalties the
agent must override.

--quick runs 10 items per suite into run group "eval-quick" (kept out of the
leaderboard) for a first look in a few minutes.

Usage:
    python evaluate.py qwen3.6:8b                  # must already be pulled
    python evaluate.py qwen3.6:8b --pull           # pull it first
    python evaluate.py phi5:14b --quick
    python evaluate.py --report                    # leaderboard only
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import urllib.request

import common

pf = common._load("p10preflight", "01_preflight.py")
q = common._load("p10quality", "02_quality.py")
cost = common._load("p10cost", "03_cost.py")
rep = common._load("p10report", "04_report.py")

FIRST, REPEAT, QUICK, COST = "eval", "eval-d-repeat", "eval-quick", "eval-cost"
STATE = os.path.join(common.LOG_DIR, "eval.json")
SVG = os.path.join(common.LOG_DIR, "eval-pareto.svg")
COST_EPISODES, FAN = 20, 4
FIT_GB = 14.0               # weights past this leave no room for 8k of KV cache on 16 GB
FLIPS_WARN = 5              # of 60 episodes; gpt-oss flipped 9, the 14b 7


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --- detect ----------------------------------------------------------------------

def pull(tag):
    if tag in common.ollama_tags():
        return
    print(f"pulling {tag}", flush=True)
    subprocess.run(["ollama", "pull", tag], check=True)


def show(tag):
    req = urllib.request.Request(common.OLLAMA + "/api/show", json.dumps({"model": tag}).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def detect(tag):
    """The candidate entry for a tag, and the warnings `ollama show` alone gives."""
    s = show(tag)
    caps, det, info = s.get("capabilities") or [], s.get("details") or {}, s.get("model_info") or {}
    if "completion" not in caps:
        raise SystemExit(f"{tag} is not a chat model (capabilities {caps})")
    size = next(m["size"] for m in json.load(urllib.request.urlopen(
        common.OLLAMA + "/api/tags", timeout=10))["models"] if m["name"] == tag)
    values = (s.get("thinking") or {}).get("values") or []
    thinks = "thinking" in caps
    entry = {"quantization": det.get("quantization_level", "?"),
             "family": info.get("general.architecture") or det.get("family", "?"),
             "thinks": thinks,
             # False when think takes a bool; else the lowest level (gpt-oss: "low")
             "think_off": False if not thinks or False in values else (values[0] if values else False),
             "parameters": det.get("parameter_size"), "size_gb": round(size / 1e9, 1),
             "capabilities": caps,
             "draft": any(l.startswith("DRAFT ") for l in (s.get("modelfile") or "").splitlines()),
             "added": datetime.date.today().isoformat()}
    warn = []
    if "tools" not in caps:
        warn.append("no tool support: suites C and D are skipped, so it cannot run the agent")
    if thinks and entry["think_off"] is not False:
        warn.append(f"thinking cannot be turned off; runs at its lowest level "
                    f"'{entry['think_off']}' with an {q.THINK_PREDICT}-token budget")
    if entry["draft"]:
        warn.append("Modelfile bundles a draft model (Ollama speculative decoding); "
                    "nemotron's gained nothing alone and blocked batching")
    params = dict(re.findall(r"^(\w+)\s+(.+?)\s*$", s.get("parameters") or "", re.M))
    pinned = {k: v for k, v in params.items()
              if k in ("presence_penalty", "frequency_penalty", "repeat_penalty")
              and float(v) not in (0.0, 1.0 if k == "repeat_penalty" else 0.0)}
    if pinned:
        warn.append(f"Modelfile sets {pinned}; pinned to neutral here, so the agent must "
                    "pin it too or it runs a different model")
    if entry["size_gb"] > FIT_GB:
        warn.append(f"{entry['size_gb']} GB of weights: little or no room for KV cache on "
                    "16 GB; expect it to spill")
    extra = [c for c in caps if c in ("vision", "audio")]
    if extra:
        warn.append(f"loads a {'/'.join(extra)} projector, which takes VRAM the agent does not use")
    return entry, warn


def register(tag, entry):
    reg = load_json(common.REGISTRY, {})
    reg[tag] = entry
    save_json(common.REGISTRY, reg)
    common.CANDIDATES.setdefault(tag, (entry["quantization"], entry["family"], entry["thinks"]))
    if entry["thinks"] and entry["think_off"] is not False:
        common.THINK_OFF.setdefault(tag, entry["think_off"])


# --- preflight -------------------------------------------------------------------

def preflight(tag, entry):
    """01_preflight's checks, stored where 02_quality.py looks for them."""
    res = {"installed": True, "digest": common.ollama_tags()[tag],
           "quantization": entry["quantization"], "family": entry["family"],
           "tools": pf.check_tools(tag)}
    if entry["thinks"]:
        res["think"] = pf.check_think(tag)
    path = os.path.join(common.LOG_DIR, "preflight.json")
    data = load_json(path, {"models": {}})
    data["models"][tag] = res
    save_json(path, data)
    common.ollama_chat(tag, [{"role": "user", "content": "x"}], num_predict=1, keep_alive=0)

    warn, t = [], res["tools"]
    if not t["accepted"]:
        warn.append("Ollama refuses tools for it: " + t["error"][:100])
    elif not t.get("n_calls"):
        text = t.get("content", "")
        warn.append("tool call left as text by Ollama's parser (like ministral): " + text[:80]
                    if pf.EXPECTED_CALL[0] in text or "[ARGS]" in text or '"name"' in text
                    else "answered the tool prompt in prose, no call")
    elif not t["correct"]:
        warn.append(f"wrong first tool call: {t.get('name')} {t.get('arguments')}")
    off = res.get("think", {}).get("think_false")
    if off and entry["think_off"] is False:
        if off["thinking_chars"]:
            warn.append(f"think:false still thought ({off['thinking_chars']} chars)")
        if not off["content"]:
            warn.append("think:false gave an empty answer")
    return res, warn


# --- quality, repeat, cost ---------------------------------------------------------

def quality(conn, tag, run_group, suites, limit=None):
    args = argparse.Namespace(backend="ollama", think="off", limit=limit, config="default",
                              run_group=run_group)
    q.run_model(conn, tag, args, list(suites))


def flips(conn, tag):
    rows = conn.execute("SELECT run_group, question_id, correct FROM task_results WHERE model = ? "
                        "AND suite = 'D' AND run_group IN (?, ?)", (tag, FIRST, REPEAT)).fetchall()
    first = {r["question_id"]: r["correct"] for r in rows if r["run_group"] == FIRST}
    return sum(1 for r in rows if r["run_group"] == REPEAT
               and r["question_id"] in first and first[r["question_id"]] != r["correct"])


def cost_units(conn, tag):
    path = os.path.join(common.LOG_DIR, COST + ".json")
    res = load_json(path, {"models": {}})
    res["ollama_version"] = common.ollama_version()
    done = res["models"].setdefault(tag, {})
    args = argparse.Namespace(run_group=COST)
    items = cost.mailbox_items()
    if "default" not in done:
        common.p9.evict_ollama()
        done.update(cost.seq_unit(conn, args, tag, items[:COST_EPISODES]))
        save_json(path, res)
    if f"np{FAN}" not in done:
        done.update(cost.fanout_unit(conn, args, tag, FAN, items[:2 * FAN]))
        save_json(path, res)
    return done


def cost_warnings(conn, tag, units):
    warn = []
    seq, fan = units.get("default", {}), units.get(f"np{FAN}", {})
    shared = seq.get("vram", {}).get("shared_mb", 0)
    if shared > cost.SHARED_LIMIT_MB:
        warn.append(f"spills {shared / 1024:.1f} GB into system RAM at one slot: its timings "
                    "measure paging over PCIe")
    v = fan.get("vram", {})
    if v.get("runner_np") is not None and v["runner_np"] < FAN:
        warn.append(f"Ollama forces -np {v['runner_np']} for architecture "
                    f"{common.CANDIDATES[tag][1]}: NUM_PARALLEL only queues (like qwen3.5)")
    if "skipped" in fan:
        warn.append(f"{FAN} slots of 8192 tokens do not fit on the GPU: no fan-out")
    elif fan:
        seq_ok = {r["question_id"]: r["correct"] for r in conn.execute(
            "SELECT question_id, correct FROM task_results WHERE run_group = ? AND model = ? "
            "AND suite = 'D'", (FIRST, tag))}
        changed = sum(1 for r in conn.execute(
            "SELECT question_id, correct FROM task_results WHERE run_group = ? AND model = ? "
            "AND config = ?", (COST, tag, f"np{FAN}"))
            if r["question_id"] in seq_ok and seq_ok[r["question_id"]] != r["correct"])
        if changed:
            warn.append(f"{changed} of {fan['n']} episodes change answer at NUM_PARALLEL {FAN} "
                        "(batched decode; gemma4 broke this way)")
        if seq and fan["episodes_per_min"] < seq["episodes_per_min"] * 1.2:
            warn.append(f"fan-out buys little: {fan['episodes_per_min']:.1f} episodes/min at "
                        f"np{FAN} vs {seq['episodes_per_min']:.1f} one at a time")
    return warn


# --- report ----------------------------------------------------------------------

def report():
    rep.full(first=rep.FIRST + (FIRST,), repeat=rep.REPEAT + (REPEAT,),
             cost_files=rep.COST_FILES + (COST + ".json",), svg_path=SVG, phase6=False)
    state = load_json(STATE, {})
    if state:
        print("\n## Warnings for evaluated models\n")
        for tag, s in state.items():
            print(f"**{tag}** ({', '.join(s.get('steps', []))})")
            for w in s.get("warnings", []) or ["none"]:
                print(f"- {w}")
            print()


def evaluate(conn, tag, args):
    state = load_json(STATE, {})
    s = state.setdefault(tag, {"steps": [], "warnings": []})

    def step(name, warn=()):
        s["warnings"] = [w for w in s["warnings"] if not w.startswith(name + ": ")]
        s["warnings"] += [f"{name}: {w}" for w in warn]
        if name not in s["steps"]:
            s["steps"].append(name)
        save_json(STATE, state)
        for w in warn:
            print(f"  WARNING {w}", flush=True)

    if args.pull:
        pull(tag)
    if tag not in common.ollama_tags():
        raise SystemExit(f"{tag} is not installed; add --pull to download it")
    print(f"== {tag}", flush=True)
    entry, warn = detect(tag)
    register(tag, entry)
    print(f"  {entry['family']} {entry['parameters']} {entry['quantization']} "
          f"{entry['size_gb']} GB, think_off={entry['think_off']}, caps={entry['capabilities']}")
    step("detect", warn)
    s["preflight"], warn = preflight(tag, entry)
    step("preflight", warn)
    if args.quick:
        quality(conn, tag, QUICK, q.SUITES, limit=10)
        q.report(conn, QUICK)
        step("quick")
        return
    quality(conn, tag, FIRST, q.SUITES)
    step("quality")
    if "tools" in entry["capabilities"]:
        quality(conn, tag, REPEAT, "D")
        n = flips(conn, tag)
        step("repeat", [f"not repeatable at temperature 0: {n} of 60 episodes flip "
                        "between identical runs"] if n > FLIPS_WARN else [])
        units = cost_units(conn, tag)
        step("cost", cost_warnings(conn, tag, units))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", help="Ollama tags, e.g. qwen3.6:8b")
    ap.add_argument("--pull", action="store_true", help="download tags that are not installed")
    ap.add_argument("--quick", action="store_true",
                    help="10 items per suite into run group eval-quick; no repeat or cost")
    ap.add_argument("--report", action="store_true", help="only print the leaderboard")
    args = ap.parse_args()
    q.SUITE_C = q.load_suite("C")
    conn = common.storage.connect()
    if not args.report:
        if not args.tags:
            ap.error("give at least one tag, or --report")
        for tag in args.tags:
            evaluate(conn, tag if ":" in tag else tag + ":latest", args)
    if not args.quick:
        report()


if __name__ == "__main__":
    main()

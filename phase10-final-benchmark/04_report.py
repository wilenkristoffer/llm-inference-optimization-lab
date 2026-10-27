"""
Phase 10, file 4: Stage 3 - the choice.

Reads what Stages 1 and 2 stored (lab.db run groups p10-quality and
p10-d-repeat, logs/p10-cost.json) and prints, as markdown:

    leaderboard   suites A-E per model and backend, Wilson 95% intervals,
                  sorted by suite D (the agent-shaped episodes)
    pairs         exact McNemar on suite D, each shortlisted model against
                  the leader, D pooled over both runs (paired per run and item)
    sizes         the Phase 6 question: qwen2.5 7B Q4_K_M vs 3B Q8_0, per suite
    pareto        every Stage 2 configuration: D pass rate, s/episode, VRAM,
                  correct episodes per hour; the front is what nothing beats
                  on all three at once

and writes the chart (x s/episode, y D pass rate, area VRAM) to
docs/phase10-pareto.svg. No new runs.

Rows where a reasoning model ran without a think field (gemma4's first run,
which thought through its whole budget) are left out whenever the same model
also has think=0 rows.

Usage:
    python 04_report.py
"""

import json
import math
import os

import common

q = common._load("p10quality", "02_quality.py")
cost_mod = common._load("p10cost", "03_cost.py")

# Run groups holding each model's first full run, and its suite D repeat.
# evaluate.py passes these plus its own, so later models join the same table.
FIRST = ("p10-quality",)
REPEAT = ("p10-d-repeat",)
COST_FILES = ("p10-cost.json",)
SVG_PATH = os.path.join(common.HERE, "..", "docs", "phase10-pareto.svg")


def load(conn, first=FIRST, repeat=REPEAT):
    """{(model, backend): {("first" | "repeat", suite, question_id): correct}}.
    Rows are read oldest first, so a later run of the same item wins."""
    groups = first + repeat
    rows = conn.execute(
        "SELECT run_group, model, backend, think, suite, question_id, correct FROM task_results "
        f"WHERE run_group IN ({','.join('?' * len(groups))}) ORDER BY id", groups).fetchall()
    has_off = {(r["model"], r["backend"]) for r in rows if r["think"] == 0}
    out = {}
    for r in rows:
        key = (r["model"], r["backend"])
        if key in has_off and r["think"] is None:
            continue
        run = "first" if r["run_group"] in first else "repeat"
        out.setdefault(key, {})[(run, r["suite"], r["question_id"])] = r["correct"]
    return out


def load_cost(files=COST_FILES):
    """Stage 2 units per model, later files winning per model."""
    models = {}
    for name in files:
        path = os.path.join(common.LOG_DIR, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                models.update(json.load(f)["models"])
    return {"models": models}


def rate(items, suite, groups=("first",)):
    v = [c for (g, s, _), c in items.items() if s == suite and g in groups]
    return (sum(v), len(v)) if v else (0, 0)


def mcnemar(a, b):
    """Exact two-sided McNemar on paired 0/1 results. Returns (a only, b only, p)."""
    keys = a.keys() & b.keys()
    x = sum(1 for k in keys if a[k] and not b[k])
    y = sum(1 for k in keys if b[k] and not a[k])
    n = x + y
    if n == 0:
        return x, y, 1.0
    tail = sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n
    return x, y, min(1.0, 2 * tail)


def pct(k, n):
    return f"{k / n:.0%}" if n else "-"


def ci(k, n):
    lo, hi = q.wilson(k, n)
    return f"{max(lo, 0):.0%}-{min(hi, 1):.0%}"


def leaderboard(data):
    print("## Leaderboard (fully right, 95% interval)\n")
    print("| model | backend | A reasoning | B extraction | C tool choice | D episodes | E long context |")
    print("|---|---|---|---|---|---|---|")
    order = sorted(data, key=lambda k: (-(rate(data[k], "D")[0] / max(rate(data[k], "D")[1], 1)), k))
    for key in order:
        cells = []
        for s in "ABCDE":
            k, n = rate(data[key], s)
            cells.append(f"{pct(k, n)} ({ci(k, n)})" if n else "-")
        print(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    print()


def d_items(items):
    return {k: v for k, v in items.items() if k[1] == "D"}


def pairs(data, shortlist):
    # The repeat is not an independent sample: most models answer each episode
    # the same way both times, so pairing over 120 would count every
    # disagreement twice. McNemar runs on the 60 distinct episodes (first run);
    # the pooled rate and the flip count say how repeatable each model is.
    print("## Suite D against the leader (exact McNemar on the 60 episodes)\n")
    pooled = {m: d_items(data[(m, "ollama")]) for m in shortlist if (m, "ollama") in data}
    first = {m: {k[2]: v for k, v in it.items() if k[0] == "first"} for m, it in pooled.items()}
    ranked = sorted(pooled, key=lambda m: (-sum(pooled[m].values()) / len(pooled[m]), m))
    lead = ranked[0]
    print("| model | D pooled (both runs) | 95% interval | flips between runs | "
          "leader only | model only | p |")
    print("|---|---|---|---|---|---|---|")
    for m in ranked:
        k, n = sum(pooled[m].values()), len(pooled[m])
        flips = sum(1 for (g, _, i), v in pooled[m].items()
                    if g == "repeat" and first[m].get(i) is not None and first[m][i] != v)
        head = f"| {m} | {pct(k, n)} ({k}/{n}) | {ci(k, n)} | {flips} |"
        if m == lead:
            print(head + " - | - | - |")
            continue
        x, y, p = mcnemar(first[lead], first[m])
        print(f"{head} {x} | {y} | {p:.3g} |")
    print()
    return pooled


def sizes(data):
    print("## Phase 6's question: qwen2.5 7B Q4_K_M vs 3B Q8_0\n")
    a, b = data[("qwen2.5:7b", "ollama")], data[("qwen2.5:3b-instruct-q8_0", "ollama")]
    print("| suite | 7B Q4_K_M | 3B Q8_0 | 7B only | 3B only | p |")
    print("|---|---|---|---|---|---|")
    for s in "ABCDE":
        sa = {k: v for k, v in a.items() if k[1] == s and k[0] == "first"}
        sb = {k: v for k, v in b.items() if k[1] == s and k[0] == "first"}
        x, y, p = mcnemar(sa, sb)
        print(f"| {s} | {pct(sum(sa.values()), len(sa))} | {pct(sum(sb.values()), len(sb))} "
              f"| {x} | {y} | {p:.3g} |")
    print()


def configs(cost, data, pooled):
    """One point per Stage 2 configuration that ran one episode at a time."""
    pts = []
    for model, units in cost["models"].items():
        if model not in pooled and model not in cost_mod.VLLM_MODEL:
            continue                    # costed but never quality-scored
        for config, u in units.items():
            if u.get("concurrency") != 1:
                continue
            if config == "default":
                k, n = sum(pooled[model].values()), len(pooled[model])
            elif config == "vllm-tuned":
                k, n = rate(data[(cost_mod.VLLM_MODEL[model], "vllm")], "D")
            else:                       # measured only in Stage 2's 20 episodes
                k, n = u["correct"], u["n"]
            v = u.get("vram", {})
            # Fan-out ran only the first 8 / 32 episodes, so it is reported as
            # episodes per minute, not scored against the 120-episode rate.
            fan = []
            if config in ("default", "vllm-tuned"):
                for c, w in units.items():
                    if w.get("concurrency", 1) == 1:
                        continue
                    if "skipped" in w:
                        fan.append(f"{c}: no fit")
                        continue
                    queued = w["vram"].get("runner_np", w["concurrency"]) < w["concurrency"]
                    fan.append(f"{c}: {w['episodes_per_min']:.1f}" + (" queued" if queued else ""))
            pts.append({"model": model, "config": config, "k": k, "n": n, "pass": k / n,
                        "s_ep": u["s_per_episode"], "s_turn": u["s_per_turn"],
                        "vram_gb": v["dedicated_mb"] / 1024 if "dedicated_mb" in v else None,
                        "per_hour": k / n * 3600 / u["s_per_episode"],
                        "ep_min": u["episodes_per_min"], "fan": fan})
    for p in pts:
        # a configuration that never finishes an episode is fast, not a choice
        p["front"] = p["pass"] > 0 and not any(
            o is not p and o["pass"] >= p["pass"] and o["s_ep"] <= p["s_ep"]
            and (o["vram_gb"] or 0) <= (p["vram_gb"] or 0)
            and (o["pass"] > p["pass"] or o["s_ep"] < p["s_ep"]
                 or (o["vram_gb"] or 0) < (p["vram_gb"] or 0))
            for o in pts)
    return pts


def pareto(pts):
    print("## Cost and the Pareto front (one episode at a time)\n")
    print("Pass rate: suite D pooled over both runs (120) for Ollama defaults, the 60 "
          "quality-run episodes for vLLM, the 20 Stage 2 episodes for spec-0.5b.\n")
    print("| model | config | D pass | n | s/episode | s/turn | VRAM GB | correct/hour | "
          "episodes/min | at once (episodes/min) | front |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(pts, key=lambda p: -p["per_hour"]):
        vram = f"{p['vram_gb']:.1f}" if p["vram_gb"] else "WSL"
        print(f"| {p['model']} | {p['config']} | {p['pass']:.0%} | {p['n']} | {p['s_ep']:.1f} "
              f"| {p['s_turn']:.2f} | {vram} | {p['per_hour']:.0f} | {p['ep_min']:.1f} "
              f"| {', '.join(p['fan']) or '-'} | {'yes' if p['front'] else ''} |")
    print()


def svg(pts, path=SVG_PATH):
    W, H, L, R, T, B = 760, 460, 70, 30, 30, 60
    xs = [p["s_ep"] for p in pts]
    x0, x1 = math.log10(0.3), math.log10(max(xs) * 1.4)

    def X(v):
        return L + (math.log10(v) - x0) / (x1 - x0) * (W - L - R)

    def Y(v):
        return T + (1 - v) * (H - T - B)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           'font-family="sans-serif" font-size="12">',
           '<style>.bg{fill:#fff}.g{stroke:#e2e2e2}.t{fill:#333}'
           '.l{fill:#222;stroke:#fff;stroke-width:4px;paint-order:stroke;stroke-linejoin:round}'
           '.f{fill:#2a6fdb;fill-opacity:.55;stroke:#2a6fdb}'
           '.o{fill:#999;fill-opacity:.35;stroke:#777}'
           '@media (prefers-color-scheme:dark){.bg{fill:#1b1b1b}.t{fill:#ccc}.g{stroke:#3a3a3a}'
           '.l{fill:#eee;stroke:#1b1b1b}}</style>',
           f'<rect class="bg" width="{W}" height="{H}"/>']
    for v in (0, .2, .4, .6, .8, 1):
        out.append(f'<line class="g" x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}"/>'
                   f'<text class="t" x="{L - 8}" y="{Y(v) + 4}" text-anchor="end">{v:.0%}</text>')
    for v in (0.5, 1, 2, 5, 10, 20):
        if x0 <= math.log10(v) <= x1:
            out.append(f'<line class="g" y1="{T}" y2="{H - B}" x1="{X(v)}" x2="{X(v)}"/>'
                       f'<text class="t" x="{X(v)}" y="{H - B + 18}" text-anchor="middle">{v:g}</text>')
    out.append(f'<text class="t" x="{(L + W - R) / 2}" y="{H - 15}" text-anchor="middle">'
               'seconds per episode (log scale, one episode at a time)</text>'
               f'<text class="t" transform="translate(18 {(T + H - B) / 2}) rotate(-90)" '
               'text-anchor="middle">suite D pass rate</text>')
    labels, placed = [], []              # label boxes so close points don't overprint
    for p in sorted(pts, key=lambda p: -(p["vram_gb"] or 0)):
        r = 6 * math.sqrt(p["vram_gb"] or 1)
        label = p["model"] + ("" if p["config"] == "default" else " " + p["config"])
        cx, cy = X(p["s_ep"]), Y(p["pass"])
        w = 6.6 * len(label)
        tx = cx - r - 4 - w if cx > (L + W - R) / 2 else cx + r + 4
        ty = cy + 4
        while any(abs(ty - y) < 14 and tx < x2 and x1 < tx + w for x1, x2, y in placed):
            ty += 14
        placed.append((tx, tx + w, ty))
        out.append(f'<circle class="{"f" if p["front"] else "o"}" cx="{cx:.1f}" '
                   f'cy="{cy:.1f}" r="{r:.1f}"/>')
        labels.append(f'<text class="l" x="{tx:.1f}" y="{ty:.1f}">{label}</text>')
    out += labels
    out.append(f'<text class="t" x="{L}" y="18">Blue: Pareto front (pass rate, time, VRAM). '
               'Circle area: VRAM.</text></svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def full(first=FIRST, repeat=REPEAT, cost_files=COST_FILES, svg_path=SVG_PATH, phase6=True):
    """Every table, then the chart. evaluate.py calls this with its own groups."""
    conn = common.storage.connect()
    data = load(conn, first, repeat)
    cost = load_cost(cost_files)
    shortlist = [m for m in cost["models"] if (m, "ollama") in data]
    leaderboard(data)
    pooled = pairs(data, shortlist)
    if phase6:
        sizes(data)
    pts = configs(cost, data, pooled)
    pareto(pts)
    svg(pts, svg_path)
    print("wrote", os.path.normpath(svg_path))


def main():
    full()


if __name__ == "__main__":
    main()

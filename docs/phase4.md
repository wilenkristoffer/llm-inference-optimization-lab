# Phase 4 - Observability

Technical notes. A read-only API over `lab.db`, a React UI, and backend process
control - plus two data-model fixes that had to land first.

Date: 2026-09-17

---

## Goal

Every result so far has been read once in a terminal and then forgotten. The
database has 107 runs across 25 configurations, two backends and two
precisions; that is enough to answer real questions if it can be queried
visually.

Built now rather than at the end on purpose: it is **infrastructure**. Every
later experiment becomes immediately visible instead of being examined once
after all the work is done.

---

## Two fixes that had to come first

### `gpu_vram_used_mb` was board-wide

It recorded the whole card, including the desktop and browser (~2-3 GB here).
A chart labelled "VRAM used" showing 16 GB for a 1 GB model is worse than no
chart - it looks authoritative.

Fixed by capturing a baseline. `HardwareSampler`'s availability probe already
read every sensor once, so that reading became the baseline for free; the
harnesses now construct the sampler **before** the warm-up, since the warm-up
is what loads the model. The new `gpu_vram_baseline_mb` column lets the UI show
`peak - baseline` as attributable VRAM alongside the raw board figure.

Honest caveat, stated in the code: if the model is already resident when the
baseline is taken, the delta is near zero. That is true - no new memory was
allocated - but it is not "how much VRAM this model needs". For that, evict
first.

### Three names for two models

```
qwen2.5:1.5b                  Ollama, Q4_K_M   (Ollama's default tag IS the
                                                instruct model, unlabelled)
qwen2.5:1.5b-instruct-fp16    Ollama, FP16
qwen2.5-1.5b-instruct         vLLM,   FP16     <- same weights as the row above
```

Grouping a comparison by `model` would treat the last two as different models.
Added a derived `model_key` column, filled in by `record_run()` rather than by
each harness - so no caller can forget it and no two callers can disagree.

It uses an **explicit alias table**, not a clever rule: nothing in the string
`qwen2.5:1.5b` reveals that it is the instruct variant, and guessing wrong
corrupts every downstream comparison. Unknown names fall back to a conservative
normalisation (lowercase, `:`→`-`, drop a trailing precision suffix).

Backfilled across all 107 existing rows. The comparison now reads:

```
qwen2.5-1.5b-instruct   Q4_K_M   ollama   222.2 tok/s
qwen2.5-1.5b-instruct   FP16     vllm     153.5 tok/s
qwen2.5-1.5b-instruct   FP16     ollama   124.9 tok/s
```

One model, three configurations, properly grouped.

---

## What was built

| File | Purpose |
|---|---|
| `api.py` | Read-only JSON over `lab.db`. stdlib `http.server` + `sqlite3`. |
| `servers.py` | Start/stop/readiness for Ollama and vLLM. |
| `ui/src/api.ts` | Typed client. Every metric is `number \| null`. |
| `ui/src/App.tsx` | Dashboard: one hero figure, stat tiles, coverage. |
| `ui/src/ServerControls.tsx` | Backend panel with readiness polling. |
| `ui/src/RunsTable.tsx` | Filterable, sortable, expandable rows. |
| `ui/src/Comparison.tsx` | Concurrency line chart, decode-rate bar chart. |

```
python phase4-observability/api.py      # 127.0.0.1:8010
cd phase4-observability/ui && npm run dev   # localhost:5173
```

Note the asymmetry, which is the Phase 1 IPv6 bug in reverse: Ollama binds
IPv4-only (so `localhost` costs a 2-second IPv6 fallback), while **Vite binds
IPv6-only** (so `127.0.0.1:5173` does not connect at all). The UI is opened on
`localhost`; it calls the API on `127.0.0.1`.

---

## API design

**Groups are the comparison unit.** Within-session noise is 1-2% and
between-session drift ~4%, so a single trial is not a result. `/api/groups`
returns one entry per configuration with median, min, max, sd and cv;
`/api/runs` exists for drill-down only. Encoding that in the API means the UI
cannot accidentally compare two single rows and call it a finding.

**Keyed on (run_group, concurrency), not run_group.** A concurrency sweep
stores every level under one run_group. The first version grouped on run_group
alone and silently averaged concurrency 1 together with 8 - it reported one
"configuration" with 15 trials and a 12.5% cv. Caught immediately because the
cv was implausible for our noise floor.

**Medians in Python, not SQL.** SQLite has no median, and AVG is wrong for
latency - one hiccup drags it while the median shrugs.

**NULL stays null** all the way to the UI, which renders "unavailable". The
TypeScript types are `number | null` for every metric, so the compiler forces
each call site to handle it. A zero in `gpu_percent` would average into a chart
and lie.

---

## Backend process control

Swapping backends by hand was the slowest part of running an experiment. Three
things the panel encodes, all learned the hard way:

**Starting one stops the other.** vLLM reserves VRAM at startup and holds it.
With both running, Ollama's allocation still *succeeds* - WDDM overcommits and
pages the excess to system RAM - so `ollama ps` reports `100% GPU` while ~450 MB
is resident and every token crosses PCIe. That measured 8.1 tok/s instead of
121. A hotswap button that could leave both running would rebuild that trap.

**Stopping Ollama kills `llama-server` too**, by image name rather than only our
own process handle - the tray app may have started it, and `llama-server` is a
separate child holding the weights. An orphaned one held 4.2 GB and blocked
vLLM's startup.

**Readiness is polled, not log-parsed.** Watching for "Application startup
complete" is fragile across versions; asking `/v1/models` whether it answers is
the definition of ready. A server started outside the tool therefore shows
`ready` with `managed: false`, which is correct.

vLLM runs in WSL, and a process started with `wsl -- bash -c "... &"` dies when
that command returns. So the API holds the `wsl.exe` child open instead of
backgrounding it inside the guest, and the guest command uses `exec` so
terminating reaches vLLM rather than orphaning it behind a shell.

**This API executes processes.** It binds 127.0.0.1 only and has no auth - fine
for a local lab tool, not fine for anything reachable.

---

## Charts

Only two, each because a table would answer worse.

**Concurrency: a line chart.** Two lines that *cross* - vLLM leads to batch 4,
plateaus at 563 tok/s, Ollama passes it at 8. Crossing is a picture, not a table
row. Ordinal x-axis (four discrete levels evenly spaced): a linear scale would
bunch 1 and 2 together and imply a continuum the measurements do not have.

**Decode rate: a horizontal bar chart.** Magnitude across categories.

**Colour was validated, not eyeballed.** Ollama is always slot 1 (blue), vLLM
slot 2 (orange), in both charts and both themes - colour follows the entity, so
filtering never repaints a survivor. The pair passes all six checks in both
modes: worst-case CVD ΔE 24.7 light / 26.8 dark against a ≥8 target, and ≥3:1
contrast on both surfaces.

Text never wears the series colour - values and labels use text tokens, and
identity comes from the coloured mark beside them. Every chart carries a legend
*and* a "show data" table, so nothing is gated behind colour vision.

### The bar chart was rebuilt in HTML

The first version was SVG and the labels collided:
`1.5b · Q4_K_M · ollama222±0.8%`. SVG `<text>` cannot wrap or reflow, so a
label longer than its column ran into the bar, and the value ran into the
spread.

Rebuilt as a CSS grid - label, track, value each in their own column. Nothing
can overlap at any width, long labels wrap rather than clip, and it stacks on a
narrow screen instead of squeezing. Value and spread became one label rather
than two texts competing for the same space.

The lesson generalises: hand-measuring text to avoid collisions is work CSS
already does correctly. Use SVG for the marks, HTML for anything with text
layout.

### One reading trap left in place

In the decode-rate chart the longest bar is Ollama at 222 tok/s, which looks
like Ollama winning. It is not a backend result - that row is **Q4_K_M against
everything else's FP16**. The fair backend comparison is the two FP16 rows
(vLLM 153.5 vs Ollama 124.9). The labels carry the precision, but the eye reads
bar length first. If this misleads in practice, splitting into one chart per
precision is the fix.

### Mixed-generation data, handled

The database holds two generations of concurrency sweeps: early ones whose
requests generated ~25 tokens, and later 300-token ones. Comparing across them
is meaningless. The chart takes the most recent sweep per (backend, level) -
which selects the 300-token ones - and prints the `notes` value it used, so the
selection is visible rather than assumed.

---

## Limitations

- Charts are hand-rolled SVG/CSS, no chart library. Fine at this scale; a real
  time-series view would want one.
- No routing - one scrolling page. Adequate for a handful of views.
- `/api/runs` caps at 200 rows and has no pagination.
- The API trusts its caller completely. Localhost-only is the whole security
  model.
- Hardware telemetry is still absent for any run recorded before the sampler
  learned to cover multi-second workloads.

---

## Open items

- Store vLLM's `/metrics` per run (KV cache usage, queue depth, preemptions)
  rather than only reading them live during a diagnosis.
- Quality comparison view for Phase 6 - must be semantic, since greedy decoding
  is not reproducible across runtimes.
- Evict-then-measure flow so `vram_delta_mb` means "what this model needs"
  rather than "what this run newly allocated".

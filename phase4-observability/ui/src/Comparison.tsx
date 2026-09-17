/*
 * The comparison view - where charts finally earn their place.
 *
 * Two charts, each chosen because a table would answer worse:
 *
 *   concurrency  two lines that CROSS. vLLM leads to batch 4, plateaus at 563
 *                tok/s, and Ollama passes it at 8. Crossing is a picture.
 *   throughput   decode rate per configuration - magnitude across categories,
 *                which is a bar chart's job.
 *
 * Colour follows the ENTITY, not rank: Ollama is always slot 1 (blue), vLLM
 * always slot 2 (orange), in both charts and in both themes. Filtering never
 * repaints a survivor. The pair was validated - CVD dE 24.7 light / 26.8 dark
 * against a >=8 target - rather than eyeballed.
 *
 * Text never wears the series colour; identity comes from the coloured mark
 * beside the label. Every chart carries a legend AND a "show data" table,
 * so nothing is gated behind colour vision.
 *
 * ONE DATA TRAP handled here: the database contains two generations of
 * concurrency sweeps - early ones whose requests generated ~25 tokens, and
 * later 300-token ones. Comparing across them is meaningless, so we take the
 * most recent sweep per (backend, level) and print which note it came from.
 */

import { useEffect, useMemo, useState } from "react";
import { fetchGroups } from "./api";
import type { Group } from "./api";

const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

interface Point { label: string; value: number | null; }
interface Series { name: string; color: string; points: Point[]; }

/* ---------------------------------------------------------------- line chart */

function LineChart({ series, yLabel, xLabel }: {
  series: Series[]; yLabel: string; xLabel: string;
}) {
  const [hover, setHover] = useState<number | null>(null);

  const W = 720, H = 300;
  const pad = { top: 16, right: 74, bottom: 40, left: 56 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const labels = series[0]?.points.map((p) => p.label) ?? [];
  const values = series.flatMap((s) => s.points.map((p) => p.value))
                       .filter((v): v is number => v !== null);
  if (!values.length) return <p className="muted">no data</p>;

  const maxY = Math.ceil(Math.max(...values) / 100) * 100 || 100;
  // Ordinal x: four discrete levels evenly spaced. A linear scale would bunch
  // 1 and 2 together and strand 8 at the edge, implying a continuum that the
  // measurements do not have.
  const x = (i: number) => pad.left + (labels.length === 1 ? plotW / 2
                            : (i / (labels.length - 1)) * plotW);
  const y = (v: number) => pad.top + plotH - (v / maxY) * plotH;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(maxY * f));

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`${yLabel} by ${xLabel}, ${series.map(s => s.name).join(" and ")}`}
           onMouseLeave={() => setHover(null)}>
        {/* Gridlines: hairline, solid, recessive - never dashed. */}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.left} x2={W - pad.right} y1={y(t)} y2={y(t)}
                  stroke="var(--gridline)" strokeWidth="1" />
            <text x={pad.left - 10} y={y(t) + 4} textAnchor="end"
                  fill="var(--text-muted)" fontSize="11">{t.toLocaleString()}</text>
          </g>
        ))}
        <line x1={pad.left} x2={W - pad.right} y1={y(0)} y2={y(0)}
              stroke="var(--baseline)" strokeWidth="1" />

        {labels.map((l, i) => (
          <text key={l} x={x(i)} y={H - pad.bottom + 20} textAnchor="middle"
                fill="var(--text-muted)" fontSize="12">{l}</text>
        ))}

        {hover !== null && (
          <line x1={x(hover)} x2={x(hover)} y1={pad.top} y2={y(0)}
                stroke="var(--baseline)" strokeWidth="1" />
        )}

        {series.map((s) => {
          const pts = s.points
            .map((p, i) => (p.value === null ? null : [x(i), y(p.value)] as const))
            .filter((p): p is readonly [number, number] => p !== null);
          if (!pts.length) return null;
          const last = s.points[s.points.length - 1];
          return (
            <g key={s.name}>
              {/* 2px line, round join/cap. */}
              <polyline fill="none" stroke={s.color} strokeWidth="2"
                        strokeLinejoin="round" strokeLinecap="round"
                        points={pts.map(([px, py]) => `${px},${py}`).join(" ")} />
              {/* Markers >=8px with a 2px surface ring so they stay legible
                  where the two lines cross. */}
              {pts.map(([px, py], i) => (
                <circle key={i} cx={px} cy={py} r="4.5" fill={s.color}
                        stroke="var(--surface-1)" strokeWidth="2" />
              ))}
              {/* Direct end-label: supplements the legend, does not replace it.
                  Text wears a text token; the coloured dot carries identity. */}
              {last.value !== null && (
                <text x={x(s.points.length - 1) + 12}
                      y={y(last.value) + 4} fontSize="12"
                      fill="var(--text-secondary)">{Math.round(last.value)}</text>
              )}
            </g>
          );
        })}

        {/* Hit targets deliberately wider than the marks. */}
        {labels.map((l, i) => (
          <rect key={l} x={x(i) - plotW / (labels.length * 2)} y={pad.top}
                width={plotW / labels.length} height={plotH}
                fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}
      </svg>

      {hover !== null && (
        <div className="tooltip">
          <strong>{xLabel} {labels[hover]}</strong>
          {series.map((s) => (
            <div key={s.name} className="tip-row">
              <span className="swatch" style={{ background: s.color }} />
              <span className="secondary">{s.name}</span>
              <span className="tip-value">
                {s.points[hover]?.value?.toFixed(0) ?? "—"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------- bar chart */

/*
 * Built in HTML rather than SVG, deliberately.
 *
 * The first version was SVG and the labels collided: SVG <text> cannot wrap or
 * reflow, so a configuration name longer than its allotted column ran straight
 * into the bar, and the value label ran into the spread. Measuring text by hand
 * to avoid that is exactly the work CSS already does correctly.
 *
 * A grid gives each part its own column - label, track, value - so nothing can
 * overlap at any width, long labels wrap instead of being clipped, and the
 * whole thing collapses to stacked rows on a narrow screen.
 */
function BarChart({ bars, unit }: {
  bars: { label: string; value: number; color: string; sub?: string; title?: string }[];
  unit: string;
}) {
  const max = Math.max(...bars.map((b) => b.value)) || 1;

  return (
    <div className="bars" role="img"
         aria-label={`decode rate by configuration, ${unit}`}>
      {bars.map((b) => (
        <div className="bar-row" key={b.label} title={b.title ?? b.label}>
          <div className="bar-label">{b.label}</div>
          <div className="bar-track">
            {/* Square at the baseline, 4px rounded data-end, 20px thick. */}
            <div className="bar-fill"
                 style={{ width: `${Math.max((b.value / max) * 100, 1)}%`,
                          background: b.color }} />
          </div>
          {/* Value and spread as ONE label - they were two colliding texts
              before. Text wears a text token; the coloured bar carries identity. */}
          <div className="bar-value">
            {b.value.toFixed(0)}
            {b.sub && <span className="cv">{b.sub}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------- the view */

function Legend({ series }: { series: { name: string; color: string }[] }) {
  return (
    <div className="legend">
      {series.map((s) => (
        <span key={s.name} className="legend-item">
          <span className="swatch" style={{ background: s.color }} />
          <span className="secondary">{s.name}</span>
        </span>
      ))}
    </div>
  );
}

function DataTable({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="table-wrap" style={{ marginTop: 10 }}>
      <table>
        <thead><tr>{head.map((h) => <th key={h} style={{ textAlign: "left" }}>{h}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Comparison() {
  const [groups, setGroups] = useState<Group[]>([]);
  const [showConcData, setShowConcData] = useState(false);
  const [showBarData, setShowBarData] = useState(false);

  useEffect(() => { fetchGroups(200).then(setGroups).catch(() => setGroups([])); }, []);

  const backends = useMemo(
    () => [...new Set(groups.map((g) => g.backend))].sort(),
    [groups]);
  const colorFor = (backend: string) => COLORS[backends.indexOf(backend) % COLORS.length];

  // Concurrency: most recent sweep per (backend, level). Older sweeps used
  // ~25-token requests, where fixed per-request costs dominated and batching
  // had almost nothing to amortise - mixing the two would be nonsense.
  const { conc, concNote } = useMemo(() => {
    const levels = [...new Set(groups.filter((g) => g.concurrency !== null)
                                     .map((g) => g.concurrency!))].sort((a, b) => a - b);
    const notes = new Set<string>();
    const series: Series[] = backends.map((backend) => ({
      name: backend,
      color: colorFor(backend),
      points: levels.map((level) => {
        const matches = groups
          .filter((g) => g.backend === backend && g.concurrency === level)
          .sort((a, b) => b.timestamp_utc.localeCompare(a.timestamp_utc));
        if (matches[0]?.notes) notes.add(matches[0].notes);
        return { label: String(level), value: matches[0]?.aggregate_tok_s ?? null };
      }),
    })).filter((s) => s.points.some((p) => p.value !== null));
    return { conc: series, concNote: [...notes].join(", ") };
  }, [groups, backends]);

  // Single-stream decode by configuration: most recent per
  // (backend, model_key, precision).
  const bars = useMemo(() => {
    const best = new Map<string, Group>();
    groups.filter((g) => g.concurrency === null && g.decode_tok_s).forEach((g) => {
      const key = `${g.backend}|${g.model_key}|${g.quantization}`;
      const prev = best.get(key);
      if (!prev || g.timestamp_utc > prev.timestamp_utc) best.set(key, g);
    });
    return [...best.values()]
      .sort((a, b) => (b.decode_tok_s!.median - a.decode_tok_s!.median))
      .map((g) => {
        const short = g.model_key?.replace("qwen2.5-", "").replace("-instruct", "")
                      ?? g.model;
        const d = g.decode_tok_s!;
        return {
          label: `${short} · ${g.quantization} · ${g.backend}`,
          value: d.median,
          color: colorFor(g.backend),
          sub: d.cv ? `±${d.cv.toFixed(1)}%` : undefined,
          title: `${g.model} · ${g.quantization} · ${g.backend}\n` +
                 `median ${d.median.toFixed(1)} tok/s, ` +
                 `min ${d.min.toFixed(1)}, max ${d.max.toFixed(1)}, n=${d.n}`,
        };
      });
  }, [groups, backends]);

  if (!groups.length) return null;

  return (
    <section style={{ marginTop: 28 }}>
      <h2>Comparisons</h2>

      {conc.length > 0 && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="chart-head">
            <div>
              <strong>Aggregate throughput by concurrency</strong>
              <div className="muted" style={{ fontSize: 12 }}>
                tokens/sec across all concurrent requests
                {concNote && ` · ${concNote}`}
              </div>
            </div>
            <button className="sort" onClick={() => setShowConcData(!showConcData)}>
              {showConcData ? "hide data" : "show data"}
            </button>
          </div>
          <Legend series={conc} />
          <LineChart series={conc} yLabel="tokens/sec" xLabel="concurrency" />
          {showConcData && (
            <DataTable
              head={["concurrency", ...conc.map((s) => s.name)]}
              rows={(conc[0]?.points ?? []).map((p, i) => [
                p.label, ...conc.map((s) => s.points[i]?.value?.toFixed(0) ?? "—"),
              ])} />
          )}
        </div>
      )}

      {bars.length > 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="chart-head">
            <div>
              <strong>Single-stream decode rate</strong>
              <div className="muted" style={{ fontSize: 12 }}>
                one request at a time · ± is the coefficient of variation
              </div>
            </div>
            <button className="sort" onClick={() => setShowBarData(!showBarData)}>
              {showBarData ? "hide data" : "show data"}
            </button>
          </div>
          <Legend series={backends.map((b) => ({ name: b, color: colorFor(b) }))} />
          <BarChart bars={bars} unit="tokens/sec" />
          {showBarData && (
            <DataTable head={["configuration", "tok/s", "spread"]}
                       rows={bars.map((b) => [b.label, b.value.toFixed(1), b.sub ?? "—"])} />
          )}
        </div>
      )}
    </section>
  );
}

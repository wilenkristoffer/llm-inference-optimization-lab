/*
 * The runs table: every measured configuration, sortable and filterable.
 *
 * A table, not a chart, on purpose. "Which configuration was fastest" is a
 * sort; a chart would add ink without adding an answer. Charts come with the
 * comparison view, where the question is how a metric varies along an axis.
 *
 * Two things this makes hard to get wrong:
 *
 * - Every median is shown with its spread. Our within-session noise floor is
 *   1-2% and between-session drift is ~4%, so two numbers 3% apart may be the
 *   same measurement. The cv column is what stops that being invisible.
 *
 * - The notes column is always visible. Two sweeps in this database used
 *   ~25-token requests and the rest use 300-token ones; comparing across them
 *   is meaningless, and the note is the only thing that says so.
 */

import { useEffect, useMemo, useState } from "react";
import { fetchGroups, fetchRuns } from "./api";
import type { Group, Run, Spread } from "./api";

type SortKey = "when" | "decode" | "ttft" | "aggregate" | "gpu";

const num = (s: Spread | null) => (s ? s.median : null);

function Cell({ value, digits = 1 }: { value: number | null; digits?: number }) {
  if (value === null) return <span className="muted">&mdash;</span>;
  return <>{value.toFixed(digits)}</>;
}

/** Median with its coefficient of variation, so spread is never hidden. */
function WithSpread({ s, digits = 1 }: { s: Spread | null; digits?: number }) {
  if (!s) return <span className="muted">&mdash;</span>;
  const cv = s.cv;
  return (
    <>
      {s.median.toFixed(digits)}
      {cv != null && cv > 0 && (
        <span className="cv" title={`min ${s.min.toFixed(1)}, max ${s.max.toFixed(1)}, n=${s.n}`}>
          &plusmn;{cv.toFixed(1)}%
        </span>
      )}
    </>
  );
}

function Detail({ group }: { group: Group }) {
  const [runs, setRuns] = useState<Run[] | null>(null);

  useEffect(() => {
    // Lazily loaded: the trial rows carry full response text and would bloat
    // the list request for every row you never open.
    fetchRuns(group.run_group, 50).then(setRuns).catch(() => setRuns([]));
  }, [group.run_group]);

  const sample = runs?.find((r) => r.response);

  return (
    <div className="detail">
      <div className="detail-grid">
        <div><span className="tile-label">backend version</span>
             <div>{group.backend_version ?? <span className="muted">unknown</span>}</div></div>
        <div><span className="tile-label">context length</span>
             <div>{group.context_length ?? <span className="muted">&mdash;</span>}</div></div>
        <div><span className="tile-label">output tokens</span>
             <div><Cell value={num(group.output_tokens)} digits={0} /></div></div>
        <div><span className="tile-label">latency median</span>
             <div><Cell value={num(group.latency_ms)} digits={0} /> ms</div></div>
        <div><span className="tile-label">gpu clock</span>
             <div><Cell value={num(group.gpu_clock_mhz)} digits={0} /> MHz</div></div>
        <div><span className="tile-label">gpu power</span>
             <div><Cell value={num(group.gpu_power_w)} digits={0} /> W</div></div>
        <div><span className="tile-label">gpu temp</span>
             <div><Cell value={num(group.gpu_temp_c)} digits={0} /> &deg;C</div></div>
        <div><span className="tile-label">cpu</span>
             <div><Cell value={num(group.cpu_percent)} /> %</div></div>
        <div>
          <span className="tile-label">vram attributable</span>
          <div>
            {group.vram_delta_mb !== null
              ? <>{group.vram_delta_mb.toFixed(0)} MB</>
              : <span className="muted">no baseline recorded</span>}
          </div>
        </div>
        <div>
          <span className="tile-label">vram board-wide peak</span>
          <div><Cell value={group.vram_peak_mb} digits={0} /> MB</div>
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <span className="tile-label">model / run group</span>
        <div className="mono">{group.model}</div>
        <div className="mono muted">{group.run_group}</div>
      </div>

      {sample && (
        <div style={{ marginTop: 12 }}>
          <span className="tile-label">generated output (first trial)</span>
          <p className="sample">{sample.response?.slice(0, 600)}
            {(sample.response?.length ?? 0) > 600 && "..."}</p>
        </div>
      )}
      {runs === null && <p className="muted" style={{ marginTop: 10 }}>loading trials...</p>}
    </div>
  );
}

export default function RunsTable() {
  const [groups, setGroups] = useState<Group[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("when");
  const [desc, setDesc] = useState(true);
  const [open, setOpen] = useState<string | null>(null);
  const [backend, setBackend] = useState("all");
  const [model, setModel] = useState("all");
  const [quant, setQuant] = useState("all");
  const [conc, setConc] = useState("all");

  useEffect(() => { fetchGroups(200).then(setGroups).catch(() => setGroups([])); }, []);

  const options = useMemo(() => ({
    backends: [...new Set(groups.map((g) => g.backend))].sort(),
    models: [...new Set(groups.map((g) => g.model_key ?? g.model))].sort(),
    quants: [...new Set(groups.map((g) => g.quantization ?? "unknown"))].sort(),
    concs: [...new Set(groups.map((g) => String(g.concurrency ?? "single")))].sort(),
  }), [groups]);

  const rows = useMemo(() => {
    const filtered = groups.filter((g) =>
      (backend === "all" || g.backend === backend) &&
      (model === "all" || (g.model_key ?? g.model) === model) &&
      (quant === "all" || (g.quantization ?? "unknown") === quant) &&
      (conc === "all" || String(g.concurrency ?? "single") === conc));

    const value = (g: Group): number | string | null => {
      switch (sortKey) {
        case "decode": return num(g.decode_tok_s);
        case "ttft": return num(g.ttft_ms);
        case "aggregate": return g.aggregate_tok_s;
        case "gpu": return num(g.gpu_percent);
        default: return g.timestamp_utc;
      }
    };

    return [...filtered].sort((a, b) => {
      const av = value(a), bv = value(b);
      // Nulls always sink, whichever direction is selected - "unavailable"
      // is not a small number and must not sort as one.
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return desc ? -cmp : cmp;
    });
  }, [groups, sortKey, desc, backend, model, quant, conc]);

  function header(key: SortKey, label: string, align: "left" | "right" = "right") {
    const active = sortKey === key;
    return (
      <th style={{ textAlign: align }}>
        <button className="sort" onClick={() => {
          if (active) setDesc(!desc); else { setSortKey(key); setDesc(true); }
        }}>
          {label}{active && (desc ? " ↓" : " ↑")}
        </button>
      </th>
    );
  }

  const select = (v: string, set: (s: string) => void, opts: string[], label: string) => (
    <label className="filter">
      <span className="tile-label">{label}</span>
      <select value={v} onChange={(e) => set(e.target.value)}>
        <option value="all">all</option>
        {opts.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  );

  return (
    <section style={{ marginTop: 28 }}>
      <h2>Runs</h2>
      <p className="muted" style={{ margin: "4px 0 0" }}>
        One row per measured configuration. &plusmn; is the coefficient of
        variation &mdash; the noise floor here is 1-2%, so anything closer than
        that is the same measurement.
      </p>

      <div className="filters">
        {select(backend, setBackend, options.backends, "backend")}
        {select(model, setModel, options.models, "model")}
        {select(quant, setQuant, options.quants, "precision")}
        {select(conc, setConc, options.concs, "concurrency")}
        <span className="muted" style={{ alignSelf: "end", fontSize: 12 }}>
          {rows.length} of {groups.length}
        </span>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>backend</th>
              <th style={{ textAlign: "left" }}>model</th>
              <th>prec</th>
              <th>conc</th>
              <th>n</th>
              {header("decode", "decode tok/s")}
              {header("aggregate", "aggregate")}
              {header("ttft", "ttft ms")}
              {header("gpu", "gpu %")}
              {header("when", "when")}
              <th style={{ textAlign: "left" }}>notes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((g) => {
              const id = `${g.run_group}|${g.concurrency ?? "s"}`;
              const isOpen = open === id;
              return [
                <tr key={id} onClick={() => setOpen(isOpen ? null : id)}
                    className={isOpen ? "row open" : "row"}>
                  <td>{g.backend}</td>
                  <td className="mono">{g.model_key ?? g.model}</td>
                  <td style={{ textAlign: "center" }}>{g.quantization ?? "?"}</td>
                  <td style={{ textAlign: "right" }}>{g.concurrency ?? "-"}</td>
                  <td style={{ textAlign: "right" }}>{g.trials}</td>
                  <td style={{ textAlign: "right" }}><WithSpread s={g.decode_tok_s} /></td>
                  <td style={{ textAlign: "right" }}><Cell value={g.aggregate_tok_s} digits={0} /></td>
                  <td style={{ textAlign: "right" }}><WithSpread s={g.ttft_ms} /></td>
                  <td style={{ textAlign: "right" }}><Cell value={num(g.gpu_percent)} digits={0} /></td>
                  <td style={{ textAlign: "right" }} className="muted">
                    {g.timestamp_utc.slice(5, 16).replace("T", " ")}
                  </td>
                  <td className="muted">{g.notes ?? ""}</td>
                </tr>,
                isOpen && (
                  <tr key={id + "-d"}>
                    <td colSpan={11} style={{ padding: 0 }}><Detail group={g} /></td>
                  </tr>
                ),
              ];
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

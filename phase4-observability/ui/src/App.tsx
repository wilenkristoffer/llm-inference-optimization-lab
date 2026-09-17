/*
 * Phase 4, the dashboard.
 *
 * Answers "what is in the lab, and what does it say" at a glance. No charts
 * yet - a chart earns its place when it makes a comparison easier to read than
 * a number would, and these are single values.
 *
 * One hero figure per view (the run count - this dashboard's subject), then
 * stat tiles. Anything not measured renders as "unavailable", never as 0.
 */

import { useEffect, useState } from "react";
import { fetchSummary, fetchGroups, compact } from "./api";
import ServerControls from "./ServerControls";
import RunsTable from "./RunsTable";
import Comparison from "./Comparison";
import type { Summary, Group } from "./api";

function Tile({ label, value, unit, note }: {
  label: string;
  value: number | null;
  unit?: string;
  note?: string;
}) {
  return (
    <div className="card">
      <div className="tile-label">{label}</div>
      {value === null ? (
        // A null is a fact: it was not measured. Saying so is more useful than
        // a zero that would silently average into every later comparison.
        <div className="tile-value"><span className="unavailable">unavailable</span></div>
      ) : (
        <div className="tile-value">
          {compact(value)}
          {unit && <span className="tile-unit">{unit}</span>}
        </div>
      )}
      {note && <div className="tile-label" style={{ marginTop: 6 }}>{note}</div>}
    </div>
  );
}

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [groups, setGroups] = useState<Group[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchSummary(), fetchGroups()])
      .then(([s, g]) => { setSummary(s); setGroups(g); })
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="app">
        <div className="card error">
          <h2>Cannot reach the API</h2>
          <p className="secondary">{error}</p>
          <p className="muted">
            Start it with <code>python phase4-observability/api.py</code>
          </p>
        </div>
      </div>
    );
  }

  if (!summary) return <div className="app"><p className="muted">Loading...</p></div>;

  // Which configurations have hardware telemetry at all. Worth surfacing: it
  // is null for every vLLM run made before the sampler learned to cover
  // multi-second workloads, and for any run where LHM was not running.
  const withGpu = groups.filter((g) => g.gpu_percent !== null).length;
  const concurrencyRuns = groups.filter((g) => g.concurrency !== null).length;

  return (
    <div className="app">
      <header style={{ marginBottom: 20 }}>
        <h1>LLM inference lab</h1>
        <p className="muted" style={{ margin: "4px 0 0" }}>
          Qwen2.5 on a Radeon RX 7800 XT via ROCm. Ollama and vLLM.
        </p>
      </header>

      <div className="card">
        <div className="tile-label">Inference runs recorded</div>
        <div className="hero-value">{summary.total_runs.toLocaleString()}</div>
        <div className="secondary" style={{ marginTop: 6 }}>
          across {summary.run_groups} measured configurations
          {summary.latest && ` · latest ${summary.latest.slice(0, 10)}`}
        </div>
      </div>

      <div className="tiles">
        <Tile label="Median decode rate" value={summary.median_decode_tok_s} unit="tok/s"
              note="single-stream configurations" />
        <Tile label="Median time to first token" value={summary.median_ttft_ms} unit="ms" />
        <Tile label="Models" value={summary.models} note="distinct weights" />
        <Tile label="Backends" value={summary.backends} />
        <Tile label="Precisions" value={summary.quantizations} />
        <Tile label="Peak VRAM recorded" value={summary.max_recorded_vram_mb} unit="MB"
              note="board-wide, includes desktop" />
      </div>

      <ServerControls />

      <Comparison />

      <RunsTable />

      <h2 style={{ marginTop: 28 }}>Coverage</h2>
      <div className="tiles">
        <Tile label="Configurations with GPU telemetry" value={withGpu}
              note={`of ${groups.length} total`} />
        <Tile label="Concurrency measurements" value={concurrencyRuns} />
      </div>
    </div>
  );
}

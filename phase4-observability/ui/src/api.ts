/*
 * Typed client for the Phase 4 API.
 *
 * Every numeric field is `number | null`, never `number`. That is not defensive
 * typing - it is the schema's contract. gpu_percent is genuinely null for runs
 * where LibreHardwareMonitor was not running, and server-side timings are null
 * for every run over the OpenAI-compatible interface because that schema has no
 * timing fields at all. TypeScript forcing us to handle null at each call site
 * is exactly what stops "unavailable" being rendered as 0.
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8010";

/** Median plus enough spread to judge whether a difference is real. */
export interface Spread {
  median: number;
  min: number;
  max: number;
  n: number;
  sd?: number;
  /** Coefficient of variation, %. Misleading on small absolute values. */
  cv?: number | null;
}

export interface Summary {
  total_runs: number;
  run_groups: number;
  models: number;
  backends: number;
  quantizations: number;
  latest: string | null;
  median_decode_tok_s: number | null;
  median_ttft_ms: number | null;
  /** Highest board-wide VRAM ever recorded - NOT a live reading. */
  max_recorded_vram_mb: number | null;
}

/**
 * One measured configuration. Keyed on (run_group, concurrency): a concurrency
 * sweep stores every level under one run_group, so the level is part of the
 * identity rather than an attribute of it.
 */
export interface Group {
  run_group: string;
  timestamp_utc: string;
  backend: string;
  backend_version: string | null;
  model: string;
  /** Canonical identity - the same weights across backends that name them differently. */
  model_key: string | null;
  quantization: string | null;
  context_length: number | null;
  concurrency: number | null;
  trials: number;
  notes: string | null;

  decode_tok_s: Spread | null;
  ttft_ms: Spread | null;
  latency_ms: Spread | null;
  output_tokens: Spread | null;
  /** Only present for concurrency runs. */
  aggregate_tok_s: number | null;

  gpu_percent: Spread | null;
  gpu_clock_mhz: Spread | null;
  gpu_power_w: Spread | null;
  gpu_temp_c: Spread | null;
  cpu_percent: Spread | null;

  /** Board-wide peak - includes the desktop and browser. */
  vram_peak_mb: number | null;
  vram_baseline_mb: number | null;
  /** peak - baseline. Prefer this; say which one is on screen. */
  vram_delta_mb: number | null;
}

/**
 * "starting" means we hold a live process that is not answering yet - vLLM
 * takes about a minute to load weights. "ready" means it answered, whoever
 * started it, which is why a server launched outside this tool still shows
 * ready with managed=false.
 */
export type ServerStatus = "stopped" | "starting" | "ready";

export interface ServerState {
  name: string;
  label: string;
  where: string;
  port: number;
  status: ServerStatus;
  managed: boolean;
  log: string[];
}

export type Servers = Record<string, ServerState>;

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

/** One trial - a single request. Comparisons use Group; this is for drill-down. */
export interface Run {
  id: number;
  run_group: string;
  trial_index: number;
  timestamp_utc: string;
  backend: string;
  model: string;
  quantization: string | null;
  prompt: string | null;
  response: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  ttft_ms: number | null;
  client_total_ms: number | null;
  decode_tokens_per_s: number | null;
  concurrency: number | null;
  gpu_percent: number | null;
  gpu_vram_used_mb: number | null;
  notes: string | null;
}

export const fetchRuns = (runGroup: string, limit = 50) =>
  get<Run[]>(`/api/runs?run_group=${encodeURIComponent(runGroup)}&limit=${limit}`);

export const fetchSummary = () => get<Summary>("/api/summary");
export const fetchGroups = (limit = 100) => get<Group[]>(`/api/groups?limit=${limit}`);
export const fetchServers = () => get<Servers>("/api/servers");

/**
 * Start or stop a backend. Starting is exclusive server-side unless keepOther
 * is set: it stops the other one first, because if both do not fit, Ollama
 * pages over PCIe at ~8 tok/s while still reporting "100% GPU". keepOther is
 * safe for the 1.5B on both sides, not for bigger models.
 *
 * Returns as soon as the process is spawned - poll fetchServers for readiness.
 */
export async function controlServer(
  name: string, action: "start" | "stop", keepOther = false,
): Promise<Servers> {
  const query = action === "start" && keepOther ? "?keep_other=1" : "";
  const response = await fetch(`${API_BASE}/api/servers/${name}/${action}${query}`,
                               { method: "POST" });
  if (!response.ok) throw new Error(`${name}/${action} failed: ${response.status}`);
  return response.json() as Promise<Servers>;
}

/** Compact large numbers the way a stat tile should: 1,284 / 12.9K / 1.3M. */
export function compact(value: number | null, digits = 1): string {
  if (value === null || value === undefined) return "-";
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(digits)}M`;
  if (Math.abs(value) >= 10_000) return `${(value / 1000).toFixed(digits)}K`;
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

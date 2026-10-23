/*
 * Backend control panel.
 *
 * Swapping backends by hand is the slowest part of an experiment: stop one,
 * remember that llama-server is a separate process, start the other, then
 * watch a log for a minute waiting for vLLM to finish loading. This makes it
 * one click and says when the server is actually ready.
 *
 * Readiness is polled from the server, not parsed out of a log line. The log
 * tail is shown while starting because a minute of silence looks like a hang -
 * but it is decoration, not the signal.
 *
 * Status is never colour alone: each state ships a dot AND a word, because a
 * red/green dot is invisible to a good share of readers and to anyone printing
 * this.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchServers, controlServer } from "./api";
import type { Servers, ServerStatus } from "./api";

const DOT: Record<ServerStatus, string> = {
  ready: "var(--status-good)",
  starting: "var(--status-warning)",
  stopped: "var(--text-muted)",
};

const WORD: Record<ServerStatus, string> = {
  ready: "ready",
  starting: "starting",
  stopped: "stopped",
};

export default function ServerControls() {
  const [servers, setServers] = useState<Servers | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Off by default and not remembered: it is only safe for the 1.5B, and a
  // setting that silently persists into a 14B run is how the 8 tok/s happens.
  const [keepOther, setKeepOther] = useState(false);
  // 8 for concurrency work, 1 for VRAM footprints (each slot preallocates
  // its own KV cache). Applied on Start - Ollama reads it once at launch.
  const [ollamaSlots, setOllamaSlots] = useState(8);
  // Avoids a stale poll response overwriting a fresh one after an action.
  const inflight = useRef(false);

  const poll = useCallback(async () => {
    if (inflight.current) return;
    inflight.current = true;
    try {
      setServers(await fetchServers());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      inflight.current = false;
    }
  }, []);

  useEffect(() => {
    poll();
    // 2s: fast enough that "ready" feels immediate, slow enough that polling
    // two health endpoints costs nothing.
    const timer = setInterval(poll, 2000);
    return () => clearInterval(timer);
  }, [poll]);

  async function act(name: string, action: "start" | "stop") {
    setBusy(name);
    try {
      setServers(await controlServer(name, action, keepOther,
                                     name === "ollama" ? ollamaSlots : undefined));
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  if (!servers) return null;

  const list = Object.values(servers);
  const anyStarting = list.some((s) => s.status === "starting");
  // Same for every server: it is a property of where the API runs.
  const control = list.every((s) => s.control);

  return (
    <section style={{ marginTop: 28 }}>
      <h2>Backends</h2>
      <p className="muted" style={{ margin: "4px 0 0" }}>
        Starting one stops the other. If both do not fit in 16&nbsp;GB, the
        loser pages over PCIe and reads ~8&nbsp;tok/s while still reporting
        &ldquo;100% GPU&rdquo;.
      </p>
      <label className="muted" style={{
        display: "flex", alignItems: "center", gap: 6, marginTop: 8, fontSize: 13,
      }}>
        <input type="checkbox" checked={keepOther}
               onChange={(e) => setKeepOther(e.target.checked)} />
        Keep the other one running &mdash; 1.5B models only
      </label>
      {!control && (
        <p className="muted" style={{ margin: "8px 0 0" }}>
          The API is running in a container, so it can report health but not
          start or stop the servers. Start them natively.
        </p>
      )}

      {error && (
        <div className="card error" style={{ marginTop: 12 }}>
          <span className="secondary">{error}</span>
        </div>
      )}

      <div className="tiles">
        {list.map((s) => (
          <div className="card" key={s.name}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <span aria-hidden style={{
                width: 9, height: 9, borderRadius: "50%",
                background: DOT[s.status], flex: "0 0 auto",
              }} />
              <strong>{s.label}</strong>
              <span className="muted" style={{ fontSize: 12 }}>{s.where}</span>
            </div>

            <div className="tile-value" style={{ fontSize: 18 }}>
              {WORD[s.status]}
              {s.status === "ready" && !s.managed && (
                <span className="tile-unit">started outside</span>
              )}
            </div>

            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <button
                onClick={() => act(s.name, "start")}
                disabled={!control || busy !== null || s.status !== "stopped"}
              >
                {busy === s.name ? "working..." : "Start"}
              </button>
              <button
                onClick={() => act(s.name, "stop")}
                disabled={!control || busy !== null || s.status === "stopped"}
              >
                Stop
              </button>
              {s.name === "ollama" && (
                <label className="muted" style={{
                  display: "flex", alignItems: "center", gap: 6, fontSize: 13,
                }}>
                  slots
                  <select value={ollamaSlots}
                          disabled={s.status !== "stopped"}
                          onChange={(e) => setOllamaSlots(Number(e.target.value))}>
                    <option value={8}>8 (concurrency)</option>
                    <option value={1}>1 (VRAM footprint)</option>
                  </select>
                </label>
              )}
            </div>

            {s.status === "starting" && s.log.length > 0 && (
              <pre style={{
                marginTop: 10, marginBottom: 0, maxHeight: 120, overflow: "auto",
                fontSize: 11, lineHeight: 1.4, color: "var(--text-muted)",
                background: "var(--page)", padding: 8,
                borderRadius: 6, whiteSpace: "pre-wrap",
              }}>
                {s.log.slice(-6).join("\n")}
              </pre>
            )}
          </div>
        ))}
      </div>

      {anyStarting && (
        <p className="muted" style={{ marginTop: 10 }}>
          vLLM loads weights and profiles the KV cache before it answers &mdash;
          usually about a minute. The dot turns green when it responds. The
          full timestamped log is in phase4-observability/logs/.
        </p>
      )}
    </section>
  );
}

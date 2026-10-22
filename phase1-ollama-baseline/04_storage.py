"""
Phase 1, file 4: persistent storage for experiment results.

This schema is the contract for the whole project. Phase 4's UI reads it,
Phases 6 and 9 write to it, Phase 10 queries across all of it.

Two rules it is built around:

1. Nothing is NOT NULL except identity. A field we cannot measure stores NULL,
   never 0 and never a guess. A zero for gpu_utilisation is a lie that will
   quietly average into a chart later.

2. Every run records the CONFIGURATION that produced it, not just the result.
   A tokens/sec number without its model, quantization, context length and
   backend is not a measurement, it is trivia.

Usage:
    python 04_storage.py          # create the db and run a self-test
"""

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "lab.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Grouping. One experiment = many trials. We compare medians of groups,
    -- never single rows, because we now know the noise floor is ~1-2%.
    run_group           TEXT    NOT NULL,
    trial_index         INTEGER NOT NULL,
    timestamp_utc       TEXT    NOT NULL,   -- ISO8601 UTC, always UTC

    -- WHAT WAS RUN -------------------------------------------------------
    backend             TEXT    NOT NULL,   -- 'ollama', later 'vllm'
    backend_version     TEXT,
    model               TEXT    NOT NULL,   -- 'qwen2.5:1.5b'
    model_digest        TEXT,               -- exact weights id; tags get reused
    quantization        TEXT,               -- 'Q4_K_M' - the Phase 6 variable
    context_length      INTEGER,            -- num_ctx; drives KV cache size
    temperature         REAL,
    seed                INTEGER,
    streaming           INTEGER,            -- 0/1; SQLite has no bool

    prompt              TEXT,
    response            TEXT,               -- kept: quality is a result too

    -- TOKENS -------------------------------------------------------------
    input_tokens        INTEGER,
    cached_input_tokens INTEGER,            -- KV prefix reuse; inflates prefill
    output_tokens       INTEGER,
    total_tokens        INTEGER,

    -- TIMING, milliseconds ------------------------------------------------
    -- Client-side. The only place TTFT can honestly be measured.
    ttft_ms             REAL,
    client_total_ms     REAL,
    -- Server-side, as the backend reports it. Kept separately so we can see
    -- the gap rather than assume it away.
    load_ms             REAL,               -- cold start; excluded from steady state
    prefill_ms          REAL,
    decode_ms           REAL,
    server_total_ms     REAL,

    -- DERIVED -------------------------------------------------------------
    decode_tokens_per_s  REAL,              -- the stable, comparable number
    prefill_tokens_per_s REAL,              -- over UNCACHED tokens only

    -- HARDWARE, all nullable ----------------------------------------------
    -- On AMD + Windows most of these are unavailable. NULL means unavailable,
    -- and the UI must render it as "unavailable", not as zero.
    -- Concurrency experiments. Every row of one concurrent batch carries the
    -- same batch_wall_ms, so aggregate throughput is recoverable later as
    -- SUM(output_tokens) / batch_wall_ms rather than only living in a printout.
    concurrency         INTEGER,
    batch_wall_ms       REAL,

    hardware_id         TEXT,
    cpu_percent         REAL,
    system_ram_used_mb  REAL,
    gpu_percent         REAL,
    gpu_vram_used_mb    REAL,
    gpu_temp_c          REAL,
    gpu_power_w         REAL,
    gpu_clock_mhz       REAL,
    cpu_temp_c          REAL,

    notes               TEXT
);

-- Indexes on what we will actually filter and group by.
CREATE INDEX IF NOT EXISTS idx_runs_group   ON inference_runs(run_group);
CREATE INDEX IF NOT EXISTS idx_runs_compare ON inference_runs(model, backend, quantization);

-- Phase 6: fidelity of a quantized model to a reference, one row per token
-- position of the reference text. A separate table because the unit is a
-- token, not a request - ~2,000 rows per model, which would drown
-- inference_runs.
CREATE TABLE IF NOT EXISTS fidelity_positions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_group           TEXT NOT NULL,
    timestamp_utc       TEXT NOT NULL,
    backend             TEXT,
    backend_version     TEXT,
    model               TEXT NOT NULL,      -- the model being scored
    model_key           TEXT,
    model_digest        TEXT,
    quantization        TEXT,
    reference_model     TEXT NOT NULL,      -- whose greedy text is the reference
    prompt_index        INTEGER NOT NULL,
    position            INTEGER NOT NULL,   -- token index within the reference
    ref_token           TEXT,
    aligned             INTEGER,            -- 1 if the server saw exactly the
                                            -- expected prefix length
    top1_match          INTEGER,            -- model's top token == reference token
    ref_logprob         REAL,               -- model's logprob OF the reference
                                            -- token; NULL if outside its top-k
    ref_in_topk         INTEGER,
    kl_topk             REAL,               -- approx KL(reference || model) over
                                            -- the reference's top-k tokens
    top_k               INTEGER
);
CREATE INDEX IF NOT EXISTS idx_fid_group ON fidelity_positions(run_group, model);

-- Phase 6: one row per (model, question) of the checkable task set. The
-- response is kept so any scoring decision can be re-checked by eye later.
CREATE TABLE IF NOT EXISTS task_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_group           TEXT NOT NULL,
    timestamp_utc       TEXT NOT NULL,
    backend             TEXT,
    backend_version     TEXT,
    model               TEXT NOT NULL,
    model_key           TEXT,
    model_digest        TEXT,
    quantization        TEXT,
    question_id         TEXT NOT NULL,
    category            TEXT,
    correct             INTEGER,
    reason              TEXT,               -- why it failed, when it did
    response            TEXT,
    output_tokens       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_task_group ON task_results(run_group, model);

-- Phase 7: what the hardware can do, measured by synthetic probes rather than
-- taken from a datasheet. One row per (probe, buffer size). The unit is not a
-- request, so it does not belong in inference_runs.
CREATE TABLE IF NOT EXISTS hardware_ceilings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_group           TEXT NOT NULL,
    timestamp_utc       TEXT NOT NULL,
    hardware_id         TEXT,
    probe               TEXT NOT NULL,      -- e.g. vram_read, h2d_pinned, matmul_fp16
    environment         TEXT,               -- where it ran: wsl2-rocm, wsl2-cpu, ...
    bytes               INTEGER,            -- working-set size; NULL for compute probes
    best                REAL,               -- best repeat: the ceiling
    median              REAL,               -- median repeat: what you usually get
    unit                TEXT,               -- GB/s or TFLOPS
    repeats             INTEGER,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_ceil_group ON hardware_ceilings(run_group, probe);
"""


# Columns added after the table already existed in the wild. CREATE TABLE IF
# NOT EXISTS does nothing to an existing table, and SQLite has no
# "ADD COLUMN IF NOT EXISTS" - so we diff against the live schema.
MIGRATIONS = {
    "gpu_clock_mhz": "REAL",
    "cpu_temp_c": "REAL",
    "concurrency": "INTEGER",
    "batch_wall_ms": "REAL",
    # Board VRAM BEFORE this run's workload, so the UI can show
    # (gpu_vram_used_mb - gpu_vram_baseline_mb) instead of a board-wide total
    # that includes the desktop and browser.
    "gpu_vram_baseline_mb": "REAL",
    # Canonical identity for "the same weights", across backends that name
    # them differently. Derived, not typed in - see canonical_model().
    "model_key": "TEXT",
    # 1 if the baseline was taken with NO model resident (evicted and
    # confirmed), so (used - baseline) is this model's footprint. 0 if
    # something was loaded; NULL for rows from before this was tracked. Only
    # rows with 1 answer "how much VRAM does this model need".
    "gpu_vram_baseline_clean": "INTEGER",
    # The server's own claim of the model's VRAM (Ollama /api/ps size_vram).
    # A cross-check, not a measurement: Ollama reported "100% GPU" while
    # paging in Phase 3.
    "server_vram_mb": "REAL",
    # Parallel slots the runner was launched with (llama-server -np), read
    # from the process. The KV cache is preallocated for num_ctx x this, so
    # footprints are only comparable at the same value.
    "server_num_parallel": "INTEGER",
    # Phase 7. Layers on the GPU as requested (num_gpu -> llama-server -ngl);
    # NULL means not set, so the runtime chose. The model's total resident size
    # (/api/ps size) goes beside server_vram_mb so the GPU share is derivable.
    "num_gpu_layers": "INTEGER",
    "server_size_mb": "REAL",
    # The GPU library the runner loaded (ROCm, Vulkan, CPU), from server.log -
    # the same runtime on different libraries is Phase 7's backend comparison.
    "gpu_library": "TEXT",
    "num_thread": "INTEGER",
}


# Backends name identical weights differently:
#
#   qwen2.5:1.5b                  Ollama, Q4_K_M  (Ollama's default tag IS the
#                                                  instruct model, unlabelled)
#   qwen2.5:1.5b-instruct-fp16    Ollama, FP16
#   qwen2.5-1.5b-instruct         vLLM,   FP16
#
# Grouping a comparison by `model` would treat the last two as different
# models when they are the same weights on different runtimes. model_key
# collapses them; `quantization` still separates FP16 from Q4_K_M.
#
# An explicit table, not a clever rule: "qwen2.5:1.5b" cannot be known to be
# the instruct variant from its name alone, and guessing wrong here corrupts
# every comparison downstream.
MODEL_ALIASES = {
    "qwen2.5:1.5b": "qwen2.5-1.5b-instruct",
    "qwen2.5:1.5b-instruct-fp16": "qwen2.5-1.5b-instruct",
    "qwen2.5-1.5b-instruct": "qwen2.5-1.5b-instruct",
    "qwen2.5:3b": "qwen2.5-3b-instruct",
    "qwen2.5:7b": "qwen2.5-7b-instruct",
    "qwen2.5:14b": "qwen2.5-14b-instruct",
    "qwen2.5:32b": "qwen2.5-32b-instruct",
}

_PRECISION_SUFFIXES = ("-fp16", "-f16", "-bf16", "-fp32", "-f32",
                       "-q8_0", "-q6_k", "-q5_k_m", "-q4_k_m", "-q4_0")


def canonical_model(name):
    """
    Map a backend-specific model name onto a canonical identity.

    Known names come from MODEL_ALIASES. Anything else falls back to a
    conservative normalisation - lowercase, ':' to '-', drop a trailing
    precision suffix - which keeps new models usable without silently
    asserting that two differently-named things are the same weights.
    """
    if not name:
        return None
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    key = name.lower().replace(":", "-")
    for suffix in _PRECISION_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key


def _migrate(conn):
    existing = {row["name"]
                for row in conn.execute("PRAGMA table_info(inference_runs)")}
    for name, coltype in MIGRATIONS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE inference_runs ADD COLUMN {name} {coltype}")
            print(f"[migration] added column {name}")
    conn.commit()


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row      # rows behave like dicts
    conn.executescript(SCHEMA)
    _migrate(conn)                      # existing databases get new columns too
    return conn


def record_run(conn, **fields):
    """
    Insert one run. Any column left out of **fields stays NULL, which is the
    whole point - we never invent a value we did not measure.
    """
    fields.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    # Derived here rather than in each harness, so no caller can forget it and
    # no caller can disagree with another about what the same model is called.
    if fields.get("model") and not fields.get("model_key"):
        fields["model_key"] = canonical_model(fields["model"])
    columns = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    cursor = conn.execute(
        f"INSERT INTO inference_runs ({columns}) VALUES ({placeholders})",
        list(fields.values()),
    )
    conn.commit()
    return cursor.lastrowid


def record_rows(conn, table, rows):
    """Insert many rows into one of the Phase 6 tables in one transaction.
    The table name is code-supplied, never user input."""
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        row.setdefault("timestamp_utc", now)
        if row.get("model") and not row.get("model_key"):
            row["model_key"] = canonical_model(row["model"])
    columns = list(rows[0])
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        [[row.get(c) for c in columns] for row in rows])
    conn.commit()


if __name__ == "__main__":
    conn = connect()
    print(f"database: {os.path.abspath(DB_PATH)}")

    run_id = record_run(
        conn,
        run_group="selftest",
        trial_index=0,
        backend="ollama",
        model="qwen2.5:1.5b",
        quantization="Q4_K_M",
        context_length=4096,
        temperature=0.0,
        streaming=1,
        input_tokens=41,
        output_tokens=73,
        decode_tokens_per_s=225.1,
        notes="synthetic row, gpu_* deliberately left NULL",
    )
    print(f"inserted row id={run_id}")

    row = conn.execute(
        "SELECT * FROM inference_runs WHERE id = ?", (run_id,)
    ).fetchone()

    print("\nstored values (None = not measured, and that is the correct answer):")
    for key in row.keys():
        print(f"  {key:22s} {row[key]}")

    conn.execute("DELETE FROM inference_runs WHERE run_group = 'selftest'")
    conn.commit()
    print("\nself-test row removed. schema is ready.")

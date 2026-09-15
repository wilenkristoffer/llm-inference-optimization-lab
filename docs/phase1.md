# Phase 1 - Ollama baseline

Technical notes. What was built, what was measured, what turned out to be wrong.

Date: 2026-09-11
Hardware: Ryzen 5600X / RX 7800 XT (16 GB, ~624 GB/s) / 32 GB DDR4 / Windows 11
Stack: Ollama 0.34.0 (ROCm), Python 3.11.9, SQLite. No third-party packages.

---

## Goal

Establish a trustworthy performance baseline for local LLM inference before
optimizing anything, and understand what an inference request actually is.

"Trustworthy" is the operative word. A single latency number is not a baseline.
Phase 1 is mostly about learning which numbers are stable enough to compare.

---

## What was built

| File | Purpose |
|---|---|
| `01_raw_request.py` | One non-streaming request. Prints the raw JSON so the server's own timing fields are visible. |
| `02_streaming_ttft.py` | Streaming request. Measures time to first token from the client, pins `temperature=0`. |
| `03_repeated_trials.py` | Warm-up + N trials, reports median and spread. Exposes `run_once()` for reuse. |
| `04_storage.py` | SQLite schema (`inference_runs`) and `record_run()`. |
| `05_benchmark.py` | The harness: reads model metadata from the API, runs trials, persists one row per trial. |

Run a configuration and store it:

```
python phase1-ollama-baseline\05_benchmark.py --model qwen2.5:7b --trials 5
python phase1-ollama-baseline\05_benchmark.py --history
```

---

## Concepts

### Client and server are roles, not machines

Ollama runs as a background process listening on `127.0.0.1:11434`. It starts
independently, holds models in VRAM, and outlives any client. `ollama run
llama3` is itself just a client of that server - it does not load or execute
the model, it POSTs to the same endpoint our Python does.

A separate runner process (llama.cpp) actually holds the weights and does the
math; the API process routes to it.

**Running vs serving:**

```
running   load weights, forward pass, exit. one process, one user.
serving   weights resident, API exposed, many clients, queueing and
          memory managed across them.
```

Batching, KV cache management, admission control and paged attention are all
problems that only exist in the serving case.

### Prefill and decode are different workloads

```
prefill   all prompt tokens processed together. compute-bound, parallel.
          determines TTFT.
decode    one token at a time, each depending on the last. cannot be
          parallelized within a request. memory-bandwidth-bound.
          determines tokens/sec.
```

Every token of decode requires streaming the entire model's weights out of
VRAM. That is why decode speed tracks model size and memory bandwidth, not
compute throughput.

Ollama reports these separately: `prompt_eval_duration` (prefill) and
`eval_duration` (decode), both in nanoseconds.

### Streaming

The model always generates one token at a time. Streaming does not change
inference - it changes whether tokens are delivered as produced or buffered
until complete. Total time to the last token is identical; time to the first
collapses.

Costs: you cannot validate output before the user sees it, and structured
output (JSON) needs buffering or an incremental parser.

**TTFT can only be measured client-side.** The server does not know when its
bytes arrived. Any benchmark reporting only server-side timings is blind to
everything between the two.

### KV cache

Each token, at each layer, produces Query, Key and Value vectors. Attention
compares the current token's Query against every previous token's Key, then
mixes the Values by the resulting weights.

K and V for token `i` depend only on tokens `0..i` and never change as more
tokens are appended - causal masking guarantees it. So they are cached. Q is
not cached; only the current token's Query is ever needed.

Without it, generation would be quadratic in sequence length.

**Size:** `2 (K,V) x layers x kv_heads x head_dim x bytes x tokens`

For qwen2.5:1.5b (28 layers, 2 KV heads, 128 dim, FP16) that is ~28 KB/token:

```
 4,096 context   ~115 MB
32,768 context   ~917 MB     roughly the size of the model itself
```

Measured directly via `ollama ps`: SIZE went 1.2 GB -> 2.1 GB when `num_ctx`
was raised from 4096 to 32768. Weights did not change.

**Grouped Query Attention (GQA)** exists to shrink this. qwen2.5:1.5b has 12
query heads but only 2 KV heads. Without GQA the cache would be 6x larger -
5.5 GB at full context for a 1 GB model.

**Prefix caching** reuses the cache across requests when the prompt prefix is
byte-identical. Observed: `prompt_eval_cached_count` went 0 -> 40 of 41 tokens
on a repeat, and prefill dropped 42.4 ms -> 7.5 ms. The last token is always
recomputed, since a live forward pass is needed to produce the next
distribution.

Practical consequence: put stable content (system prompt) first and varying
content last. A timestamp at the top of a system prompt destroys the cache on
every request.

### Context length

`ollama show` reports qwen2.5:1.5b supports 32,768 tokens. Ollama loads it at
**4,096** by default (`num_ctx`). Exceeding it does not error - older tokens
are silently discarded.

In an agent loop, the oldest tokens are the system prompt and tool definitions.
This is a likely explanation for agents that "work for several turns then go
off the rails".

Detect it by comparing what you sent against `prompt_eval_count`. Fix it with
`options.num_ctx`, a Modelfile `PARAMETER`, or `OLLAMA_CONTEXT_LENGTH`. Most
third-party frameworks never set it and inherit the default.

Raising it is not free: VRAM scales linearly, prefill takes longer, and model
quality degrades well before the advertised limit.

---

## Measurements

All qwen2.5, Q4_K_M, `num_ctx=4096`, `temperature=0`, warm, 5-7 trials each.

```
model    size     decode      TTFT     effective BW    % of 624 GB/s
1.5b    0.99 GB   222.1 t/s   13.3 ms    219 GB/s          35%
3b      1.9 GB    129.4 t/s   16.7 ms    246 GB/s          39%
7b      4.7 GB     83.4 t/s   20.7 ms    392 GB/s          63%
14b     9.0 GB     44.3 t/s   34.8 ms    399 GB/s          64%
```

`effective BW = model size x decode rate` - what the GPU actually achieved
against its 624 GB/s peak.

**Noise floor: ~1-2% cv on decode rate.** Anything moving more than ~5% is a
real effect. This threshold is what makes later comparisons defensible.

**Larger models use the GPU better.** 14b is 9.1x the weights of 1.5b but only
5.0x slower. Fixed per-token costs (kernel launch, attention, sampling)
amortize across a longer forward pass. Efficiency plateaus near 63-64% here.

**Cold start:** first request pays model loading - 1.68 s for the 1.5b, ~3 s
for larger. Warm: ~4 ms. Ollama unloads after 5 minutes idle (`keep_alive`).

**Determinism:** `temperature=0` produces byte-identical output across runs.
With default sampling, the same prompt returned 64 then 86 tokens. Nothing is
comparable without pinning this.

---

## Hypotheses tested

### Confirmed: `localhost` costs ~2 seconds per request

Client total exceeded server total by a **constant** ~2035 ms regardless of
workload. Constant, not proportional - so it is setup cost, not per-token
overhead. That shape is what identified it.

Ollama listens on IPv4 only. Windows resolves `localhost` to `::1` first, the
connection fails, and the client waits before falling back to IPv4.

```
localhost    TTFT 2056 ms    gap 2036 ms
127.0.0.1    TTFT   18 ms    gap    5.4 ms
```

The 5.4 ms is the genuine HTTP + JSON cost. The 2 seconds was invisible to
server-side timings - Ollama's `total_duration` starts after the connection is
established, so the system looked perfectly healthy.

**Diagnostic technique worth keeping: unexplained time that is constant across
workloads is setup; unexplained time that scales is per-unit work.**

### Refuted: co-resident models contend for bandwidth

Decode was ~204 t/s while `llama3:latest` was also in VRAM, versus ~225 t/s
alone. Tested directly:

```
llama3 not loaded    225.1 t/s  (219.5-227.0)
llama3 co-resident   221.6 t/s  (212.6-224.5)
```

1.6% - within noise. An idle co-resident model occupies VRAM but does not touch
the memory bus. Hypothesis rejected.

### Supported but unproven: GPU clock ramp (DVFS)

The remaining difference was methodology. Isolated single-shot runs read ~205
t/s; sustained back-to-back trials read ~222.

```
1.5b   isolated 204.8    sustained 222.1     ~8% deficit
14b    isolated  44.5    sustained  44.3     no deficit
```

A 1.5b run finishes in ~350 ms, much of it while the GPU ramps from idle
clocks. A 14b run takes ~2.3 s, long enough for clocks to reach boost early.
The theory predicts the effect shrinks with run duration, and it does.

Cannot be confirmed - no GPU clock telemetry available (see limitations).

**Lesson: measurement methodology is itself a variable.** Warm-up plus
back-to-back trials measures sustained throughput. One-shot requests measure
what a user hitting an idle server gets. Both are valid and they are not the
same number.

---

## Two corrections

**1. The prefill throughput calculation was wrong.** It divided
`prompt_eval_count / prompt_eval_duration`, but on a cache hit most of those
tokens were never processed. It reported 5,450 t/s for what was actually one
token in 7.5 ms. Fixed to use `prompt_eval_count - prompt_eval_cached_count`.

This is the class of error that produces impressive, wrong benchmarks.

**2. "The 1.5b model lacks capacity to know what a KV cache is" was wrong.**

All four models answered the prompt "Explain what a KV cache is" by describing
Redis. Scaling 9x changed nothing - so it was not capacity, it was **prompt
ambiguity**. "KV cache" means key-value store in the overwhelming majority of
training data.

Rephrased to "In a transformer language model, what is the KV cache?", both the
1.5b and the 14b answer correctly. The size difference shows as precision, not
correctness:

```
1.5b   "store and reuse intermediate results"                   correct, vague
14b    "key and value components of the self-attention mechanism
        for previous time steps... without recomputation"       correct, exact
```

Consequence for Phase 6: the benchmark prompt is fine for measuring speed and
useless for measuring quality. Quality probes need unambiguous, verifiable
answers - and a correct/incorrect score would rate the two answers above as
identical, so scoring has to capture precision too.

---

## Hardware limitations hit

- **No GPU telemetry.** `nvidia-smi` has no AMD/Windows equivalent available
  here. `rocm-smi` ships with ROCm on Linux. So GPU utilization, VRAM usage,
  temperature, power and clock speed cannot be read programmatically. All
  `gpu_*` columns in the schema stay NULL. This is why the DVFS hypothesis
  stays unproven, and it is a real gap for Phase 4.
- **VRAM ceiling 16 GB.** `qwen2.5:32b` (19 GB) cannot fit. Deliberately useful
  later for observing CPU/GPU offload.
- Confirmed working: ROCm drives the RX 7800 XT correctly. `ollama ps` reports
  `100% GPU`, and decode at 222 t/s is 4x above what DDR4-3200 could physically
  deliver - so GPU execution was provable from the numbers alone.

---

## Schema notes

`inference_runs`, in `lab.db`. Design rules:

- **Nothing is NOT NULL except identity.** An unmeasured field stores NULL,
  never 0. A zero in `gpu_percent` would silently drag down every average.
- **`run_group` + `trial_index`.** Comparisons are between medians of groups,
  never single rows - a direct consequence of the measured noise floor.
- **Client and server timings stored separately.** Collapsing them into one
  "latency" column would have hidden the IPv6 stall permanently.
- **`response` text is stored.** Output quality is a result, not a side effect.
- **`model_digest` alongside `model`.** Tags like `:latest` get repointed;
  comparing `latest` from March against `latest` from September is comparing
  two models and reporting it as one.
- **`cached_input_tokens`.** Without it, prefill throughput cannot be computed
  correctly (see correction 1).

---

## What would differ in production

- Ollama's per-request `num_ctx` forces a model reload. Production servers
  (vLLM, TGI) fix context at startup and never renegotiate.
- Ollama holds multiple models and unloads on idle, trading VRAM for cold-start
  latency. vLLM pins one model for the process lifetime.
- Benchmarks here are single-stream. Production cares about throughput under
  concurrency, where continuous batching changes the picture entirely (Phase 9).
- Real serving needs p50/p95/p99 latency, not medians of seven trials.
- Model loading would be from a warm local volume or cache, not cold disk.

---

## Open items

- Verify the context-truncation theory against a real over-length prompt, and
  determine which end Ollama discards (Phase 5).
- Find any usable AMD GPU telemetry path on Windows, or accept the gap and
  document it (Phase 4).
- Design quality probes with verifiable answers (Phase 6).

---

## Terminology

| Term | Meaning |
|---|---|
| TTFT | Time to first token. Client-side. Dominated by prefill. |
| Prefill | Processing all prompt tokens. Compute-bound, parallel. |
| Decode | Generating tokens one at a time. Bandwidth-bound, sequential. |
| Inter-token latency | Gap between consecutive tokens. Reciprocal of decode rate. |
| KV cache | Cached per-token Key/Value tensors, per layer, so attention does not recompute the prefix. |
| GQA | Grouped Query Attention. Multiple query heads share a KV head, shrinking the cache. |
| `num_ctx` | Context window size. Drives KV cache VRAM linearly. |
| Prefix caching | Reusing KV cache across requests with identical prompt prefixes. |
| Cold start | First request after load, paying model-into-VRAM cost. |
| cv | Coefficient of variation. stdev as % of mean. The noise figure. |
| Q4_K_M | A 4-bit quantization format (Phase 6). |
| DVFS | Dynamic voltage and frequency scaling. GPU clock ramping under load. |

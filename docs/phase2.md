# Phase 2 - Model serving

Technical notes. The API surface, the chat template, and what happens when
requests arrive at the same time.

Date: 2026-09-14
Stack: Ollama 0.34.0, Python 3.11.9 stdlib only.

---

## Goal

Understand Ollama *as a server* rather than as a way to run a model, and
measure what it does under concurrent load. The concurrency result is what
gives Phase 3 a concrete baseline instead of a theoretical argument.

---

## What was built

| File | Purpose |
|---|---|
| `01_api_surface.py` | Four ways to send the same question. Compares token counts, stop reasons, and maps Ollama's fields to OpenAI's. |
| `02_concurrency.py` | Fires N simultaneous requests, measures aggregate throughput against per-request latency. |

Also: `run_once()` in `phase1-ollama-baseline/03_repeated_trials.py` gained a
`num_predict` passthrough, so every request in a concurrency sweep generates
exactly the same number of tokens and the throughput maths is clean.

---

## The API surface

```
/api/generate            one prompt string
/api/chat                a list of role/content messages
/v1/chat/completions     the same as /api/chat, in OpenAI's shape
/api/show, /api/tags, /api/ps, /api/version    introspection
```

### All three inference endpoints apply the chat template

This was the surprise, and it contradicted an earlier assumption in this
project. Despite the name, `/api/generate` is **not** a raw completion
endpoint - Ollama renders your string into the model's template as well. Only
`"raw": true` opts out.

Measured on qwen2.5:1.5b with "What is the capital of France? Answer in one
word.":

```
endpoint                       in   out    stop
/api/generate                  41     2    stop
/api/chat                      41     2    stop
/v1/chat/completions           41     2    stop
/api/generate  "raw": true     12    60    length
```

The 29-token difference is qwen2.5's default system prompt plus ChatML role
markers, paid in prefill on **every request**. In an agent loop with tool
definitions in the system prompt this is hundreds of tokens per call - and it
is also the most cacheable part, since it never changes.

### The template also supplies the stop token

The `raw: true` row above never stopped. It answered correctly and then looped:

```
'Paris. The capital of France is Paris. Paris is the capital of France.
 Paris is the capital of France. Paris is the ca...'
```

`done_reason: length`, not `stop` - it only halted at `num_predict: 60`.

The reason: during instruction tuning the model learned `emit <|im_end|> when
the assistant turn is complete`. That behaviour is conditioned on being inside
`<|im_start|>assistant`. Strip the frame and the condition never fires - the
model reverts to plain text continuation, which has no natural end.

This is a useful way to see what instruction tuning actually is. The base model
only continues text. "Answering and then stopping" is a format convention
layered on top.

Consequence: in raw mode you must supply your own stopping rule
(`options.stop`) and always cap `num_predict`, or a request can run until it
fills the context window.

### The template is string concatenation

The template shown by `/api/show` is a Go text/template with variables
`.Messages`, `.System`, `.Tools`. It renders JSON messages into one flat
string. The model never sees JSON and has no concept of a "role" field -
`<|im_start|>` and `<|im_end|>` are simply tokens in its vocabulary.

Proved by hand-writing the ChatML for the same question and sending it with
`raw: true`: identical 41-token count as `/api/chat`.

### The OpenAI-compatible endpoint

`/v1/chat/completions` is a **specification**, not a service. Ollama implements
it locally; nothing reaches OpenAI. Verified by sending
`Authorization: Bearer sk-this-key-is-completely-made-up`, which was accepted -
there is no account, no billing, no remote call. `/v1/models` returns local
models with `owned_by: library`, and response ids are a local counter
(`chatcmpl-302`) rather than OpenAI's random strings.

Field mapping:

```
prompt_eval_count             -> usage.prompt_tokens
eval_count                    -> usage.completion_tokens
response / message.content    -> choices[0].message.content
done_reason                   -> choices[0].finish_reason
prompt_eval_duration          -> absent
eval_duration                 -> absent
total_duration                -> absent
```

**The OpenAI schema carries no timing at all.** On that interface every latency
figure must be measured client-side. That is not an Ollama limitation, it is
the standard - and it means the client-side timing work from Phase 1 becomes
the only option rather than a nice-to-have.

Caveat: "compatible" means a common subset, and unsupported parameters are
often silently ignored rather than rejected. When a parameter matters for an
experiment, verify it took effect.

---

## Concurrency

`02_concurrency.py` fires N requests simultaneously behind a
`threading.Barrier` (without it, thread startup staggers them and you measure
sequential requests while calling it concurrency). Distinct prompts, so we
measure batching rather than KV prefix caching. `num_predict=100` fixed.

### Default: Ollama queues

`OLLAMA_NUM_PARALLEL` unset, qwen2.5:1.5b:

```
conc  wall s  aggregate  per-req   ttft ms   speedup
  1    0.14     196.5     229.9      18.8     1.00x
  2    0.25     200.3     231.7      76.4     1.02x
  4    0.45     197.3     231.0     200.1     1.00x
  8    0.88     197.6     232.8     427.3     1.01x
```

Wall time doubles with every doubling of load; aggregate throughput is flat;
per-request rate never drops. That is serial execution. 8 requests x ~110 ms
of decode = 880 ms, matching the 0.88 s measured exactly.

Rising TTFT is queue wait - the eighth request waits for seven others.

Terminology worth keeping straight:

```
concurrency   requests in flight at once     - yes, 8 were
parallelism   requests executing at once     - no, 1 was
```

### With OLLAMA_NUM_PARALLEL=4: batching

qwen2.5:1.5b:

```
conc  aggregate  per-req   ttft ms   speedup
  1     195.7     232.5      20.7     1.00x
  2     287.0     176.3      21.1     1.47x
  4     342.1     126.3      45.5     1.75x
  8     327.9     119.0     127.0     1.68x   <- plateau, 2 batches of 4
```

qwen2.5:14b:

```
  1      42.4      46.8      62.8     1.00x
  2      60.6      38.8     105.8     1.43x
  4      80.4      25.0     191.5     1.90x
```

TTFT at concurrency 4 fell from 200 ms (queueing) to 45 ms - nobody waits in
line any more. But per-request decode rate roughly halves. That trade is the
whole subject.

### Why it is sublinear

Batching only amortises the part of each token spent **moving weights**.
Attention, kernel launches and sampling are per-sequence and do not share.

Fitting a model to the 14b data, where a batch step costs `W` (shared weight
streaming) plus `O` per sequence:

```
conc 1   46.8 t/s -> 21.4 ms/step
         W = 9 GB / 624 GB/s = 14.4 ms
         O = 21.4 - 14.4     =  7.0 ms

                predicted W + N x O     measured
conc 2          28.4 ms                 25.8 ms
conc 4          42.4 ms                 40.0 ms
```

Good fit, and the implication is blunt: **per-sequence overhead is barely
parallelising.** Four sequences cost roughly four times the overhead.

Ceiling implied by that model:

```
as concurrency -> infinity, aggregate -> 1/O = ~143 t/s   (3.4x)
at concurrency 4 we reach 80 t/s      = 56% of ceiling
```

Cross-check on the 1.5b, which batches worse (1.75x): only ~37% of its
per-token time is weight streaming (4.3 ms per token, 1.6 ms of it moving
0.99 GB), versus ~67% for the 14b. **Batching pays off more the larger the
model** - which is exactly why production serving of large models cares about
it so much.

Another confirmation that weights really are being shared: at concurrency 4 the
1.5b steps at 126/s, so weight traffic is 126 x 0.99 GB = **124 GB/s**, down
from 229 GB/s at concurrency 1. Less bandwidth consumed, more total work done.
The bottleneck has moved off bandwidth and onto per-token overhead.

### VRAM cost of concurrency

Weights load once and are shared. Only the KV cache is per-request:

```
KV VRAM = num_ctx  x  OLLAMA_NUM_PARALLEL  x  bytes_per_token
```

qwen2.5:1.5b at 4096 context (~28 KB/token, ~115 MB per slot):

```
conc   weights   KV total   total
  1     986 MB     115 MB   1.1 GB
  4     986 MB     460 MB   1.4 GB
  8     986 MB     920 MB   1.9 GB
```

8x concurrency for 1.7x memory. It flips for big-KV models - llama3.1:8b costs
128 KB/token, so 8 slots at 4096 context is 4 GB of KV against 4.9 GB of
weights. **Concurrency limits in real serving are usually KV cache limits, not
weight limits.**

Critically, Ollama **preallocates** these slots at model load, sized by
`num_ctx x num_parallel`, whether used or not. Confirmed in Phase 1: raising
`num_ctx` from 4096 to 32768 moved `ollama ps` SIZE from 1.2 GB to 2.1 GB
before any long prompt was ever sent.

The trap:

```
llama3.1:8b, num_ctx 32768, NUM_PARALLEL 4
  = 32768 x 4 x 128 KB = 16 GB of KV + 4.9 GB weights = 21 GB
  on a 16 GB card: will not fit
```

None of that is about how much context you *use*. It is about how much you
*declare*.

---

## What this sets up for Phase 3

`O = 7 ms` of per-sequence overhead is not a law of physics. It is llama.cpp's
batching implementation: attention computed per-sequence, matmuls staying
GEMV-shaped instead of becoming efficient GEMMs when batched.

vLLM's design attacks exactly this:

- **Continuous batching** - new requests join the running batch immediately
  instead of waiting for the current one to drain.
- **PagedAttention** - KV cache allocated in small blocks on demand, like OS
  virtual memory pages, instead of reserving full context per slot. A
  200-token conversation occupies 200 tokens of VRAM, not 4096.
- **Batched attention kernels** written for the many-sequence case.

**Measured baseline to hold vLLM against: 1.90x at concurrency 4 on a 14B
model, with per-request rate falling 46.8 -> 25.0 t/s.** If vLLM cannot beat
that on this card, its advantages are theoretical for this hardware - which
would be a legitimate finding.

---

## Corrections made in this phase

1. **"`/api/generate` is a raw completion endpoint"** - wrong. It templates by
   default like the others. Caught because the template-overhead calculation
   printed 0 instead of a positive number.

2. **Batching speedup predictions were too optimistic three times** (predicted
   500-700 t/s at conc 4 on the 1.5b, got 342; predicted 2.5x on the 14b, got
   1.90x). The error each time was assuming shared weight movement dominates
   per-token cost. It does not on small models, and per-sequence overhead does
   not parallelise as well as assumed.

---

## Terminology

| Term | Meaning |
|---|---|
| Chat template | Markup (here ChatML) that wraps messages into the flat string the model was tuned on. Also supplies the stop token. |
| ChatML | `<|im_start|>role ... <|im_end|>` format used by Qwen and others. |
| Batching | Running several sequences through the model together so weight reads are shared. |
| Continuous batching | Letting new requests join an in-flight batch rather than waiting for it to finish. |
| PagedAttention | Allocating KV cache in small on-demand blocks instead of reserving full context per slot. |
| Aggregate throughput | Total tokens/sec across all requests. The capacity-planning number. |
| Per-request rate | Tokens/sec one user experiences. What batching trades away. |
| `OLLAMA_NUM_PARALLEL` | How many requests Ollama will execute simultaneously. Default 1. Multiplies preallocated KV cache. |
| GEMV / GEMM | Matrix-vector vs matrix-matrix multiply. Batching should turn the former into the latter; that is where the efficiency comes from. |

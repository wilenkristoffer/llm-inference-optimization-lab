# Phase 8 - Docker and Compose

Technical notes. How the lab's stack runs in containers on this machine
(Windows 11, Docker Desktop on WSL2, RX 7800 XT), how a container reaches an
AMD GPU without `/dev/kfd`, and what the containers cost compared with the
native setup from Phases 1-7.

Date: 2026-10-23. Docker Desktop 29.9.x (WSL2 backend, containerd image
store), Compose 5.6.1, vLLM 0.29.0+rocm723, ROCm 7.2.4, ROCDXG 1.2.2.
Plan: `phase8-plan.md`. Code: `phase8-docker/`.

---

## Goal

Run the lab with `docker compose up`, and measure the cost of doing so:

1. **GPU in a container (go/no-go)** - under WSL there is no `/dev/kfd` or
   `/dev/dri`, so the usual ROCm container recipe does not apply.
2. **The CPU-only parts** - the Phase 4 API and UI.
3. **vLLM in a container** - the same build as the native `~/vllm-env`.
4. **Compose** - one command, with healthchecks and start order.
5. **Container cost** - decode, startup, request latency, image size,
   measured native and container interleaved in one session.

---

## Results against the predictions

| metric | prediction | measured | verdict |
|---|---|---|---|
| GPU reachable through `/dev/dxg` | yes, after 1-2 library problems | yes, after 2 library-path problems | right |
| GPU compute in a container | - | fp16 matmul 54.0 vs native 52.3 TFLOP/s (noise ±6%) | no cost |
| vLLM decode, batch 1 | within 2% | -1.2% (145.3 vs 147.0 tok/s) | right |
| vLLM decode, batch 32 | within 2% | -1.8% (1984 vs 2020 tok/s) | right |
| vLLM warm start | about equal to native | 53.8 s vs 43.1 s (+10.7 s) | wrong |
| vLLM first start | +5-15 s | 105-113 s (+60-70 s) on an empty cache volume | wrong |
| request latency through the port | < 1 ms added | +0.24 ms from WSL; no change in the median from Windows | right |
| image size | 15-25 GB | vLLM 45.8 GB on disk (10.9 GB content) | wrong |

The steady state (decode) is within the prediction. Everything that happens
once per container (startup, first start, image size) costs more than
predicted, for reasons that sections 03 and 04 trace.

---

## 01 - A GPU in a container, without /dev/kfd

On bare-metal Linux a ROCm container gets `/dev/kfd` (the compute driver)
and `/dev/dri` (render nodes). WSL2 has neither. The GPU belongs to the
Windows driver, and Linux reaches it through one paravirtual device:

```
container process (torch / vLLM)
    |  HIP runtime  ->  libhsa-runtime64
    |  libhsakmt (ROCDXG build)      /opt/rocm-wsl/lib   from the ROCDXG debs
    |  librocdxg                     translates KFD calls to D3DKMT calls
    |  libdxcore                     /usr/lib/wsl/lib    bind-mounted from the VM
/dev/dxg                             the WSL GPU paravirtualisation device
    |  VMBus
Windows: dxgkrnl -> AMD WDDM driver (the same driver Ollama uses)
```

Docker Desktop's VM has the same `/dev/dxg` and the same `/usr/lib/wsl` as
the Ubuntu distro. So a container needs three things:

- `--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro`
- The ROCDXG debs installed in the image. `librocdxg.so` is not in
  `/usr/lib/wsl`; it ships with ROCDXG.
- Three env vars:
  - `HSA_ENABLE_DXG_DETECTION=1`
  - `LD_PRELOAD` of the `norocprof` stub, because rocprofiler-sdk aborts
    without `/sys/class/kfd`
  - `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm-wsl/lib:/opt/rocm/lib`

The order of `LD_LIBRARY_PATH` is the whole Stage 0 story:
1. **Without `/usr/lib/wsl/lib`**, the runtime fails with
   `Failed to load libdxcore.so`.
2. **Without `/opt/rocm-wsl/lib` before `/opt/rocm/lib`**, the container
   loads the base image's bare-metal `libhsakmt`/`libamd_smi`. torch then
   fails with `undefined symbol: amdsmi_set_cpu_rail_isofreq_policy`.

`SYS_PTRACE` and `seccomp=unconfined` from the ROCDXG README were not needed.
The official `ollama/ollama:rocm` image expects `/dev/kfd`, so it can't work
here. That is why Ollama stays native.

fp16 4096x4096 matmul, 3 interleaved rounds:

| run | TFLOP/s | mean |
|---|---|---|
| native WSL, torch 2.14 | 50.2 / 56.3 / 50.4 | 52.3 |
| container, same torch 2.14 binary | 54.3 / 56.5 / 51.1 | 54.0 |
| container, image's torch 2.10 | 46.2 / 44.7 / 46.0 | 45.6 |

The container adds no GPU cost. A container doesn't add work on the GPU: the
kernels are the same and they go to the same device. The 2.10 row is the
lesson: **the torch build matters more than the container**, so the vLLM
image has to repeat the native versions exactly.

---

## 02 - The CPU-only parts: API and UI

| image | base | size |
|---|---|---|
| `lab/api` | `python:3.12-slim`, stdlib only | 177 MB |
| `lab/ui` | Node 24 build stage -> `nginx:1.29-alpine` | 93 MB |

Changes to `phase4-observability/`. All are env vars, and the native defaults
are unchanged:

- `LAB_DB`, `API_HOST`: where the database is, and what to bind. The API now
  opens lab.db read-only (`mode=ro`), natively too, since it never writes.
- `OLLAMA_URL`, `VLLM_URL`: health-check targets.
- `LAB_SERVER_CONTROL=0`: a container can't start processes on the host,
  so start/stop return 409 and the UI disables the buttons with a note.
- `VLLM_WHERE`: the UI's location label ("WSL2" natively, "container"
  under Compose).

Two findings:
- **`host.docker.internal` reaches native Ollama**, even though Ollama binds
  `127.0.0.1` only. Docker Desktop proxies that name to the host's loopback.
- **lab.db stays read-only in containers.** SQLite's file locks don't cross
  the Windows/VM file-sharing boundary. A dashboard that polls can live with a
  rare torn read; a writer would corrupt the database. Rows written natively
  showed up in the container at once.

---

## 03 - The vLLM image

`phase8-docker/vllm/Dockerfile` repeats the native `~/vllm-env` recipe:
- the `rocm/dev-ubuntu-24.04:7.2.4-complete` base, which has the same ROCm
  userspace as native;
- the ROCDXG debs and the compiled `norocprof` stub;
- a venv installed from `requirements.lock`, the native venv's exact
  versions, regenerated by `freeze.sh`;
- the env vars from 01.

The serve flags are the image's `CMD`. They match `VLLM_GUEST_CMD` in
`servers.py`.

Five problems, each fixed at its cause:

1. **The wheel index moves.** `wheels.vllm.ai/rocm/` serves only the latest
   nightly (torch 2.13 and aiter 0.1.24 by the build date), so a lock from a
   month earlier didn't resolve. The fix is the pinned per-release index,
   `wheels.vllm.ai/rocm/0.29.0/rocm723/`.
2. **`pip freeze` reported the wrong amdsmi** (1.2.2 instead of 26.2.2). The
   ROCDXG deb installs a `.pth` that puts `/opt/rocm-wsl/share/amd_smi` on
   `sys.path`, and freeze reported that copy. The fix: `freeze --path` reads
   only the venv's own site-packages.
3. **The base image was downloaded twice (7 GB).** A base pulled only by
   `FROM` lives in the build cache, and the cache is trimmed (see 06). The fix
   is to `docker pull` the base once, which makes it a tagged image that cache
   cleanup never removes. The pip step also has a cache mount for wheels.
4. **torch couldn't load `libmpi_cxx.so.40`.** The torch wheel links to
   system libraries that it doesn't ship: OpenMPI, TBB, libjpeg, libdw. Native
   Ubuntu had them from earlier installs, the ROCm base doesn't. They're
   installed in their own step after pip, so that rebuild took 31 s.
5. **Every fresh container compiled a C++ addon.** tvm-ffi builds
   `libtorch_c_dlpack_addon_torch212-rocm.so` on first import. It does that
   only when a GPU is present, so it can't be done at build time. The result
   goes to `~/.cache/tvm-ffi`, which persists natively and is thrown away
   with each container. That cost about 18 s per start. The fix is a named
   volume `lab-cache` on `/root/.cache`, which now also holds vLLM's
   torch.compile cache and comgr's kernel cache. Warm start went from 75.6 s
   to 52-57 s.

Problem 4 is the general lesson: **a native environment carries undeclared
dependencies**, installed long ago for something else. The first container
build is where they show up.

Build cost: the base pull took 659 s (once). The first full build took about
17 min: pip 630 s (mostly downloading), layer export 308 s.

---

## 04 - What the container costs

All comparisons are native vs container, interleaved in one session (3 rounds
each), because results drift about 4% between sessions.

### Decode: within 2%

`/v1/completions`, 256 forced tokens per request (`ignore_eos`), temperature
0, one warm-up round.

| | native (tok/s) | container (tok/s) | diff |
|---|---|---|---|
| batch 1 | 144.7 / 148.7 / 147.6, mean 147.0 | 144.5 / 144.0 / 147.3, mean 145.3 | -1.2% |
| batch 32 | 2002 / 2020 / 2037, mean 2020 | 1934 / 2000 / 2017, mean 1984 | -1.8% |

Decode is bandwidth-bound GPU work, and section 01 showed the GPU path costs
nothing. The container came out lower in all 6 pairs, though, so 1-2% looks
like a small real cost rather than pure noise. Its cause isn't traced. A
plausible candidate is CPU-side scheduling: vLLM's per-step Python overhead
runs in a cgroup.

### Startup: +11 s warm, about +65 s on first start

Time from launch to `/v1/models` answering:

| run | native | container |
|---|---|---|
| warm (caches filled) | 44.3 / 43.0 / 42.1, mean 43.1 s | 51.3 / 52.6 / 57.4, mean 53.8 s |
| first start (empty `lab-cache`) | - | 105-113 s |

On first start the container rebuilds everything the cache volume holds. The
largest parts are torch.compile (17.7 s) and the tvm-ffi addon (about 18 s).

Warm, about 1 s of the 10.7 s gap is import time: 10.2 s vs 9.0 s per process
with warm files, and two processes, the API server and the engine core. The
rest is not explained (see Open items).

### Request latency: the container adds under 1 ms

`GET /health` with a new TCP connection per request, 300 requests per round,
like the lab's clients:

| path | median | p90 |
|---|---|---|
| inside the vLLM container (no port) | 0.49 ms | 0.54-0.65 ms |
| api container -> vllm (Compose network) | 0.80 ms | 0.90 ms |
| WSL distro -> native vLLM | 0.59 ms | 0.65-0.74 ms |
| WSL distro -> container (published port) | 0.80-0.86 ms | 0.91-1.01 ms |
| Windows -> native vLLM (WSL localhost forwarding) | 1.77-1.90 ms | 14-15 ms |
| Windows -> container (published port) | 1.75-1.84 ms | 17-21 ms |

From WSL, the published port adds 0.24 ms. From Windows the median is the
same either way, because both paths cross from Windows into a VM. The p90
tail of 14-21 ms appears with native vLLM too, so it belongs to that
crossing, not to Docker. Against a generation of hundreds of milliseconds,
neither figure matters.

### Image size: 45.8 GB

| image | on disk |
|---|---|
| `rocm/dev-ubuntu-24.04:7.2.4-complete` (base) | 29.2 GB |
| `lab/vllm` | 45.8 GB (10.9 GB content on top of the base) |
| `rocm/pytorch` ROCm 7.2.4 (Stage 0 base) | 40.1 GB |

The prediction of 15-25 GB was wrong because the "complete" ROCm userspace
alone is 29 GB, with every math library for every GPU target. The venv adds
about 11 GB, most of it torch with its bundled kernels and aiter.

---

## 05 - Compose

`phase8-docker/compose.yaml`, project name `lab`:

```
vllm  (GPU: /dev/dxg + /usr/lib/wsl, HF cache :ro, lab-cache)  127.0.0.1:8000
  ^ service_healthy (/health, start_period 180 s)
api   (lab.db :ro, VLLM_URL=http://vllm:8000)                   127.0.0.1:8010
  ^ service_healthy (/api/summary)
ui    (nginx)                                                    127.0.0.1:8080
Ollama: native Windows, reached through host.docker.internal:11434
```

```
cd /mnt/h/iwow/Development/llm-inference-optimization-lab/phase8-docker
docker compose up -d --wait     # all three healthy in about 70 s, warm
docker compose down
```

- **Run it from Ubuntu.** The HF cache bind-mounts from
  `${HOME}/.cache/huggingface` in the distro; Windows has no such HOME.
  lab.db mounts from `/mnt/h` without problems.
- The ports are the native ones, so the native servers must be stopped first.
- The 180 s start period covers a first start. The healthcheck polls every
  2 s during it, so `--wait` returns as soon as vLLM is ready.
- `lab-cache` has a fixed `name:`. Without it, Compose would create
  `lab_lab-cache`, which would be empty and cost a first start.
- Verified end to end: `/api/servers` shows Ollama (native) and vLLM
  (container) ready, a chat request answers, and the UI returns 200.

---

## 06 - How rebuilds behave

Rebuild speed matters more than build speed in a lab. Every change to a step
reruns that step and every step after it, and here one step is a multi-GB
pip install.

| change | cost |
|---|---|
| serve flags (`command:`, `docker run ... <flags>`) | none, they are runtime arguments |
| env vars at run time | none |
| a step after pip (system libs, `ENV`) | seconds to about 30 s |
| `requirements.lock` | pip install (about 6 min with cached wheels) + export (about 4 min) |
| nothing, but the pip layer was evicted from the cache | the same as a lock change, about 11 min |

The last row happened during Stage 3, and its cause is the builder's
cleanup policy. Docker Desktop derives the policy from
`defaultKeepStorage: 20GB`, and `docker buildx inspect` shows two caps:

- **The whole build cache is capped at 20 GiB.** It was at 107 GB, and this
  one image's layers are about 11 GB, so the pip layer gets trimmed.
- **Cache mounts are capped at 2.76 GiB and kept 48 h.** The pip wheel cache
  is over that cap too, so after trimming the wheels would download again.

Two working rules follow:

- `docker compose up --build api` also rebuilds vllm, because api depends on
  it. To rebuild one service alone:
  `docker compose build api && docker compose up -d --no-deps api`.
- Raising `defaultKeepStorage` fixes the eviction, but it is a machine-wide
  setting and needs a Docker Desktop restart. It was left at 20 GB: the
  Phase 9 experiments change serve flags, not the image. Revisit when the
  vLLM or torch version changes.

---

## The code

| file | what |
|---|---|
| `phase8-docker/stage0/Dockerfile`, `gpu_check.py`, `norocprof.c` | the GPU go/no-go image and the rocprofiler stub |
| `phase8-docker/api/Dockerfile` | the Phase 4 API, stdlib only |
| `phase8-docker/ui/Dockerfile` (+ `.dockerignore`) | the UI, Node build -> nginx |
| `phase8-docker/vllm/Dockerfile` | vLLM, repeating `~/vllm-env` |
| `phase8-docker/vllm/requirements.lock`, `freeze.sh` | the native venv's exact versions, and how to regenerate them |
| `phase8-docker/compose.yaml` | the stack |

The ROCDXG debs in `stage0/` come from github.com/ROCm/librocdxg releases
(1.2.2).

---

## Limitations

- One model (qwen2.5-1.5b-instruct) for the vLLM comparisons. A larger model
  is more bandwidth-bound and would likely show an even smaller relative
  decode cost.
- 3 rounds per comparison. The 1-2% decode difference is consistent in
  direction but close to the run-to-run spread.
- The request latency was measured on `/health`, not on generation requests,
  and over new connections only. Keep-alive clients would see less.
- Two rounds of "Windows -> native" survived; the output of the first round
  was lost.
- Ollama was not containerised. Its ROCm image needs `/dev/kfd`, and the
  Vulkan-through-`dzn` route was not tried.

---

## Open items

- **The remaining ~10 s of warm-start gap.** Import time explains about 1 s.
  Candidates: model weights read through the HF bind mount (distro ext4 ->
  Docker VM), the comgr/torch.compile cache on a volume vs the native disk,
  or cgroup CPU scheduling during CUDA-graph capture. A per-phase timeline
  from both logs would split it.
- **The 1-2% decode difference.** Check if it holds on a 7B model, and if
  `--cpuset-cpus` or `--cpu-shares` change it.
- **Ollama in a container through Vulkan** (`dzn`, Mesa's D3D12 Vulkan
  driver). Optional. Phase 7 found Vulkan slower than ROCm natively, so the
  result would likely be "works, slower".
- **The Windows-side p90 tail (14-21 ms).** It's the same with native vLLM,
  so it belongs to WSL networking, not Docker. Worth knowing for any
  latency-sensitive client on Windows.

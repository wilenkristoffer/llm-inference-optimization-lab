# Phase 8 plan - Docker and Compose

Status: **complete 2026-10-23.** Write-up: `phase8.md`. The stage results
below were recorded as each stage finished; the predictions are kept as
written.

Goal: run the lab's stack (model servers, Phase 4 API and UI) with
`docker compose up`, and measure what containers cost compared with the native
setup from Phases 1-7.

Kept small on purpose. The one real unknown is whether a container can reach
the GPU at all on this machine, so that question comes first and decides
how the rest of the phase goes.

---

## Starting point

| piece | runs today | notes |
|---|---|---|
| Ollama | native Windows | ROCm 7.1 backend; use `127.0.0.1`, not `localhost` |
| vLLM 0.29.0+rocm723 | WSL2 Ubuntu venv `~/vllm-env` | needs ROCDXG, `HSA_ENABLE_DXG_DETECTION=1`, `norocprof.so` LD_PRELOAD |
| API + UI | native, `phase4-observability/` | `servers.py` starts and stops the model servers |
| lab.db | SQLite file in the repo | gitignored |
| Docker | Desktop 29.9.1, WSL2 backend | installed, not yet used by the lab |

---

## Stage 0 - Can a container see the GPU? (go/no-go)

Under WSL there is no `/dev/kfd` or `/dev/dri`. The GPU is reached through
`/dev/dxg` plus the driver libraries in `/usr/lib/wsl/lib`. vLLM's WSL setup
already depends on that path (ROCDXG). A container needs the same pieces passed
in. Command based on the ROCDXG README's own docker example:

```
docker run --rm -it \
  --device /dev/dxg \
  -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  --ipc=host --shm-size 8G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  <image with ROCm 7.2 + ROCDXG installed + torch rocm7.2> \
  python -c "import torch; print(torch.cuda.is_available())"
```

`librocdxg.so` is **not** in `/usr/lib/wsl`. It lives in the Ubuntu distro's
ROCm install, which Docker Desktop's VM can't see. So install the ROCDXG debs
in the image instead of bind-mounting the library. The README mounts
`libdxcore.so` to `/usr/lib/libdxcore.so`. If the runtime can't find it, add
`/usr/lib/wsl/lib` to `LD_LIBRARY_PATH`.

Cheapest first try: base the test image on `rocm/pytorch` (ROCm 7.2 tag) and
add the ROCDXG debs. Don't build ROCm from scratch.

Outcomes:

- **GPU visible** - continue with the full plan (vLLM in a container).
- **Not visible** - CPU-only parts go in containers and the model servers stay
  native. Compose then reaches them through `host.docker.internal`. Still a
  valid result: write down why it failed.

Prediction: works for vLLM via `/dev/dxg`, after one or two missing-library
problems. The official `ollama/ollama:rocm` image will **not** work, because it
expects `/dev/kfd`.

---

### Stage 0 result (2026-10-23): GO

Files in `phase8-docker/stage0/`. The image is `rocm/pytorch` ROCm 7.2.4 plus
the ROCDXG debs, the `norocprof` stub and three env vars. Run with only
`--device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro` (`SYS_PTRACE` and
`seccomp=unconfined` turned out not to be needed).

Two problems, both library paths:
1. `Failed to load libdxcore.so`: add `/usr/lib/wsl/lib` to `LD_LIBRARY_PATH`.
2. The ROCDXG debs install to **`/opt/rocm-wsl/lib`**, not `/opt/rocm/lib`.
   Native WSL has that path on `LD_LIBRARY_PATH`. Without it, the container
   loads the bare-metal `libamd_smi`/`libhsakmt`, and torch 2.14 fails with
   `undefined symbol: amdsmi_set_cpu_rail_isofreq_policy`. The order must be
   `/usr/lib/wsl/lib:/opt/rocm-wsl/lib:/opt/rocm/lib`.

fp16 4096 matmul, 3 interleaved rounds:

| run | TFLOP/s | mean |
|---|---|---|
| native WSL, torch 2.14 | 50.2 / 56.3 / 50.4 | 52.3 |
| container, same torch 2.14 binary | 54.3 / 56.5 / 51.1 | 54.0 |
| container, image's torch 2.10 | 46.2 / 44.7 / 46.0 | 45.6 |

The container adds no measurable GPU cost; the spread is run-to-run noise,
about ±6%. The 2.10 gap is the torch build itself. **Stage 2 must use the same
torch/vLLM build as `~/vllm-env`, not the base image's torch.**

Image size: the base image is 40.1 GB uncompressed (10.4 GB download), well
over the 15-25 GB prediction. Build on top of it: 132 s.

## Stage 1 - Containerise the CPU-only parts

- Dockerfile for the Phase 4 API and UI (Python slim base).
- `lab.db` as a bind mount, not baked into the image, so native scripts and
  containers write to the same database.
- Model server URLs come from env vars instead of hard-coded addresses, with
  native defaults unchanged.
- `servers.py` cannot start processes on the host from inside a container.
  Under Compose the containers handle that and the API only checks health.
  Keep the native path working.

### Stage 1 result (2026-10-23): done

Images: `lab/api` 177 MB (`python:3.12-slim`, stdlib only) and `lab/ui`
93 MB (Node build, served by nginx). Dockerfiles are in `phase8-docker/api/`
and `phase8-docker/ui/`; build commands are in each file's header.

```
docker run -d --name lab-api -p 127.0.0.1:8010:8010 -v <repo>\lab.db:/data/lab.db:ro lab/api
docker run -d --name lab-ui  -p 127.0.0.1:8080:80 lab/ui
```

Changes to `phase4-observability/`. The native defaults are unchanged, and
the native API was re-checked:
- `api.py`: `LAB_DB` and `API_HOST` env vars. Opens the database
  **read-only** (`mode=ro`), natively too, since the API never writes.
- `servers.py`: `OLLAMA_URL` and `VLLM_URL` for the health checks.
  `LAB_SERVER_CONTROL=0` turns start/stop off; POST then returns 409.
  `/api/servers` reports `control`.
- UI: Start/Stop are disabled with a one-line note when `control` is false.

Verified:
- Both native servers answer from a container through `host.docker.internal`.
  That includes Ollama, which binds `127.0.0.1` only, because Docker Desktop
  proxies to the host loopback.
- Rows written natively are visible in the container at once
  (1,373 -> 1,378, tested on a copy of lab.db).

Known limit: SQLite file locks don't cross the Windows/VM boundary, so a read
that lands in the middle of a native write could, rarely, fail or see a
partial page. That's acceptable for a dashboard that polls; it would not be
for a writer. Containers must never write to lab.db.

## Stage 2 - vLLM image (only if Stage 0 passed)

- Dockerfile that repeats the `~/vllm-env` recipe: ROCm 7.2.4 userspace,
  ROCDXG 1.2.2 debs, the vLLM ROCm wheel, the `norocprof.c` stub compiled in
  the image, and the env vars.
- HF model cache as a named volume or bind mount, so models are not downloaded
  again.
- Record image size and build time.

### Stage 2 result (2026-10-23): done

`phase8-docker/vllm/`: Dockerfile, `requirements.lock` (the native venv's
exact versions) and `freeze.sh`, which regenerates the lock. The base is
`rocm/dev-ubuntu-24.04:7.2.4-complete`, the same ROCm userspace as native.

```
docker run -d --name lab-vllm --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  --ipc=host --shm-size 8G -p 127.0.0.1:8000:8000 \
  -v /home/work/.cache/huggingface:/root/.cache/huggingface:ro \
  -v lab-cache:/root/.cache lab/vllm
```

Run from Ubuntu, so that the HF cache bind-mounts from the distro. Serve flags
are the image's CMD, the same as `VLLM_GUEST_CMD`. Override them at run time;
changing them needs no rebuild.

Problems, each fixed at the cause:
1. **The vLLM wheel index moves.** `wheels.vllm.ai/rocm/` serves only the
   latest nightly (torch 2.13 and aiter 0.1.24 by now), so the lock didn't
   resolve. The pinned index is `wheels.vllm.ai/rocm/0.29.0/rocm723/`.
2. **`pip freeze` reported the wrong amdsmi** (1.2.2 instead of 26.2.2). The
   ROCDXG deb ships a `.pth` that puts `/opt/rocm-wsl/share/amd_smi` on
   `sys.path`, and freeze read that copy. `freeze.sh` uses `--path` so it reads
   only the venv's own packages.
3. **The base image got re-downloaded (7 GB).** A base pulled only by `FROM`
   lives in the build cache, and Docker Desktop trims that to 20 GB
   (`defaultKeepStorage`); it was at 64 GB. `docker pull` the base once so
   it's a tagged image, which is never collected. The pip step uses a cache
   mount, so wheels survive rebuilds.
4. **torch couldn't load `libmpi_cxx.so.40`.** The torch wheel links to system
   libs that it doesn't ship and that the ROCm base doesn't have: OpenMPI, TBB,
   libjpeg, libdw (`libopenmpi3t64 libtbb12 libjpeg-turbo8 libdw1t64`). Native
   Ubuntu had them from earlier installs. They're installed in their own step
   after pip, so that rebuild took 31 s.
5. **Every fresh container compiled a C++ addon (about 18 s).** tvm-ffi
   builds `libtorch_c_dlpack_addon_torch212-rocm.so` on first import (only
   when a GPU is present, so it can't be baked in at build time) and caches it
   in `~/.cache/tvm-ffi`. Natively that persists. The named volume on
   `/root/.cache` now holds it, along with vLLM's torch.compile cache and
   comgr's kernel cache. Warm start went from 75 s to 52-55 s.

Image: 45.5 GB on disk, 10.9 GB content. Base pull 659 s. First full build
about 17 min (pip 630 s, mostly downloads; layer export 308 s). Rebuilds of
steps after pip take seconds.

Native vs container, interleaved in one session, 3 rounds each:

| metric | native | container | diff |
|---|---|---|---|
| warm start to `/v1/models` | 44.3 / 43.0 / 42.1, mean 43.1 s | 51.3 / 52.6 / 57.4, mean 53.8 s | +10.7 s |
| decode batch 1 (tok/s) | 144.7 / 148.7 / 147.6, mean 147.0 | 144.5 / 144.0 / 147.3, mean 145.3 | -1.2% |
| decode batch 32 (tok/s) | 2002 / 2020 / 2037, mean 2020 | 1934 / 2000 / 2017, mean 1984 | -1.8% |

Decode is within the 2% prediction. The container came out slightly lower in
all 6 pairs, so it may be a real, small cost rather than pure noise. The cold
first start (empty cache volume) is 105-113 s, mostly torch.compile (17.7 s)
and the addon build. The remaining ~11 s warm-start gap isn't explained yet:
per process, imports are only ~1 s slower with warm files.

## Stage 3 - Compose file

- Services: `vllm` (GPU), `api`, `ui`. Ollama is either `external` (native,
  through `host.docker.internal`) or a Vulkan container if the follow-up note
  below works out.
- Healthchecks, plus `depends_on: condition: service_healthy` so the API waits
  for vLLM to finish its ~37 s startup.
- Ports published on `127.0.0.1` only.

### Stage 3 result (2026-10-23): done

`phase8-docker/compose.yaml`, project name `lab`. Run it from Ubuntu:

```
cd /mnt/h/iwow/Development/llm-inference-optimization-lab/phase8-docker
docker compose up -d --wait     # UI on http://127.0.0.1:8080
```

- `vllm`: the Stage 2 run flags as Compose keys. The healthcheck is
  `/health` with a 180 s start period, since a cold start takes about 110 s.
- `api`: `depends_on: vllm: service_healthy`. It reaches vLLM as
  `http://vllm:8000` on the Compose network and Ollama through
  `host.docker.internal`. lab.db is mounted read-only.
- `ui`: `depends_on: api: service_healthy`. It's healthy once nginx serves `/`.
- Volume `lab-cache` has a fixed `name:`, so it's the same volume the Stage 2
  `docker run`s filled.
- Ports are the native ones (8000, 8010, 8080), so stop the native servers
  first.
- Run from Ubuntu because the HF cache bind-mounts from `${HOME}/.cache/huggingface`;
  Windows has no such HOME. lab.db mounts from `/mnt/h` without problems.

Verified: `up -d --wait` brings all three services to healthy in about 70 s
on a warm cache. `/api/servers` reports both servers ready, the chat request
answers "Stockholm", and the UI returns 200.

Change to `phase4-observability/servers.py`: a `VLLM_WHERE` env var (default
`WSL2`) for the UI's location label. Compose sets it to `container`.

**Gotcha: unchanged rebuilds of `lab/vllm` take about 11 min.** A
`docker compose up --build api` also builds vllm, because api depends on it.
Every step up to the pip install was cached, but the pip step re-ran: 362 s of
installing with zero downloads (422 wheels from the cache mount), then 247 s
of layer export and 83 s of unpacking. The cause is the builder GC policy
that Docker Desktop derives from `defaultKeepStorage: 20GB`:
`docker buildx inspect` shows the whole cache capped at 20 GiB (it's at
107 GB) and cache mounts capped at 2.76 GiB. The roughly 11 GB pip layer is
the first one trimmed, and the pip wheel cache is over its cap too. To fix it,
raise `defaultKeepStorage` (Docker Desktop > Settings > Docker Engine), then
restart Docker Desktop. Until then, rebuild one service alone:
`docker compose build api && docker compose up -d --no-deps api`.

## Stage 4 - Measure container cost

Run native and container interleaved in a single session, because results
drift about 4% between sessions.

| metric | native ref | prediction for container |
|---|---|---|
| vLLM decode tok/s, batch 1 and 32 | Phase 3 / batch-cliff runs | within 2% (GPU-bound, the container adds no GPU work) |
| vLLM cold start | ~37 s | +5-15 s the first time (image layers, volume reads), about equal after that |
| request latency through the published port | - | < 1 ms added |
| image size | - | 15-25 GB (ROCm userspace is large) |

Write up as `docs/phase8.md` in the same format as the earlier phases.

---

## Pre-checks (done 2026-10-23)

1. **Docker Desktop's VM exposes `/dev/dxg`: yes.**
   `wsl -d docker-desktop ls -l /dev/dxg` shows `crw-rw-rw- 10,125`. Its
   `/usr/lib/wsl` matches the Ubuntu distro's: the same three libs
   (`libdxcore`, `libd3d12`, `libd3d12core`) and the full Windows driver store,
   including the AMD `amdocl`/`amdogl` packages. Desktop is enough, so no
   Docker Engine install inside Ubuntu.
2. **No official image with WSL support.** `rocm/vllm` and `rocm/vllm-dev`
   are deprecated in favour of `vllm/vllm-openai-rocm` (on Docker Hub since
   2026-01-20). Its docs only show `/dev/kfd` + `/dev/dri` and don't mention
   WSL. The ROCDXG README runs `rocm/pytorch` with `/dev/dxg`, so start from
   that and add ROCDXG (see Stage 0). Stage 2 can try
   `vllm/vllm-openai-rocm` plus the ROCDXG debs before building our own image.
3. Ollama Vulkan in a container through `dzn`: not checked, still optional.
   Native Ollama behind `host.docker.internal` is the default.
4. Extra paths inside the container: the README's command adds `--ipc=host`,
   `--shm-size 8G`, `SYS_PTRACE` and `seccomp=unconfined`. These are now in the
   Stage 0 command. `LD_LIBRARY_PATH` only if the library fails to load.
5. **Disk is fine.** C: 237 GB free, H: 154 GB free. A 15-25 GB image fits.
   Docker Desktop's disk image lives on C: by default.

Sources: ROCDXG README (github.com/hlky/librocdxg); vLLM docker deployment
docs (docs.vllm.ai/en/v0.20.2/deployment/docker/).

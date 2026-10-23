# Phase 8 plan - Docker and Compose

Status: **planned 2026-10-22, not started.** This is a plan, not results.
Predictions are written down so the measurements can prove them wrong.

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

## Stage 2 - vLLM image (only if Stage 0 passed)

- Dockerfile that repeats the `~/vllm-env` recipe: ROCm 7.2.4 userspace,
  ROCDXG 1.2.2 debs, the vLLM ROCm wheel, the `norocprof.c` stub compiled in
  the image, and the env vars.
- HF model cache as a named volume or bind mount, so models are not downloaded
  again.
- Record image size and build time.

## Stage 3 - Compose file

- Services: `vllm` (GPU), `api`, `ui`. Ollama is either `external` (native,
  through `host.docker.internal`) or a Vulkan container if the follow-up note
  below works out.
- Healthchecks, plus `depends_on: condition: service_healthy` so the API waits
  for vLLM to finish its ~37 s startup.
- Ports published on `127.0.0.1` only.

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

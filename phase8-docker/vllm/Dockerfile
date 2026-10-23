# vLLM 0.29.0+rocm723 in a container, repeating the native ~/vllm-env recipe.
# Build context: phase8-docker/ (it reuses the ROCDXG debs and stub in stage0/)
#   docker build -f phase8-docker/vllm/Dockerfile -t lab/vllm phase8-docker
#
# Pull the base once first: `docker pull rocm/dev-ubuntu-24.04:7.2.4-complete`.
# A base pulled implicitly by FROM lives only in the build cache, which Docker
# Desktop trims to 20 GB, so it gets evicted and re-downloaded (7 GB). A tagged
# image is never garbage-collected.
#
# The venv's torch (2.12.0+git, from vLLM's wheel index) does not bundle ROCm:
# torch and vLLM's kernels link against the system ROCm in /opt/rocm. So the
# base is the ROCm 7.2.4 userspace the native setup has, not rocm/pytorch,
# whose own torch 2.10 measured ~13% slower in Stage 0.
FROM rocm/dev-ubuntu-24.04:7.2.4-complete

# Headers and a compiler: torch.compile (inductor) builds C++ at startup.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12-venv python3.12-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

# ROCDXG: the WSL route to the GPU. See stage0/Dockerfile.
COPY stage0/rocdxg-roct_1.2.2_amd64.deb stage0/rocdxg-amd-smi-lib_1.2.2_amd64.deb /tmp/
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      /tmp/rocdxg-roct_1.2.2_amd64.deb /tmp/rocdxg-amd-smi-lib_1.2.2_amd64.deb \
 && rm -rf /var/lib/apt/lists/* /tmp/*.deb

COPY stage0/norocprof.c /opt/norocprof/
RUN gcc -shared -fPIC -O2 -o /opt/norocprof/norocprof.so /opt/norocprof/norocprof.c

# Exact versions of the working native venv (vllm/freeze.sh regenerates it).
# Installed before the env vars below, so pip never runs with the stub loaded.
# The pinned per-release index: plain /rocm/ only serves the latest nightly
# (torch 2.13, aiter 0.1.24 by 2026-10-23), so the lock can't resolve there.
# The cache mount keeps downloaded wheels (torch alone is GBs) between builds,
# outside the image, so changing the lock doesn't re-download everything.
COPY vllm/requirements.lock /tmp/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.12 -m venv /opt/vllm-env \
 && /opt/vllm-env/bin/pip install \
      --extra-index-url https://wheels.vllm.ai/rocm/0.29.0/rocm723/ \
      -r /tmp/requirements.lock

# System libraries the torch wheel links to but does not ship (native Ubuntu
# had them from earlier installs): OpenMPI, TBB, libjpeg, libdw. A separate
# step after pip, so changing it does not rebuild the multi-GB venv layer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libopenmpi3t64 libtbb12 libjpeg-turbo8 libdw1t64 \
 && rm -rf /var/lib/apt/lists/*

# Library order matters, see stage0/Dockerfile. HF_HUB_OFFLINE: the model
# cache is mounted, so skip the Hub checks on every start.
ENV PATH=/opt/vllm-env/bin:$PATH \
    HSA_ENABLE_DXG_DETECTION=1 \
    LD_PRELOAD=/opt/norocprof/norocprof.so \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm-wsl/lib:/opt/rocm/lib \
    HF_HUB_OFFLINE=1

EXPOSE 8000
# Same flags as VLLM_GUEST_CMD in phase4-observability/servers.py, so native
# and container runs are comparable. Override CMD to serve another model.
ENTRYPOINT ["vllm", "serve"]
CMD ["Qwen/Qwen2.5-1.5B-Instruct", \
     "--served-model-name", "qwen2.5-1.5b-instruct", \
     "--dtype", "float16", \
     "--max-model-len", "4096", \
     "--gpu-memory-utilization", "0.35", \
     "--compilation-config", "{\"cudagraph_capture_sizes\": [1,2,3,4,5,6,7,8,10,12,16,24,32]}", \
     "--host", "0.0.0.0", "--port", "8000"]

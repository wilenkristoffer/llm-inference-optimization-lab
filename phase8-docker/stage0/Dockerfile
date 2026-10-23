# Stage 0: can a container reach the RX 7800 XT through /dev/dxg?
# Repeats the WSL recipe (see memory: rocm-wsl-working-setup) on top of AMD's
# rocm/pytorch image instead of building ROCm ourselves.
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

# ROCDXG 1.2.2 from github.com/ROCm/librocdxg releases. The debs are
# gitignored; copy them from ~/ in the Ubuntu distro before building.
COPY rocdxg-roct_1.2.2_amd64.deb rocdxg-amd-smi-lib_1.2.2_amd64.deb /tmp/
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      /tmp/rocdxg-roct_1.2.2_amd64.deb /tmp/rocdxg-amd-smi-lib_1.2.2_amd64.deb \
 && rm -rf /var/lib/apt/lists/* /tmp/*.deb

# rocprofiler-sdk aborts when /sys/class/kfd is missing; stub its gatekeeper.
COPY norocprof.c /opt/norocprof/
RUN gcc -shared -fPIC -O2 -o /opt/norocprof/norocprof.so /opt/norocprof/norocprof.c

# Required for ROCm below 7.13, otherwise no GPU is detected.
# libdxcore.so comes from the /usr/lib/wsl bind mount; without it on the path
# the runtime fails with "[hsaKmtOpenKFD] Failed to load libdxcore.so".
# /opt/rocm-wsl/lib is where the ROCDXG debs put their WSL builds of
# libhsakmt and libamd_smi; it must come before /opt/rocm/lib, as it does in
# the native WSL setup, or torch 2.14 fails on an amdsmi undefined symbol.
ENV HSA_ENABLE_DXG_DETECTION=1 \
    LD_PRELOAD=/opt/norocprof/norocprof.so \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm-wsl/lib:/opt/rocm/lib

COPY gpu_check.py /opt/gpu_check.py
CMD ["python", "/opt/gpu_check.py"]

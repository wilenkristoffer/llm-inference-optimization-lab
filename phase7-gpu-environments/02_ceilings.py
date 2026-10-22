"""
Phase 7, file 2: what this machine can actually do, against its datasheets.

Every efficiency figure in the lab so far divides by a datasheet number:
624 GB/s for VRAM, 51 GB/s for DDR4. A datasheet number is a product of clock
and bus width; no real kernel reaches it. This measures the ceilings that
inference actually runs into:

    VRAM bandwidth     decode reads every weight once per token  (02_bandwidth.hip)
    PCIe bandwidth     what offload pays at every CPU/GPU layer boundary
    GPU matmul         prefill multiplies a whole prompt by each weight
    RAM bandwidth      decode on the CPU
    CPU matmul         prefill on the CPU

The GPU probes run in WSL, through the same ROCm stack vLLM uses. Ollama runs on
native Windows with its own ROCm 7.1 build - same silicon, different driver
path. Memory bandwidth is a property of the silicon and should carry over;
the write-up states the caveat rather than assuming it away.

Measured 2026-10-22 (first run, before this script stored anything):

    vram_read   ~600 GB/s from 128 MB up = 96% of 624. Higher than predicted
                (80-90%). So Phase 1's 14B at 399 GB/s really is ~66% of what
                the card can do - the gap is the runtime, not an inflated spec.
    caches      reads run at ~2,000 GB/s up to 64 MB (the Infinity Cache), and
                copy drops a step earlier because it touches two buffers.
    PCIe        ~28 GB/s each way = 89% of PCIe 4.0 x16.
    FP16 matmul ~60 TFLOPS = 80% of AMD's figure. FP32 only 10 = 27%: the
                FP32 spec counts RDNA3 dual-issue, which real GEMMs rarely get.
    RAM read    46.6 GB/s in WSL. The RAM turned out to be DDR4-3600 (XMP),
                not the DDR4-3200 assumed since Phase 1: 57.6 GB/s on paper.

Usage:
    python 02_ceilings.py              # build if needed, run all, store
    python 02_ceilings.py --no-store   # print only
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(name, relative):
    path = os.path.join(os.path.dirname(__file__), relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


storage = _load("storage", "../phase1-ollama-baseline/04_storage.py")

HARDWARE_ID = "Ryzen 5600X / RX 7800 XT / 32GB DDR4"

# Datasheet figures, so every measurement is printed as a share of its claim.
# FP16/FP32 matmul peaks are AMD's published numbers for the RX 7800 XT and
# assume RDNA3's dual-issue / WMMA paths - the best case, not a typical one.
DATASHEET = {
    "vram_read": (624.0, "GB/s", "256-bit GDDR6 @ 19.5 Gbps"),
    "vram_copy": (624.0, "GB/s", "same bus"),
    "d2d_memcpy": (624.0, "GB/s", "same bus"),
    "h2d_pinned": (31.5, "GB/s", "PCIe 4.0 x16"),
    "d2h_pinned": (31.5, "GB/s", "PCIe 4.0 x16"),
    "matmul_fp16": (74.65, "TFLOPS", "AMD spec, FP16"),
    "matmul_fp32": (37.32, "TFLOPS", "AMD spec, FP32 dual-issue"),
    # DDR4-3600, not the 3200 the lab assumed until 2026-10-22: the kit runs
    # its XMP profile (Win32_PhysicalMemory ConfiguredClockSpeed = 3600).
    # 3600 MT/s x 8 bytes x 2 channels.
    "cpu_read": (57.6, "GB/s", "DDR4-3600 dual channel"),
    "cpu_copy": (57.6, "GB/s", "DDR4-3600 dual channel"),
    "cpu_matmul_fp32": (None, "TFLOPS", "6 Zen 3 cores, AVX2 FMA"),
}

HERE_WSL = None   # filled in main(): this folder as WSL sees it
BUILD_DIR = "~/lab/phase7"

# The PyTorch half. Runs inside ~/rocm-env. torch.cuda is the HIP runtime here.
TORCH_PROBE = r"""
import json, time, torch

def timed(fn, repeats=7, warm=3):
    for _ in range(warm):
        fn()
    sync()
    out = []
    for _ in range(repeats):
        t = time.perf_counter(); fn(); sync(); out.append(time.perf_counter() - t)
    return sorted(out)

def emit(probe, bytes_, values, env):
    values = sorted(values)
    print(json.dumps({"probe": probe, "bytes": bytes_, "best": values[-1],
                      "median": values[len(values) // 2], "repeats": len(values),
                      "environment": env}), flush=True)

# GPU matmul: square, big enough that the matrix units, not launch overhead or
# memory, are the limit. 2*n^3 floating-point operations per multiply.
sync = torch.cuda.synchronize
for dtype, name in ((torch.float16, "matmul_fp16"), (torch.float32, "matmul_fp32")):
    for n in (4096, 8192):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        ts = timed(lambda: a @ b)
        emit(name, None, [2 * n**3 / t / 1e12 for t in ts], f"wsl2-rocm n={n}")
        del a, b
torch.cuda.empty_cache()

# CPU. 1 GiB buffers, far beyond the 32 MB L3, so this is DRAM.
sync = lambda: None
n = (1 << 30) // 4
x = torch.ones(n, dtype=torch.float32)
y = torch.empty_like(x)
for threads in (6, 12):
    torch.set_num_threads(threads)
    env = f"wsl2-cpu threads={threads}"
    emit("cpu_read", 4 * n, [4 * n / t / 1e9 for t in timed(lambda: x.sum())], env)
    emit("cpu_copy", 4 * n, [8 * n / t / 1e9 for t in timed(lambda: y.copy_(x))], env)
    a = torch.randn(4096, 4096); b = torch.randn(4096, 4096)
    emit("cpu_matmul_fp32", None,
         [2 * 4096**3 / t / 1e12 for t in timed(lambda: a @ b, repeats=5)], env)
"""


def wsl(cmd, input=None, timeout=600):
    """Run a bash command in WSL inside the ROCm venv (activate sets the DXG
    detection flag and the rocprofiler stub - see 01_stack_inventory.py)."""
    return subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-lc",
         f"source ~/rocm-env/bin/activate && {cmd}"],
        input=input, capture_output=True, text=True, timeout=timeout)


def wsl_path(path):
    drive, rest = os.path.splitdrive(os.path.abspath(path))
    return f"/mnt/{drive[0].lower()}{rest.replace(os.sep, '/')}"


def parse(stdout, default_env):
    rows, device = [], None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        if "device" in obj:
            device = obj
            continue
        obj.setdefault("environment", default_env)
        rows.append(obj)
    return rows, device


def run_hip():
    src = wsl_path(os.path.join(os.path.dirname(__file__), "02_bandwidth.hip"))
    # Rebuild only when the source is newer than the binary: hipcc is ~2 s, but
    # a stale binary measuring old code would be worse than the wait.
    build = (f"mkdir -p {BUILD_DIR} && cd {BUILD_DIR} && "
             f"if [ ! -x bandwidth ] || [ '{src}' -nt bandwidth ]; then "
             f"hipcc -O3 --offload-arch=gfx1101 -o bandwidth '{src}'; fi && ./bandwidth")
    r = wsl(build)
    if r.returncode != 0:
        sys.exit(f"HIP probe failed:\n{r.stderr[-2000:]}")
    return parse(r.stdout, "wsl2-rocm")


def run_torch():
    r = wsl("python -", input=TORCH_PROBE)
    if r.returncode != 0:
        sys.exit(f"PyTorch probe failed:\n{r.stderr[-2000:]}")
    return parse(r.stdout, "wsl2")[0]


def mb(b):
    return "-" if b is None else (f"{b / 2**20:,.0f} MB" if b < 2**30 else f"{b / 2**30:,.0f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-store", action="store_true")
    args = parser.parse_args()

    print("building and running the HIP bandwidth probe (WSL)...")
    hip_rows, device = run_hip()
    if device:
        print(f"  {device['device']} {device['arch']}, {device['multiprocessors']} "
              f"WGPs, L2 {device['l2_bytes'] / 2**20:.0f} MB, {device['bus_bits']}-bit bus")
    print("running the PyTorch compute and CPU probes (WSL)...")
    rows = hip_rows + run_torch()

    for r in rows:
        r["unit"] = DATASHEET.get(r["probe"], (None, "GB/s", None))[1]

    print(f"\n{'probe':<16} {'size / where':<24} {'best':>9} {'median':>9} "
          f"{'unit':<7} {'datasheet':>10} {'share':>6}")
    print("-" * 86)
    for r in rows:
        spec, unit, _ = DATASHEET.get(r["probe"], (None, "GB/s", None))
        where = mb(r["bytes"]) if r["probe"].startswith(("vram", "d2d", "h2d", "d2h")) \
            else r["environment"]
        share = f"{r['best'] / spec:.0%}" if spec else "-"
        print(f"{r['probe']:<16} {where:<24} {r['best']:>9.2f} {r['median']:>9.2f} "
              f"{unit:<7} {spec or '-':>10} {share:>6}")

    # The figure that matters for decode: the large-buffer read plateau.
    plateau = [r["best"] for r in rows
               if r["probe"] == "vram_read" and (r["bytes"] or 0) >= 128 * 2**20]
    if plateau:
        p = max(plateau)
        print(f"\nVRAM read plateau (>= 128 MB, past the 64 MB Infinity Cache): "
              f"{p:.0f} GB/s = {p / 624:.0%} of the 624 GB/s datasheet.")
        print(f"Phase 1's 14B decoded at 399 GB/s effective: {399 / p:.0%} of measured.")

    if not args.no_store:
        conn = storage.connect()
        group = f"ceilings|{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
        storage.record_rows(conn, "hardware_ceilings", [
            {"run_group": group, "hardware_id": HARDWARE_ID, "probe": r["probe"],
             "environment": r["environment"], "bytes": r["bytes"],
             "best": r["best"], "median": r["median"], "unit": r["unit"],
             "repeats": r["repeats"]} for r in rows])
        print(f"stored {len(rows)} rows under {group}")


if __name__ == "__main__":
    main()

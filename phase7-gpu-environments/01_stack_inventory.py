"""
Phase 7, file 1: what each layer of the GPU stack actually is on this machine.

    Application          this script, the lab, the user's agent
        |
    Inference runtime    Ollama (API + scheduler) -> llama-server (llama.cpp)
        |
    GPU libraries        ggml-hip.dll -> ROCm (HIP runtime, rocBLAS, hipBLASLt)
        |
    GPU                  RX 7800 XT, gfx1101

No benchmark. It reads what is installed and what the runtime picked, so the
diagram above is backed by files and log lines rather than asserted.

Things this showed when it was written (2026-10-22, Ollama 0.40.7):

1. ONE RUNTIME, MANY BACKENDS. Ollama ships CUDA 12, CUDA 13, ROCm 7.1 and
   Vulkan builds side by side, plus a CPU build per instruction-set generation,
   and picks at startup. "GPU support" means having compiled code for the
   exact chip; the runtime is a dispatcher over those builds.
2. THE ROCm BACKEND IS THE CUDA BACKEND. llama.cpp's ROCm log line is
   `ggml_cuda_init: found 1 ROCm devices`: the same CUDA source, compiled for
   AMD through HIP. That is what HIP is for.
3. THE CPU BUILD IS CHOSEN BY INSTRUCTION SET, NOT VENDOR. A Ryzen 5600X loads
   `ggml-cpu-haswell` (AVX2), not `zen4`, because Zen 3 has no AVX-512.
4. PLACEMENT IS A RUNNER FLAG. `num_gpu` in a request becomes llama-server's
   `-ngl` (layers on GPU). Without it, Ollama 0.40 passes no -ngl at all and
   llama.cpp decides. /api/ps reports the result as size vs size_vram.
5. WSL SEES THE CARD DIFFERENTLY. `rocminfo` fails there - it looks for the
   Linux kfd driver, and WSL reaches the GPU through /dev/dxg instead - so the
   WSL layer is read through PyTorch, which goes through the HIP runtime. It
   reports 30 "compute units" for a 60-CU card: RDNA3 pairs CUs into
   work-group processors, and HIP counts those.

Usage:
    python 01_stack_inventory.py
    python 01_stack_inventory.py --no-wsl     # skip the (slow) WSL probe
"""

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA = "http://127.0.0.1:11434"
OLLAMA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama")
OLLAMA_LIB = os.path.join(OLLAMA_DIR, "lib", "ollama")
SERVER_LOG = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama", "server.log")

# IsProcessorFeaturePresent constants, from winnt.h.
PF_AVX2 = 40
PF_AVX512F = 41

# Runs in the ROCm venv from Phase 3. torch.cuda is the HIP runtime on a ROCm
# build - PyTorch kept the CUDA name, which is the same story as ggml_cuda_init.
WSL_PROBE = r"""
import json, subprocess, torch
p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
hipcc = subprocess.run(["hipcc", "--version"], capture_output=True, text=True).stdout
print(json.dumps({
    "torch": torch.__version__,
    "hip_runtime": torch.version.hip,
    "cuda_runtime": torch.version.cuda,
    "device": p and p.name,
    "arch": p and getattr(p, "gcnArchName", None),
    "compute_units": p and p.multi_processor_count,
    "vram_gib": p and round(p.total_memory / 2**30, 1),
    "hipcc": next((l for l in hipcc.splitlines() if "HIP version" in l), None),
}))
"""


def _ollama(path):
    with urllib.request.urlopen(OLLAMA + path, timeout=10) as r:
        return json.load(r)


def dir_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 2**20


def backends():
    """Every backend Ollama ships: GPU ones are folders, CPU ones are DLLs."""
    gpu, cpu = [], []
    for name in sorted(os.listdir(OLLAMA_LIB)):
        path = os.path.join(OLLAMA_LIB, name)
        if os.path.isdir(path):
            gpu.append((name, dir_mb(path)))
        elif name.startswith("ggml-cpu-"):
            cpu.append(name[len("ggml-cpu-"):-len(".dll")])
    return gpu, cpu


def latest_server_start():
    """Lines from the most recent Ollama start, plus the most recent runner load.

    server.log accumulates runner loads after the server's own startup, so the
    startup block is found by its 'server config' line and the runner details
    by the last occurrence of each pattern."""
    try:
        with open(SERVER_LOG, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    start = max((i for i, l in enumerate(lines) if 'msg="server config"' in l), default=0)
    block = lines[start:]

    def last(pattern):
        return next((l for l in reversed(block) if re.search(pattern, l)), None)

    config = block[0] if block else ""
    env = {k: v for k, v in re.findall(r"(OLLAMA_VULKAN|LLAMA_ARG_FIT|OLLAMA_LLM_LIBRARY|"
                                       r"HIP_VISIBLE_DEVICES|GGML_VK_VISIBLE_DEVICES):(\S*)",
                                       config)}
    return {
        "env": env,
        "compute": last(r'msg="inference compute"'),
        "cpu_backend": last(r"loaded CPU backend from"),
        "gpu_backend": last(r"loaded (ROCm|CUDA|Vulkan) backend from"),
        "gpu_init": last(r"ggml_cuda_init|ggml_vulkan:"),
        "device": last(r"Device 0:"),
        "system_info": last(r"system_info:"),
    }


def runner_commands():
    """Command lines of the running llama-server processes (one per loaded model)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" "
             "| ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    return [l.strip() for l in out.splitlines() if l.strip()]


def cpu_info():
    name = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Processor | Select-Object -First 1 | "
         "ForEach-Object { \"$($_.Name)|$($_.NumberOfCores)|$($_.NumberOfLogicalProcessors)\" })"],
        capture_output=True, text=True, timeout=20).stdout.strip()
    model, cores, threads = (name.split("|") + ["?", "?"])[:3]
    present = ctypes.windll.kernel32.IsProcessorFeaturePresent
    return {"name": model.strip(), "cores": cores, "threads": threads,
            "avx2": bool(present(PF_AVX2)), "avx512": bool(present(PF_AVX512F))}


def wsl_probe():
    try:
        out = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-lc",
             # activate, not just the venv's python: it sets
             # HSA_ENABLE_DXG_DETECTION and LD_PRELOADs the rocprofiler stub,
             # without which ROCm finds no GPU under WSL (Phase 3 setup).
             "source ~/rocm-env/bin/activate && python - 2>/dev/null"],
            input=WSL_PROBE, capture_output=True, text=True, timeout=120).stdout
        rocminfo = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-lc",
             "rocminfo >/dev/null 2>&1 && echo ok || echo fails"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return {"error": str(e)}
    try:
        info = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "probe printed nothing - is ~/rocm-env present?"}
    info["rocminfo"] = rocminfo
    return info


def flag(cmd, *names):
    for n in names:
        m = re.search(rf"(?:^|\s){re.escape(n)}\s+(\S+)", cmd)
        if m:
            return m.group(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wsl", action="store_true")
    args = parser.parse_args()

    print("=== Inference runtime: Ollama ===")
    try:
        print(f"version   {_ollama('/api/version')['version']}")
        running = _ollama("/api/ps")["models"]
    except OSError:
        print("not reachable on 127.0.0.1:11434")
        running = []
    print(f"install   {OLLAMA_DIR}")

    gpu, cpu = backends()
    print("\n=== GPU libraries Ollama ships (one is used) ===")
    for name, mb in gpu:
        print(f"  {name:<12} {mb:>7,.0f} MB")
    print(f"  CPU builds: {', '.join(cpu)}")

    log = latest_server_start()
    print("\n=== What the runtime picked (latest server.log) ===")
    if log.get("env"):
        print("  env       " + "  ".join(f"{k}={v or '(unset)'}" for k, v in log["env"].items()))
    for key in ("compute", "gpu_backend", "gpu_init", "device", "cpu_backend"):
        line = log.get(key)
        if line:
            # Trim timestamps and source tags; keep the message.
            line = re.sub(r"^time=\S+ level=\S+ source=\S+ ", "", line)
            print(f"  {key:<11} {line[:150]}")
    if log.get("system_info"):
        threads = re.search(r"n_threads = (\d+)", log["system_info"])
        print(f"  cpu threads used by llama.cpp: {threads.group(1) if threads else '?'}")

    c = cpu_info()
    print("\n=== CPU (decides which ggml-cpu build loads) ===")
    print(f"  {c['name']}, {c['cores']} cores / {c['threads']} threads")
    print(f"  AVX2 {'yes' if c['avx2'] else 'no'}   AVX-512 {'yes' if c['avx512'] else 'no'}")

    print("\n=== Loaded models: placement ===")
    if not running:
        print("  none loaded (send a request first to see a runner)")
    for m in running:
        size, vram = m.get("size", 0), m.get("size_vram", 0)
        share = vram / size if size else 0
        print(f"  {m['name']:<28} {size / 2**20:>7,.0f} MB total, "
              f"{vram / 2**20:>7,.0f} MB in VRAM ({share:.0%} GPU)")
    for cmd in runner_commands():
        print(f"  runner: -c {flag(cmd, '-c')}  -np {flag(cmd, '-np')}  "
              f"-ngl {flag(cmd, '-ngl', '--n-gpu-layers') or '(not passed - llama.cpp decides)'}  "
              f"flash-attn {flag(cmd, '--flash-attn')}")

    if not args.no_wsl:
        print("\n=== WSL2 side (vLLM's stack) ===")
        w = wsl_probe()
        if "error" in w:
            print(f"  {w['error']}")
        else:
            print(f"  PyTorch {w['torch']}  HIP runtime {w['hip_runtime']}  "
                  f"(CUDA runtime: {w['cuda_runtime']})")
            print(f"  device  {w['device']}  arch {w['arch']}  "
                  f"{w['compute_units']} CUs  {w['vram_gib']} GiB")
            print(f"  hipcc   {w['hipcc']}")
            print(f"  rocminfo {w['rocminfo']}"
                  + ("  (expected under WSL: no /sys/class/kfd; GPU is reached via /dev/dxg)"
                     if w["rocminfo"] == "fails" else ""))


if __name__ == "__main__":
    main()

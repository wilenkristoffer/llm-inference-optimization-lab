"""
Phase 4, file 2: start/stop control for the two model servers.

Swapping backends by hand is the slowest part of running an experiment - vLLM
takes about a minute to load, and there is no signal other than watching a log
scroll. This makes it two clicks and reports readiness.

Three things it encodes, all learned the hard way:

1. STARTING ONE STOPS THE OTHER, BY DEFAULT. vLLM reserves its VRAM at startup
   and holds it for the process lifetime. If the two together do not fit,
   Ollama's allocation still SUCCEEDS - WDDM overcommits and pages the excess
   to system RAM - so `ollama ps` reports "100% GPU" while only ~450 MB is
   resident and every token crosses PCIe. That measured 8.1 tok/s instead of
   121, with vLLM at 0.85 of the card.

   keep_other=True skips the stop. That is only safe because vLLM now takes
   0.35 of the card and the model on both sides is the 1.5B; a bigger Ollama
   model beside vLLM would page silently exactly as above. Measured with both
   resident: Ollama 119.0 tok/s against 121.7 alone (3 runs each), vLLM
   unchanged - fine for convenience, but run headline benchmarks alone.

2. STOPPING OLLAMA MEANS KILLING llama-server TOO. The `ollama` process is only
   the API layer; a separate llama-server child holds the weights. Killing the
   parent orphans the child, which keeps several GB of VRAM.

3. READINESS IS POLLED, NOT PARSED. Watching for "Application startup complete"
   in a log is fragile across versions. Asking the server whether it answers is
   the definition of ready.

vLLM runs inside WSL, and a process started with `wsl -- bash -c "... &"` dies
when that command returns - WSL tears the session down. So we hold the wsl.exe
child open in THIS process instead of backgrounding it inside the guest.

NOTE: this API executes processes. It binds to 127.0.0.1 only and has no auth.
That is fine for a local lab tool and would not be for anything reachable.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque

IS_WINDOWS = sys.platform == "win32"
# Keeps a console window from flashing up for each spawned process.
NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

VLLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VLLM_SERVED_NAME = "qwen2.5-1.5b-instruct"
# 0.35 of 16 GB = 5.6 GB: 2.9 GB weights, graphs and activations, and the rest
# KV cache - ~50k tokens, room for 12 full 4096-token requests where the lab
# never runs more than 8. At 0.85 the KV cache held 364k tokens (89x) and left
# no room for Ollama. Raise this for a bigger vLLM model.
VLLM_GPU_FRACTION = "0.35"
VLLM_MAX_LEN = "4096"
# vLLM captures a CUDA graph per batch size, 51 by default, twice over (once to
# estimate memory). The lab never batches past 8, so capture only what it uses.
# A batch between sizes is padded up to the next one.
VLLM_CAPTURE_SIZES = "[1,2,4,8,16]"

OLLAMA_NUM_PARALLEL = "8"

# Guest-side command. `exec` replaces the shell so terminating wsl.exe reaches
# vLLM itself rather than leaving an orphan behind the shell. HF_HUB_OFFLINE
# skips the Hugging Face network checks on every start - the model is cached.
VLLM_GUEST_CMD = (
    "source ~/vllm-env/bin/activate && "
    "export HF_HUB_OFFLINE=1 && "
    f"exec vllm serve {VLLM_MODEL} "
    f"--served-model-name {VLLM_SERVED_NAME} "
    "--dtype float16 "
    f"--max-model-len {VLLM_MAX_LEN} "
    f"--gpu-memory-utilization {VLLM_GPU_FRACTION} "
    f"--compilation-config '{{\"cudagraph_capture_sizes\": {VLLM_CAPTURE_SIZES}}}' "
    "--host 0.0.0.0 --port 8000"
)

# The rolling buffer below is for the UI. The full output goes here, one file
# per server, each line stamped with seconds since start - so a slow start can
# be diagnosed after the fact instead of reproduced. Gitignored (*.log).
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

SERVERS = {
    "ollama": {
        "label": "Ollama",
        "health": "http://127.0.0.1:11434/api/version",
        "port": 11434,
        "where": "native Windows",
    },
    "vllm": {
        "label": "vLLM",
        "health": "http://127.0.0.1:8000/v1/models",
        "port": 8000,
        "where": "WSL2",
    },
}

# Processes we started, and a rolling tail of their output. Servers started
# outside this tool still show as ready - health is polled, not inferred from
# whether we own a handle.
_procs: dict[str, subprocess.Popen] = {}
_logs: dict[str, deque] = {name: deque(maxlen=40) for name in SERVERS}
_lock = threading.Lock()


class _LogFile:
    """One server's full log, each line stamped with seconds since start."""

    def __init__(self, name):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._file = open(os.path.join(LOG_DIR, f"{name}.log"), "w",
                          encoding="utf-8")
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self.write(f"-- starting {name} at {time.strftime('%Y-%m-%d %H:%M:%S')} --")

    def write(self, line):
        with self._lock:
            if self._file.closed:
                return
            self._file.write(f"[{time.monotonic() - self._started:6.1f}s] {line}\n")
            self._file.flush()

    def close(self):
        with self._lock:
            self._file.close()


def _pump(name, proc, log):
    """Drain the child's output into the UI's rolling buffer and the log file."""
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        if line:
            _logs[name].append(line)
            log.write(line)
    proc.stdout.close()
    log.write(f"-- process exited ({proc.poll()}) --")
    log.close()


def _watch_ready(name, proc, log):
    """
    Record when the server first answers - the number that matters for start
    time, and the one no server prints itself. Polled on its own thread so it
    is accurate to 0.5 s rather than to whenever the server next logs a line.
    """
    while proc.poll() is None:
        if _healthy(name, timeout=0.5):
            log.write("-- READY: first health check answered --")
            return
        time.sleep(0.5)


def _healthy(name, timeout=1.5):
    try:
        with urllib.request.urlopen(SERVERS[name]["health"], timeout=timeout):
            return True
    except Exception:
        return False


def status(name):
    """
    stopped  - not answering, and we are not starting it
    starting - we hold a live process but it does not answer yet
    ready    - it answers, whoever started it
    """
    proc = _procs.get(name)
    alive = proc is not None and proc.poll() is None
    if _healthy(name):
        return "ready"
    return "starting" if alive else "stopped"


def status_all():
    out = {}
    for name, cfg in SERVERS.items():
        state = status(name)
        out[name] = {
            "name": name,
            "label": cfg["label"],
            "where": cfg["where"],
            "port": cfg["port"],
            "status": state,
            "managed": name in _procs and _procs[name].poll() is None,
            "log": list(_logs[name])[-12:],
        }
    return out


def stop(name):
    """Stop a server whether or not we started it."""
    with _lock:
        proc = _procs.pop(name, None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        if name == "ollama":
            # By image name, because the tray app may have started it, and
            # because llama-server is a separate process holding the weights.
            for image in ("ollama.exe", "ollama app.exe", "llama-server.exe"):
                subprocess.run(["taskkill", "/F", "/IM", image],
                               capture_output=True, creationflags=NO_WINDOW)
        else:
            # Belt and braces: anything the terminate above did not reach.
            subprocess.run(["wsl", "-d", "Ubuntu", "--",
                            "bash", "-lc", "pkill -f 'vllm serve' || true"],
                           capture_output=True, creationflags=NO_WINDOW)

        _logs[name].append(f"-- stopped {name} --")
    return status_all()


def start(name, keep_other=False):
    """
    Start a server. By default stops the other one first - they cannot share
    16 GB without one of them silently paging over PCIe. keep_other=True leaves
    it running, which is safe for the 1.5B only (see the module docstring).
    """
    if not keep_other:
        for other in SERVERS:
            if other != name:
                stop(other)

    with _lock:
        existing = _procs.get(name)
        if existing and existing.poll() is None:
            return status_all()

        _logs[name].clear()
        _logs[name].append(f"-- starting {name} --")

        if name == "ollama":
            env_prefix = ["cmd", "/c",
                          f"set OLLAMA_NUM_PARALLEL={OLLAMA_NUM_PARALLEL} && "
                          "ollama serve"]
            proc = subprocess.Popen(
                env_prefix, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=NO_WINDOW)
        else:
            proc = subprocess.Popen(
                ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", VLLM_GUEST_CMD],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=NO_WINDOW)

        _procs[name] = proc
        log = _LogFile(name)
        threading.Thread(target=_pump, args=(name, proc, log), daemon=True).start()
        threading.Thread(target=_watch_ready, args=(name, proc, log),
                         daemon=True).start()

    return status_all()


if __name__ == "__main__":
    print(json.dumps(status_all(), indent=2))

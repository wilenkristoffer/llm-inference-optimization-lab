"""
Phase 4, file 2: start/stop control for the two model servers.

Swapping backends by hand is the slowest part of running an experiment - vLLM
takes about a minute to load, and there is no signal other than watching a log
scroll. This makes it two clicks and reports readiness.

Three things it encodes, all learned the hard way:

1. STARTING ONE STOPS THE OTHER. vLLM reserves its VRAM at startup and holds it
   for the process lifetime. With both running, Ollama's allocation still
   SUCCEEDS - WDDM overcommits and pages the excess to system RAM - so
   `ollama ps` reports "100% GPU" while only ~450 MB is resident and every
   token crosses PCIe. That measured 8.1 tok/s instead of 121.

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
import subprocess
import sys
import threading
import urllib.request
from collections import deque

IS_WINDOWS = sys.platform == "win32"
# Keeps a console window from flashing up for each spawned process.
NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

VLLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VLLM_SERVED_NAME = "qwen2.5-1.5b-instruct"
# 0.85 leaves room only when Ollama is stopped, which is why start is exclusive.
VLLM_GPU_FRACTION = "0.85"
VLLM_MAX_LEN = "4096"

OLLAMA_NUM_PARALLEL = "8"

# Guest-side command. `exec` replaces the shell so terminating wsl.exe reaches
# vLLM itself rather than leaving an orphan behind the shell.
VLLM_GUEST_CMD = (
    "source ~/vllm-env/bin/activate && "
    f"exec vllm serve {VLLM_MODEL} "
    f"--served-model-name {VLLM_SERVED_NAME} "
    "--dtype float16 "
    f"--max-model-len {VLLM_MAX_LEN} "
    f"--gpu-memory-utilization {VLLM_GPU_FRACTION} "
    "--host 0.0.0.0 --port 8000"
)

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


def _pump(name, proc):
    """Drain the child's output into a rolling buffer so the UI can show it."""
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        if line:
            _logs[name].append(line)
    proc.stdout.close()


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


def start(name, exclusive=True):
    """
    Start a server. By default stops the other one first - they cannot share
    16 GB without one of them silently paging over PCIe.
    """
    if exclusive:
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
        threading.Thread(target=_pump, args=(name, proc), daemon=True).start()

    return status_all()


if __name__ == "__main__":
    print(json.dumps(status_all(), indent=2))

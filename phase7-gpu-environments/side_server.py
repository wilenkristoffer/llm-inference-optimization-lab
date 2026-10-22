"""
Phase 7 helper: an Ollama server pinned to one GPU library, on a side port.

Ollama 0.40 ships ROCm, Vulkan, CUDA and CPU builds and picks one at startup.
Which one is decided by what it can see, so hiding a library is how you choose:

    rocm     Vulkan hidden (GGML_VK_VISIBLE_DEVICES=-1)
    vulkan   ROCm hidden (HIP_VISIBLE_DEVICES=-1)    - same card, other library
    cpu      both hidden                             - no GPU library at all

Verified 2026-10-22: each produces the expected `inference compute` line.
Setting OLLAMA_LLM_LIBRARY=vulkan also works; hiding devices was preferred
because it also gives a CPU-only server.

Why a side instance rather than restarting the main one: every mode then runs
under identical settings (1 slot, same model store, same keep-alive), the
library is checked from the server's own log rather than assumed, and the
user's main Ollama on 11434 is left as it was. The main instance's models are
evicted first so they do not hold VRAM during a measurement.
"""

import os
import re
import subprocess
import sys
import time
import urllib.request

PORT = 11435
BASE = f"http://127.0.0.1:{PORT}"
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

HIDE = {
    "rocm": {"GGML_VK_VISIBLE_DEVICES": "-1"},
    "vulkan": {"HIP_VISIBLE_DEVICES": "-1"},
    "cpu": {"HIP_VISIBLE_DEVICES": "-1", "GGML_VK_VISIBLE_DEVICES": "-1"},
}
EXPECT = {"rocm": "ROCm", "vulkan": "Vulkan", "cpu": "cpu"}


class SideServer:
    """with SideServer("vulkan") as s: ... point the trials module at s.base."""

    def __init__(self, library, num_parallel=1):
        if library not in HIDE:
            raise ValueError(f"library must be one of {list(HIDE)}")
        self.library = library
        self.num_parallel = num_parallel
        self.base = BASE
        self.proc = None
        self.log_path = os.path.join(LOG_DIR, f"side-ollama-{library}.log")

    def __enter__(self):
        _stop_port_owner()
        os.makedirs(LOG_DIR, exist_ok=True)
        env = {k: v for k, v in os.environ.items()
               if k not in ("HIP_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES",
                            "OLLAMA_LLM_LIBRARY")}
        env.update(HIDE[self.library])
        env.update({"OLLAMA_HOST": f"127.0.0.1:{PORT}",
                    "OLLAMA_NUM_PARALLEL": str(self.num_parallel),
                    "OLLAMA_KEEP_ALIVE": "30m"})
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(["ollama", "serve"], env=env,
                                     stdout=self._log, stderr=subprocess.STDOUT,
                                     creationflags=NO_WINDOW)
        for _ in range(60):
            if self.proc.poll() is not None:
                raise RuntimeError(f"side Ollama exited; see {self.log_path}")
            try:
                urllib.request.urlopen(BASE + "/api/version", timeout=1).read()
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("side Ollama did not answer within 30 s")
        # Discovery is logged just after the API comes up; give it a moment.
        for _ in range(40):
            found = self.discovered()
            if found:
                break
            time.sleep(0.5)
        if found != EXPECT[self.library]:
            self.__exit__()
            raise RuntimeError(f"asked for {self.library}, Ollama picked {found}")
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            # /T takes the llama-server child too - killing only the parent
            # orphans the runner and its VRAM (Phase 4, servers.py).
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True, creationflags=NO_WINDOW)
            self.proc.wait(timeout=10)
        if getattr(self, "_log", None):
            self._log.close()

    def _text(self):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def discovered(self):
        m = re.findall(r'msg="inference compute".*?library=(\S+)', self._text())
        return m[-1] if m else None

    def _offloads(self):
        return re.findall(r"offloaded (\d+)/(\d+) layers to GPU", self._text())

    def offload_count(self):
        return len(self._offloads())

    def offload_line(self, since=0):
        """llama.cpp's own placement report, e.g. 'offloaded 29/29 layers to
        GPU' - ground truth for what -ngl did. Only lines after `since` count,
        so a load that prints none cannot return a previous load's line."""
        m = self._offloads()[since:]
        return (int(m[-1][0]), int(m[-1][1])) if m else None


def _stop_port_owner():
    """A side server left over from a crashed run would make the new one fail
    to bind, and requests would silently go to the old one's library."""
    try:
        urllib.request.urlopen(BASE + "/api/version", timeout=1).read()
    except OSError:
        return
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                         text=True, creationflags=NO_WINDOW).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1].endswith(f":{PORT}") and parts[3] == "LISTENING":
            subprocess.run(["taskkill", "/F", "/T", "/PID", parts[4]],
                           capture_output=True, creationflags=NO_WINDOW)
    time.sleep(1)

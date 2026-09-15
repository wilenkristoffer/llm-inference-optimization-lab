"""
Phase 1, file 6: hardware sampling during inference.

A single inference request is over in milliseconds. Reading a sensor once,
before or after, tells you nothing useful - you need to watch the hardware
WHILE the work happens. So this runs a background thread that polls sensors on
an interval and reports what it saw.

Source is LibreHardwareMonitor's web server (tools/LibreHardwareMonitor,
must be running ELEVATED with Options > Remote Web Server > Run). It reads AMD
sensors through the driver's own API, which - unlike Windows' built-in WDDM
counters - correctly sees ROCm compute work.

Usage:
    python 06_hardware_sampler.py            # demo against a real inference run
"""

import importlib.util
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LHM_URL = "http://127.0.0.1:8085/data.json"   # 127.0.0.1, same lesson as Ollama

# (hardware substring, sensor group, sensor name) -> our metric key.
#
# Keyed on all three deliberately. "GPU Core" alone appears under Load, Clocks,
# Temperatures AND Voltages, and "Package" means the CPU on one device and the
# GPU on another. Being explicit here is what stops us silently charting
# voltage as a temperature.
SENSORS = {
    ("Radeon",       "Load",         "GPU Core"):        "gpu_percent",
    ("Radeon",       "Clocks",       "GPU Core"):        "gpu_clock_mhz",
    ("Radeon",       "Powers",       "GPU Package"):     "gpu_power_w",
    ("Radeon",       "Temperatures", "GPU Core"):        "gpu_temp_c",
    ("Radeon",       "Data",         "GPU Memory Used"): "gpu_vram_used_mb",
    ("Ryzen",        "Load",         "CPU Total"):       "cpu_percent",
    ("Ryzen",        "Temperatures", "Core (Tctl/Tdie)"): "cpu_temp_c",
    ("Total Memory", "Data",         "Memory Used"):     "system_ram_used_gb",
}


def parse_value(text):
    """
    LHM returns display strings in the user's locale: "2759,0 MB", "0,715 V".

    Decimal COMMA. Parsing this with float() after stripping units would raise;
    parsing it by grabbing digits only would turn 2759,0 into 27590. Both
    failure modes are silent enough to poison a whole dataset.
    """
    if not text:
        return None
    cleaned = text.replace(",", ".")
    digits = ""
    for char in cleaned:
        if char.isdigit() or char == ".":
            digits += char
        elif digits:
            break
    try:
        return float(digits)
    except ValueError:
        return None


def read_sensors(url=LHM_URL, timeout=3):
    """One snapshot of every sensor we care about. None if LHM is unreachable."""
    try:
        raw = urllib.request.urlopen(url, timeout=timeout).read()
    except Exception:
        return None

    # LHM serves a nested tree of {Text, Value, Children}. Leaves are sensors.
    tree = json.loads(raw.decode("utf-8", "replace"))
    found = {}

    def walk(node, path):
        children = node.get("Children", [])
        if not children:
            for (hw, group, name), key in SENSORS.items():
                if (name == node.get("Text")
                        and len(path) >= 2
                        and path[-1] == group
                        and any(hw in part for part in path)):
                    found[key] = parse_value(node.get("Value"))
        for child in children:
            walk(child, path + (node.get("Text", ""),))

    walk(tree, ())
    return found


class HardwareSampler:
    """
    Polls sensors on a background thread for the duration of a `with` block.

    A thread is the right tool here: sampling must not block or slow the
    inference request we are trying to measure. Polling every 250 ms costs
    almost nothing and still catches a 350 ms request.
    """

    def __init__(self, interval=0.25, url=LHM_URL):
        self.interval = interval
        self.url = url
        self.samples = []
        self.available = read_sensors(url) is not None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False

    def _poll(self):
        while not self._stop.is_set():
            reading = read_sensors(self.url)
            if reading:
                # Timestamped, so a caller can ask what the hardware was doing
                # during ONE request rather than across a whole session.
                self.samples.append((time.perf_counter(), reading))
            time.sleep(self.interval)

    def summary(self, start=None, end=None):
        """
        Peak AND mean, because they answer different questions.

        Peak VRAM is what decides whether a model fits. Mean GPU load is what
        tells you how well the run used the card.

        start/end (perf_counter values) narrow the window to a single request.
        Without them the mean is diluted by model loading and by the idle gaps
        between trials - which is why the 1.5b appeared to use 11% of the GPU
        while actually saturating it in short bursts.
        """
        rows = [r for ts, r in self.samples
                if (start is None or ts >= start) and (end is None or ts <= end)]
        if not rows:
            return {}
        out = {"samples": len(rows)}
        for key in set().union(*(r.keys() for r in rows)):
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                out[key + "_peak"] = max(values)
                out[key + "_mean"] = statistics.mean(values)
        return out


def _load(name, filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"

    probe = read_sensors()
    if probe is None:
        print(f"LibreHardwareMonitor is not reachable at {LHM_URL}")
        print("Start tools/LibreHardwareMonitor/LibreHardwareMonitor.exe AS ADMIN,")
        print("then Options > Remote Web Server > Run.")
        print("\nFailing loudly here on purpose: a sampler that silently stores")
        print("NULLs would look like 'this hardware has no sensors'.")
        sys.exit(1)

    print(f"LHM reachable. {len(probe)} sensors mapped.")
    trials = _load("trials", "03_repeated_trials.py")

    print(f"\nidle baseline (2s)...")
    with HardwareSampler() as idle:
        time.sleep(2)

    print(f"running {model} x3 while sampling...")
    with HardwareSampler() as load:
        for _ in range(3):
            result = trials.run_once(model, 4096)
    print(f"decode {result['decode_tok_s']:.1f} tok/s")

    idle_stats, load_stats = idle.summary(), load.summary()
    print(f"\n{'metric':<22} {'idle':>12} {'load mean':>12} {'load peak':>12}")
    print("-" * 62)
    for key in ["gpu_percent", "gpu_clock_mhz", "gpu_power_w", "gpu_temp_c",
                "gpu_vram_used_mb", "cpu_percent", "cpu_temp_c",
                "system_ram_used_gb"]:
        i = idle_stats.get(key + "_mean")
        m = load_stats.get(key + "_mean")
        p = load_stats.get(key + "_peak")
        fmt = lambda v: "n/a" if v is None else f"{v:,.1f}"
        print(f"{key:<22} {fmt(i):>12} {fmt(m):>12} {fmt(p):>12}")
    print(f"\nsamples: idle {idle_stats.get('samples')}, "
          f"load {load_stats.get('samples')}")

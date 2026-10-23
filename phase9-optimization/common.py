"""
Phase 9 helpers: the two servers speculative decoding runs on, and the
request functions that read their counters.

llama.cpp: Ollama's own llama-server.exe, started directly, because Ollama's
API has no way to attach a draft model. Two things Ollama normally does have
to be done by hand (Stage 0):

    GGML_BACKEND_PATH -> lib/ollama/rocm_v7_1/ggml-hip.dll, with that
        directory on PATH. Without it llama-server finds no GPU at all.
    --spec-type draft-simple. `-md` alone loads the draft and never uses it;
        the spec type defaults to none.

The draft length is fixed per server start: llama-server ignores a
per-request `speculative.n_max` (n_max 0 still drafted). So every K is its own
server, and the baseline is the same server with no draft - NOT Ollama, which
is 19% slower on the same blob (52.3 vs 44 tok/s, 14b).

vLLM: started in WSL the way phase4-observability/servers.py does it, with
the speculative config as a serve flag. The client runs on Windows and is the
only writer of lab.db (SQLite locks don't cross the Windows/WSL boundary).
"""

import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
HARDWARE_ID = "Ryzen 5600X / RX 7800 XT / 32GB DDR4"


def _load(name, relative):
    path = os.path.join(HERE, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


storage = _load("storage", "../phase1-ollama-baseline/04_storage.py")
workloads = _load("workloads", "workloads.py")


# --- llama.cpp ---------------------------------------------------------------

OLLAMA_LIB = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\lib\ollama")
BLOBS = os.path.expanduser(r"~\.ollama\models\blobs")
# Ollama tag -> blob, from `ollama show --modelfile`. All Q4_K_M.
GGUF = {
    "qwen2.5:0.5b": "sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515",
    "qwen2.5:1.5b": "sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4",
    "qwen2.5:3b": "sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6",
    "qwen2.5:7b": "sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
    "qwen2.5:14b": "sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54",
}
LLAMA_PORT = 11436
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"


def llama_version():
    out = subprocess.run([os.path.join(OLLAMA_LIB, "llama-server.exe"), "--version"],
                         capture_output=True, text=True, env=_llama_env(),
                         creationflags=NO_WINDOW)
    m = re.search(r"commit (\w+)", out.stdout + out.stderr)
    return f"llama.cpp {m.group(1)}" if m else "llama.cpp"


def _llama_env():
    env = dict(os.environ)
    rocm = os.path.join(OLLAMA_LIB, "rocm_v7_1")
    env["PATH"] = rocm + ";" + env["PATH"]
    env["GGML_BACKEND_PATH"] = os.path.join(rocm, "ggml-hip.dll")
    return env


def process_vram_mb(pid, counter="Dedicated Usage"):
    """Dedicated VRAM held by one Windows process, from the GPU Process Memory
    performance counter. Per process, so unlike LibreHardwareMonitor's
    board-wide figure it needs no evict-and-settle baseline. None if the
    counter has no instance for the pid. counter="Shared Usage" gives what the
    process holds in system memory instead - WDDM's overflow when VRAM is
    full (Phase 7: Windows pages rather than failing the allocation)."""
    ps = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\{counter}' "
          "-ErrorAction SilentlyContinue).CounterSamples | "
          "Measure-Object CookedValue -Sum | ForEach-Object Sum")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, creationflags=NO_WINDOW).stdout
    try:
        return float(out.strip()) / 2**20
    except ValueError:
        return None


def evict_ollama():
    """The user's Ollama must not hold VRAM while a side server measures."""
    try:
        ps = json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3))
    except OSError:
        return
    for m in ps.get("models", []):
        body = json.dumps({"model": m["name"], "keep_alive": 0}).encode()
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:11434/api/generate", body,
            {"Content-Type": "application/json"}), timeout=30).read()


class LlamaServer:
    """with LlamaServer("qwen2.5:14b", draft="qwen2.5:0.5b", k=4) as s: s.chat(...)

    k=None with a draft is llama.cpp's default (3). --jinja uses the GGUF's own
    chat template, which is Qwen's, default system prompt included - the same
    text vLLM's HF template produces.
    """

    def __init__(self, target, draft=None, k=None, ctx=4096, extra=(), label=None):
        self.target, self.draft, self.k = target, draft, k
        self.args = [os.path.join(OLLAMA_LIB, "llama-server.exe"),
                     "-m", os.path.join(BLOBS, GGUF[target]), "-ngl", "99",
                     "--flash-attn", "on", "-c", str(ctx), "-np", "1", "--jinja",
                     "--host", "127.0.0.1", "--port", str(LLAMA_PORT), "--no-webui"]
        if draft:
            self.args += ["-md", os.path.join(BLOBS, GGUF[draft]), "-ngld", "99",
                          "--spec-type", "draft-simple"]
            if k is not None:
                self.args += ["--spec-draft-n-max", str(k)]
        self.args += list(extra)
        tag = f"{target}+{draft}-k{k}" if draft else target
        if label:
            tag += "-" + label
        self.log_path = os.path.join(LOG_DIR, "llama-" + re.sub(r"[^\w.+-]", "_", tag) + ".log")
        self.proc = None

    def __enter__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        evict_ollama()
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.args, env=_llama_env(), stdout=self._log,
                                     stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        _wait_healthy(LLAMA_URL, self.proc, 180, self.log_path)
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self._log.close()

    def chat(self, prompt, max_tokens=workloads.MAX_TOKENS, timeout=600, **extra):
        """Non-streaming, so the server's own timings and draft counters come
        back with the response. Decode tok/s is llama.cpp's, which excludes
        prefill. extra goes into the request body (e.g. ignore_eos=True)."""
        body = {"messages": workloads.messages(prompt), "max_tokens": max_tokens,
                "temperature": 0, "seed": 42, "repeat_penalty": 1.0,
                "cache_prompt": False, **extra}
        t0 = time.perf_counter()
        r = _post(LLAMA_URL + "/v1/chat/completions", body, timeout)
        total = (time.perf_counter() - t0) * 1000
        t = r["timings"]
        n = t["predicted_n"]
        accepted = t.get("draft_n_accepted")
        return {
            "response": r["choices"][0]["message"]["content"],
            "input_tokens": t["prompt_n"],
            "output_tokens": n,
            "prefill_ms": t["prompt_ms"],
            "decode_ms": t["predicted_ms"],
            "client_total_ms": total,
            "decode_tokens_per_s": t["predicted_per_second"],
            "draft_tokens": t.get("draft_n"),
            "draft_accepted": accepted,
            # Every pass of the target yields its own token plus the accepted
            # drafts, so passes = tokens - accepted.
            "target_passes": n - accepted if accepted is not None else n,
        }


# --- vLLM ------------------------------------------------------------------

VLLM_URL = "http://127.0.0.1:8000"
HF = {
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
}
# Every token count from 1 to 16 is captured, so no step is padded across the
# skinny-GEMM switch (1-5 tokens -> skinny, 6+ -> generic; Phase 5). With
# speculation a decode step for B sequences is B*(K+1) tokens, so the larger
# sizes cover the concurrency sweep at K up to 8.
CAPTURE_SIZES = list(range(1, 17)) + [20, 24, 28, 32, 40, 48, 64, 80, 96, 128,
                                      144, 160, 192, 224, 256, 288]


def vllm_version():
    out = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc",
                          "~/vllm-env/bin/python -c 'import vllm; print(vllm.__version__)'"],
                         capture_output=True, text=True, creationflags=NO_WINDOW)
    return "vllm " + out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "vllm"


class VllmServer:
    """with VllmServer("qwen2.5-3b-instruct", spec={...}) as s: s.chat(...)"""

    def __init__(self, model, spec=None, gpu_fraction=0.6, max_len=4096, extra=()):
        self.model, self.spec = model, spec
        flags = [HF[model], "--served-model-name", model, "--dtype", "float16",
                 "--max-model-len", str(max_len),
                 "--gpu-memory-utilization", str(gpu_fraction),
                 "--compilation-config",
                 json.dumps({"cudagraph_capture_sizes": CAPTURE_SIZES}),
                 "--host", "0.0.0.0", "--port", "8000"]
        if spec:
            flags += ["--speculative-config", json.dumps(spec)]
        flags += list(extra)
        self.guest_cmd = ("source ~/vllm-env/bin/activate && export HF_HUB_OFFLINE=1 && "
                          "exec vllm serve " + " ".join(shlex.quote(f) for f in flags))
        tag = model + (f"-{spec['method']}-k{spec['num_speculative_tokens']}" if spec else "")
        self.log_path = os.path.join(LOG_DIR, f"vllm-{tag}.log")

    def __enter__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        evict_ollama()
        _pkill_vllm()
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.t_start = time.time()
        self.proc = subprocess.Popen(["wsl", "-d", "Ubuntu", "--", "bash", "-lc",
                                      self.guest_cmd], stdout=self._log,
                                     stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        _wait_healthy(VLLM_URL, self.proc, 600, self.log_path)
        self.startup_s = time.time() - self.t_start
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        _pkill_vllm()
        self._log.close()

    def counters(self):
        """Cumulative spec-decode counters from /metrics (0 without speculation)."""
        text = urllib.request.urlopen(VLLM_URL + "/metrics", timeout=5).read().decode()
        out = {}
        for key, name in (("drafts", "vllm:spec_decode_num_drafts_total"),
                          ("draft_tokens", "vllm:spec_decode_num_draft_tokens_total"),
                          ("accepted", "vllm:spec_decode_num_accepted_tokens_total")):
            out[key] = sum(float(m.group(1)) for m in re.finditer(
                rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.e+]+)$", text, re.M))
        return out

    def chat(self, prompt, max_tokens=workloads.MAX_TOKENS, **extra):
        """Streaming, so decode can be separated from prefill: vLLM's OpenAI
        API carries no server timings. Decode tok/s = (n - 1) / (last - first
        content chunk); with speculation several tokens arrive per chunk, which
        this still counts correctly because n comes from usage, not chunks."""
        body = {"model": self.model, "messages": workloads.messages(prompt),
                "max_tokens": max_tokens, "temperature": 0, "seed": 42,
                "repetition_penalty": 1.0, "stream": True,
                "stream_options": {"include_usage": True}, **extra}
        req = urllib.request.Request(VLLM_URL + "/v1/chat/completions",
                                     json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t_sent = time.perf_counter()
        t_first = t_last = None
        pieces, usage = [], {}
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    text = (ch.get("delta") or {}).get("content")
                    if text:
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        pieces.append(text)
        t_done = time.perf_counter()
        n = usage.get("completion_tokens", 0)
        gen = (t_last - t_first) if t_first else 0
        return {
            "response": "".join(pieces),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": n,
            "ttft_ms": (t_first - t_sent) * 1000 if t_first else None,
            "client_total_ms": (t_done - t_sent) * 1000,
            "decode_ms": gen * 1000,
            "decode_tokens_per_s": (n - 1) / gen if gen > 0 else None,
        }


def _pkill_vllm():
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc",
                    "pkill -f '[v]llm serve' || true"],
                   capture_output=True, creationflags=NO_WINDOW)
    time.sleep(2)


# --- long prompts (Stage 2) ---------------------------------------------------

# A fixed commit, so the corpus never changes under the measurements.
CORPUS_COMMIT = "d3f7e5d"
CORPUS_DOCS = ["phase1", "phase2", "phase3-vllm-setup", "phase3", "phase4", "phase5",
               "phase6-plan", "phase6", "phase7-plan", "phase7", "phase8-plan", "phase8"]
LONG_HEADER = "Below are notes from a series of experiments.\n\n"
LONG_QUESTION = "\n\nSummarize the main findings above as a numbered list."


def corpus_text(source):
    """The lab's own write-ups ("docs", ~45k tokens) or its Python ("code",
    ~100k tokens) as of CORPUS_COMMIT. Real text, so attention sees a real
    distribution rather than repeated filler."""
    root = os.path.dirname(HERE)
    if source == "docs":
        paths = [f"docs/{d}.md" for d in CORPUS_DOCS]
    else:
        names = subprocess.run(["git", "ls-tree", "-r", "--name-only", CORPUS_COMMIT],
                               cwd=root, capture_output=True, text=True).stdout.split()
        paths = sorted(n for n in names if re.match(r"^phase\d[^/]*/[^/]+\.py$", n))
    parts = []
    for p in paths:
        text = subprocess.run(["git", "show", f"{CORPUS_COMMIT}:{p}"], cwd=root,
                              capture_output=True, encoding="utf-8").stdout
        parts.append(f"=== {p} ===\n{text}")
    return "\n\n".join(parts)


def long_prompt(depth, source, base=LLAMA_URL, _cache={}):
    """A user message whose chat-templated prompt is `depth` tokens: header,
    the corpus cut in token space, the question. Needs a running llama-server
    for its tokenizer (the same Qwen2.5 tokenizer at every size). The cut is
    detokenized, so a re-tokenization can differ by a token at the seam; the
    rows record the real input_tokens."""
    def tok(text, special=False):
        return _post(base + "/tokenize", {"content": text, "parse_special": special})["tokens"]
    if source not in _cache:
        empty = _post(base + "/apply-template",
                      {"messages": workloads.messages(LONG_HEADER + LONG_QUESTION)})["prompt"]
        _cache[source] = (len(tok(empty, True)), tok(corpus_text(source)))
    overhead, corpus = _cache[source]
    if depth - overhead > len(corpus):
        raise ValueError(f"{source} corpus has {len(corpus)} tokens, need {depth - overhead}")
    cut = _post(base + "/detokenize", {"tokens": corpus[:depth - overhead]})["content"]
    return LONG_HEADER + cut + LONG_QUESTION


def prompt_tokens(prompt, base=LLAMA_URL):
    """The exact token ids llama-server's chat endpoint would see for prompt."""
    text = _post(base + "/apply-template", {"messages": workloads.messages(prompt)})["prompt"]
    return _post(base + "/tokenize", {"content": text, "parse_special": True})["tokens"]


# --- shared ------------------------------------------------------------------

def _post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _wait_healthy(base, proc, timeout_s, log_path):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited ({proc.returncode}); see {log_path}")
        try:
            urllib.request.urlopen(base + "/health", timeout=1).read()
            return
        except OSError:
            time.sleep(1)
    proc.kill()
    raise RuntimeError(f"server not healthy after {timeout_s} s; see {log_path}")


def divergence(a, b):
    """Index of the first differing character, or None if identical."""
    if a == b:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

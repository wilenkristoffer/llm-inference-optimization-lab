"""
Phase 10 helpers: the candidate list and the Ollama chat call every suite
uses. Servers, streaming clients and the TunableOp env come from Phase 9's
common.py rather than being copied.

Only models already on disk take part (decided 2026-10-26), apart from the
three non-Qwen/Llama models the user chose on 2026-10-27. A candidate that
has gone missing is dropped by 01_preflight.py, not pulled.
"""

import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")


def _load(name, relative):
    path = os.path.join(HERE, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p9 = _load("p9common", "../phase9-optimization/common.py")
storage = p9.storage

OLLAMA = "http://127.0.0.1:11434"

# Ollama tag -> (quantization, family, thinks). "thinks" means the model
# reasons before answering unless the request sets think: false.
CANDIDATES = {
    "qwen2.5:1.5b": ("Q4_K_M", "qwen2.5", False),
    "qwen2.5:3b": ("Q4_K_M", "qwen2.5", False),
    "qwen2.5:3b-instruct-q8_0": ("Q8_0", "qwen2.5", False),
    "qwen2.5:7b": ("Q4_K_M", "qwen2.5", False),
    "qwen2.5:7b-instruct-q8_0": ("Q8_0", "qwen2.5", False),
    "qwen2.5:14b": ("Q4_K_M", "qwen2.5", False),
    "qwen3:14b": ("Q4_K_M", "qwen3", True),
    "qwen3.5:4b-bf16": ("BF16", "qwen3.5", True),
    "gemma3:12b": ("Q4_K_M", "gemma3", False),
    "gemma4:12b": ("Q4_K_M", "gemma4", True),
    "llama3.1:8b": ("Q4_K_M", "llama3.1", False),
    # Added 2026-10-27: non-Qwen/Llama models that fit 16 GB with tool support.
    "gpt-oss:20b": ("MXFP4", "gpt-oss", True),
    "nemotron-3.5-bolt:e4b": ("Q4_K_M", "nemotron3.5", True),
    "ministral-3:14b": ("Q4_K_M", "ministral3", False),
}

# Models that cannot stop reasoning: Ollama takes a level, not false, so
# "think off" means the lowest level, and the budget must leave room for it.
THINK_OFF = {"gpt-oss:20b": "low"}


def think_value(model, on):
    """The request's think field for a model that reasons by default."""
    if on:
        return True
    return THINK_OFF.get(model, False)

# vLLM's FP16 twins, already in WSL's HF cache (Phase 9). Served with TunableOp.
VLLM_CANDIDATES = ["qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct"]

# Pinned for every request. Penalties at 0 because Qwen3.5's model file sets
# presence_penalty 1.5 (Phase 6); num_ctx 8192 covers 50 of the agent's 51
# logged tasks.
OPTIONS = {"temperature": 0, "seed": 42, "repeat_penalty": 1.0,
           "presence_penalty": 0, "frequency_penalty": 0, "num_ctx": 8192}


def ollama_chat(model, messages, tools=None, think=None, num_predict=512,
                keep_alive=None, timeout=600, num_ctx=None, base=OLLAMA):
    """Native /api/chat, not streamed. Returns the parsed response, or
    {"error": <text>, "status": <code>} when Ollama refuses the request -
    a model without tool support is refused this way, and that is a result,
    not a crash. num_ctx overrides the pinned 8192 (suite E only); base is
    another Ollama, e.g. a side server with its own NUM_PARALLEL."""
    options = {**OPTIONS, "num_predict": num_predict}
    if num_ctx:
        options["num_ctx"] = num_ctx
    body = {"model": model, "messages": messages, "stream": False, "options": options}
    if tools is not None:
        body["tools"] = tools
    if think is not None:
        body["think"] = think
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    req = urllib.request.Request(base + "/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        try:
            text = json.loads(text).get("error", text)
        except json.JSONDecodeError:
            pass
        return {"error": text, "status": e.code}


def ollama_tags():
    """Installed models, name -> digest."""
    tags = json.load(urllib.request.urlopen(OLLAMA + "/api/tags", timeout=10))
    return {m["name"]: m["digest"] for m in tags["models"]}


def ollama_version():
    v = json.load(urllib.request.urlopen(OLLAMA + "/api/version", timeout=10))
    return "ollama " + v["version"]

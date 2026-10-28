"""
Phase 11, file 8: the adapter served unmerged, against the merged model.

Ollama 0.40.12 no longer takes LoRA adapters ("LoRA adapters are no longer
supported", for safetensors and GGUF alike). But Ollama serves qwen35 through
its bundled llama-server.exe, and llama-server takes --lora. So all three
configs run on that binary with the arguments Ollama starts it with (from its
server log), and only the weights differ:

    base     Ollama's bf16 GGUF of the student
    merged   the qwen3.5-mail:<name> blob from 06_merge_export.py
    lora     base + runs/<name>/lora.gguf, the adapter in f32, unrounded

06 lost part of the trained delta to bf16 rounding (logs/merge/<name>.json).
lora is the adapter exactly as trained, so if merged and lora agree, the
merged results stand for the trained model.

The adapter GGUF (--export, in WSL) uses the same name mapping and value-head
permutations as 06: a row permutation moves the rows of B, a column
permutation the columns of A. llama.cpp computes scale = alpha / rank.

Per config:
    check     greedy continuations from the start of each assistant turn of
              scripted val episodes; how often each config writes merged's text
    speed     cold prefill and decode tok/s on the first D prompt, from
              llama-server's own timings, median of 3
    episodes  D and D' with Phase 10's prompt (lora only by default), paired
              with the merged model's native runs in logs/finetuned/

Usage (Windows side; unloads Ollama's models, port 8090):
    python 08_serve_adapter.py --name t373-r16 --export
    python 08_serve_adapter.py --name t373-r16
    python 08_serve_adapter.py --name t373-r16 --configs base merged lora --no-episodes
    python 08_serve_adapter.py --name t373-r16 --report
"""

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase10-final-benchmark"))
merge_export = importlib.import_module("06_merge_export")
baseline = importlib.import_module("03_baseline")
common = baseline.common
mb = baseline.mb
qwen35fmt = baseline.qwen35fmt

OLLAMA_LIB = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "lib", "ollama")
SERVER = os.path.join(OLLAMA_LIB, "llama-server.exe")
PORT = 8090
URL = f"http://127.0.0.1:{PORT}"
# Ollama's command line for qwen3.5:4b-bf16 at num_ctx 8192 (server.log)
SERVER_ARGS = ["--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "--offline",
               "-c", "8192", "-np", "1", "--no-jinja", "--chat-template", "chatml",
               "--flash-attn", "auto", "-b", "1024", "-ub", "1024",
               "--context-shift", "--keep", "4"]
CHECK_EPISODES = 20
CHECK_TOKENS = 96
OUT = os.path.join(HERE, "logs", "serve")


# ---- WSL side -------------------------------------------------------------

def export(name, blob):
    """runs/<name>/lora.gguf: the adapter in llama.cpp's LoRA format."""
    import numpy as np
    import torch
    import gguf
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    import ftcommon

    run_dir = os.path.join(HERE, "runs", name)
    cfg = json.load(open(os.path.join(run_dir, "adapter_config.json")))
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "plain LoRA only"
    snap = snapshot_download(ftcommon.MODEL, revision=ftcommon.REVISION, local_files_only=True)
    index = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    adapter = {}
    with safe_open(os.path.join(run_dir, "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            base, ab = k.split(".lora_")
            adapter.setdefault(base.removeprefix("base_model.model."), {})[ab[0]] = f.get_tensor(k)
    tensors = {t.name: t for t in gguf.GGUFReader(blob).tensors}

    w = gguf.GGUFWriter(os.path.join(run_dir, "lora.gguf"), "qwen35")
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(cfg["lora_alpha"]))
    layouts = {}
    for module, ab in sorted(adapter.items()):
        assert module.startswith(merge_export.ADAPTER_PREFIX), module
        layer, kind = module[len(merge_export.ADAPTER_PREFIX):].split(".", 1)
        t = tensors[f"blk.{layer}.{merge_export.GGUF_NAMES[kind]}.weight"]
        hf_name = f"{merge_export.CKPT_PREFIX}{layer}.{kind}.weight"
        with safe_open(os.path.join(snap, index[hf_name]), "pt") as f:
            hf_w = f.get_tensor(hf_name)
        gg_w = torch.from_numpy(np.asarray(t.data).view(np.int16).copy()).view(
            torch.bfloat16).reshape(hf_w.shape)
        axis, idx = merge_export._layout(torch, hf_w, gg_w)
        layouts.setdefault(kind, set()).add(axis or "same")
        a, b = ab["A"].float(), ab["B"].float()  # A (r, in), B (out, r)
        if axis == "rows":
            b = b[idx]
        elif axis == "cols":
            a = a[:, idx]
        w.add_tensor(t.name + ".lora_a", a.contiguous().numpy())
        w.add_tensor(t.name + ".lora_b", b.contiguous().numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(json.dumps({k: sorted(v) for k, v in layouts.items()}), flush=True)


# ---- Windows side ---------------------------------------------------------

def _post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def weights_of(model):
    """The weights FROM of an Ollama tag, and the projector FROM."""
    mf = common.p9._post(common.OLLAMA + "/api/show", {"model": model})["modelfile"]
    froms = [l.split(None, 1)[1].strip() for l in mf.splitlines() if l.startswith("FROM ")]
    weights = max(froms, key=os.path.getsize)
    return weights, [f for f in froms if f != weights]


def unload_ollama():
    with urllib.request.urlopen(common.OLLAMA + "/api/ps", timeout=30) as r:
        loaded = json.loads(r.read()).get("models", [])
    for m in loaded:
        common.p9._post(common.OLLAMA + "/api/generate", {"model": m["name"], "keep_alive": 0})


class Server:
    """One llama-server.exe, with the GPU backend found as Phase 9 needed it."""

    def __init__(self, model, mmproj, lora, log_path):
        args = [SERVER, "--model", model, *SERVER_ARGS]
        if mmproj:
            args += ["--mmproj", mmproj]
        if lora:
            args += ["--lora", lora]
        hip = os.path.join(OLLAMA_LIB, "rocm_v7_1")
        env = {**os.environ, "GGML_BACKEND_PATH": os.path.join(hip, "ggml-hip.dll"),
               "PATH": hip + os.pathsep + OLLAMA_LIB + os.pathsep + os.environ["PATH"]}
        self.log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(args, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.perf_counter()
        while True:
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server exited; see {log_path}")
            try:
                with urllib.request.urlopen(URL + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1)
        self.load_s = round(time.perf_counter() - t0, 1)

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=60)
        self.log.close()


def complete(prompt, n_predict, cache=True, ignore_eos=False):
    return _post("/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 42,
        "repeat_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0,
        "stop": [] if ignore_eos else ["<|im_end|>", "<|im_start|>"],
        "ignore_eos": ignore_eos, "cache_prompt": cache})


def check_prompts():
    """The served prompt up to the start of each assistant turn, val episodes."""
    out = []
    with open(os.path.join(HERE, "data", "val.jsonl"), encoding="utf-8") as f:
        for line in list(f)[:CHECK_EPISODES]:
            ep = json.loads(line)
            msgs = [{"role": "system", "content": mb.SYSTEM},
                    {"role": "user", "content": mb.USER}] + ep["transcript"]
            for i, m in enumerate(msgs):
                if m["role"] == "assistant":
                    out.append(qwen35fmt.render_ollama(msgs[:i], mb.TOOLS))
    return out


def speed():
    prompt = qwen35fmt.render_ollama([{"role": "system", "content": mb.SYSTEM},
                                      {"role": "user", "content": mb.USER}], mb.TOOLS)
    complete(prompt, 8)  # warm-up
    runs = [complete(prompt, 256, cache=False, ignore_eos=True)["timings"] for _ in range(3)]
    return {"prompt_tokens": runs[0]["prompt_n"],
            "prefill_tok_s": round(statistics.median(r["prompt_per_second"] for r in runs), 1),
            "decode_tok_s": round(statistics.median(r["predicted_per_second"] for r in runs), 1)}


def step_fn():
    def step(messages, tools):
        t0 = time.perf_counter()
        try:
            r = complete(qwen35fmt.render_ollama(messages, tools), baseline.quality.NUM_PREDICT["D"])
        except urllib.error.HTTPError as e:
            return {"error": e.read().decode("utf-8", "replace")[:300],
                    "wall_ms": (time.perf_counter() - t0) * 1000}
        content, calls = baseline.parse_xml_calls(r.get("content") or "")
        return {"content": content, "thinking": "", "tool_calls": calls, "empty_output": False,
                "input_tokens": r.get("tokens_evaluated"), "output_tokens": r.get("tokens_predicted"),
                "wall_ms": (time.perf_counter() - t0) * 1000}
    return step


def episodes(name, config, suite):
    path = os.path.join(OUT, name, f"{config}-p10-{suite}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [ep for ep in baseline.load_suite(suite) if ep["id"] not in done]
    print(f"{config} {suite}: {len(done)} done, {len(todo)} to run "
          f"(~{len(todo) * 15 / 60:.0f} min)", flush=True)
    step = step_fn()
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for ep in todo:
            r = mb.run_episode(step, ep)
            row = {"id": ep["id"], **{k: r[k] for k in ("correct", "score", "reason", "turns",
                                                          "input_tokens", "output_tokens",
                                                          "wall_ms")},
                   "transcript": r["transcript"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {ep['id']} {'ok ' if r['correct'] else 'BAD'} {r['wall_ms'] / 1000:5.1f}s "
                  f"{'' if r['correct'] else r['reason'][:90]}", flush=True)


def run(name, configs, suites):
    tag = f"{merge_export.TAG}:{name}"
    base_w, mmproj = weights_of(baseline.STUDENT)
    merged_w, _ = weights_of(tag)
    lora = os.path.join(HERE, "runs", name, "lora.gguf")
    assert os.path.exists(lora), "run --export first"
    models = {"base": (base_w, None), "merged": (merged_w, None), "lora": (base_w, lora)}
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    state_path = os.path.join(OUT, name, "check.json")
    state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {}
    prompts = check_prompts()
    unload_ollama()
    for config in configs:
        model, adapter = models[config]
        srv = Server(model, mmproj[0] if mmproj else None, adapter,
                     os.path.join(OUT, name, f"{config}.server.log"))
        try:
            print(f"{config}: loaded in {srv.load_s} s", flush=True)
            texts = [complete(p, CHECK_TOKENS)["content"] for p in prompts]
            state[config] = {"load_s": srv.load_s, **speed(), "texts": texts}
            print(config, {k: v for k, v in state[config].items() if k != "texts"}, flush=True)
            with open(state_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(state, f, ensure_ascii=False)
            for suite in suites if config == "lora" else []:
                episodes(name, config, suite)
        finally:
            srv.stop()
    report(name)


def report(name):
    d = os.path.join(OUT, name)
    state = json.load(open(os.path.join(d, "check.json"), encoding="utf-8"))
    out = {"name": name, "configs": {}, "check": {}, "episodes": {}}
    for c, s in state.items():
        out["configs"][c] = {k: v for k, v in s.items() if k != "texts"}
    ref = state.get("merged", {}).get("texts")
    for c, s in state.items():
        if ref and c != "merged":
            pairs = list(zip(s["texts"], ref))
            out["check"][f"{c} vs merged"] = {
                "turns": len(pairs), "identical": sum(a == b for a, b in pairs),
                "same_first_40_chars": sum(a[:40] == b[:40] for a, b in pairs)}
    merged_dir = os.path.join(HERE, "logs", "finetuned", f"{merge_export.TAG}_{name}")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jsonl"):
            continue
        rows = {r["id"]: r for r in map(json.loads, open(os.path.join(d, fn), encoding="utf-8")) if r}
        n = len(rows)
        e = {"episodes": n, "correct_share": round(sum(r["correct"] for r in rows.values()) / n, 3),
             "s_per_episode": round(sum(r["wall_ms"] for r in rows.values()) / n / 1000, 1)}
        native = os.path.join(merged_dir, "native-p10-" + fn.split("-p10-")[1])
        if os.path.exists(native):
            m = {r["id"]: r for r in map(json.loads, open(native, encoding="utf-8")) if r}
            x, y, p = baseline.mcnemar({i: r["correct"] for i, r in m.items()},
                                       {i: r["correct"] for i, r in rows.items()})
            e["vs merged native"] = {"only_merged": x, "only_this": y, "p": p,
                                     "identical_transcripts": sum(
                                         m[i]["transcript"] == rows[i]["transcript"]
                                         for i in rows if i in m)}
        out["episodes"][fn[:-6]] = e
    print(json.dumps(out, indent=1))
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--export", action="store_true", help="write runs/<name>/lora.gguf (via WSL)")
    ap.add_argument("--configs", nargs="+", default=["base", "merged", "lora"])
    ap.add_argument("--suites", nargs="+", default=["D", "Dp"])
    ap.add_argument("--no-episodes", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--wsl-export", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--blob", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.wsl_export:
        export(args.name, args.blob)
    elif args.export:
        blob, _ = weights_of(baseline.STUDENT)
        cmd = (f"cd {merge_export.wsl_path(HERE)} && {merge_export.FT_PYTHON} 08_serve_adapter.py "
               f"--wsl-export --name {args.name} --blob {merge_export.wsl_path(blob)}")
        subprocess.run(["wsl", "-e", "bash", "-lc", cmd], check=True)
    elif args.report:
        report(args.name)
    else:
        run(args.name, args.configs, [] if args.no_episodes else args.suites)


if __name__ == "__main__":
    main()

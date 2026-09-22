"""review/smoke.py — stage 2: actually load the thing and make it talk.

Metadata lies. A GGUF can be mislabeled, a quant can be broken, an
architecture can be unsupported by this llama.cpp build, and a repo whose card
promises 128k context can carry a config that llama.cpp reads as 4k. None of
that is visible before the weights are on disk and loaded.

So the survivors of the metadata screen get loaded in a SUBPROCESS with
llama_cpp and asked a handful of fixed probe prompts. We record what only a
real load can tell us:

  fit/runtime  — load seconds, VRAM actually taken, context llama.cpp really
                 reports, prompt-eval and generation tokens/sec
  capability   — the raw completions, plus cheap coherence heuristics; the
                 text is also what the hugpy-agent judge reads (judge.py)

Subprocess, not in-process, deliberately: a bad GGUF can hard-abort the
process from native code, and the reviewer must survive that with a recorded
verdict instead of taking down the caller.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict

# Fixed probes: one instruction-following, one factual, one reasoning, one
# format-compliance. Same four for every model so results are comparable
# across runs — changing them invalidates comparison with stored reviews.
PROBES = [
    "Reply with exactly the word: OK",
    "In one sentence, what is the difference between TCP and UDP?",
    "A shelf holds 3 red books and 5 blue books. Two blue books are removed. "
    "How many books remain, and how many are blue? Answer briefly.",
    "List exactly three primary colors as a comma-separated line, nothing else.",
]

DEFAULT_TIMEOUT = int(os.environ.get("REVIEW_SMOKE_TIMEOUT") or 600)


@dataclass
class SmokeResult:
    model_path: str = ""
    ok: bool = False
    error: str | None = None
    load_seconds: float | None = None
    n_ctx_train: int | None = None
    n_ctx_used: int | None = None
    n_params: int | None = None
    vram_used_bytes: int | None = None
    prompt_tokens_per_sec: float | None = None
    gen_tokens_per_sec: float | None = None
    probes: list[dict] = field(default_factory=list)
    coherence: float | None = None       # 0..1, cheap heuristic
    gpu_offloaded: bool | None = None

    def to_dict(self):
        return asdict(self)


def _vram_used_bytes() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        return int(out.strip().splitlines()[0]) * 1024**2
    except Exception:
        return None


def _score_coherence(probes: list[dict]) -> float:
    """Deliberately crude: this is a smoke test, not an eval harness. It catches
    the failure that matters — a quant that loads but emits garbage, repetition
    or nothing at all. Nuanced quality is the judge's job (judge.py)."""
    if not probes:
        return 0.0
    hits = 0.0
    for i, p in enumerate(probes):
        text = (p.get("output") or "").strip()
        if not text:
            continue
        low = text.lower()
        words = low.split()
        # degenerate repetition: one token looping
        if len(words) > 8 and len(set(words)) <= max(2, len(words) // 8):
            continue
        hits += 0.5                                  # produced usable text
        if i == 0 and "ok" in low[:20]:
            hits += 0.5
        elif i == 1 and ("connection" in low or "reliab" in low or "datagram" in low):
            hits += 0.5
        elif i == 2 and "6" in text and "3" in text:
            hits += 0.5
        elif i == 3 and low.count(",") == 2:
            hits += 0.5
    return round(hits / len(probes), 3)


# The child process. Kept as a string so the parent can run it under a
# DIFFERENT interpreter — llama_cpp lives in the worker venv, which is not
# necessarily the venv running the Flask app or the CLI.
_CHILD = r'''
import json, sys, time
spec = json.loads(sys.stdin.read())
out = {"ok": False}
try:
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(model_path=spec["model_path"], n_ctx=spec["n_ctx"],
                n_gpu_layers=spec["n_gpu_layers"], verbose=False)
    out["load_seconds"] = round(time.time() - t0, 3)
    try:
        out["n_ctx_train"] = int(llm.n_ctx_train())
    except Exception:
        pass
    try:
        out["n_ctx_used"] = int(llm.n_ctx())
    except Exception:
        pass
    # sample VRAM HERE, while the weights are resident — the parent can only
    # see the card after we exit, by which point it reads as baseline again
    try:
        import subprocess as _sp
        out["vram_mib"] = int(_sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip().splitlines()[0])
    except Exception:
        pass
    probes, ptoks, pdur, gtoks, gdur = [], 0, 0.0, 0, 0.0
    for prompt in spec["probes"]:
        t1 = time.time()
        res = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=spec["max_tokens"], temperature=0.0)
        dt = time.time() - t1
        text = (res["choices"][0]["message"].get("content") or "")
        usage = res.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        ptoks += pt; gtoks += ct; gdur += dt
        probes.append({"prompt": prompt, "output": text.strip()[:2000],
                       "completion_tokens": ct, "seconds": round(dt, 3)})
    out["probes"] = probes
    if gdur > 0:
        out["gen_tokens_per_sec"] = round(gtoks / gdur, 2)
        out["prompt_tokens_per_sec"] = round(ptoks / gdur, 2)
    out["ok"] = True
except BaseException as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print("---SMOKE-JSON---")
print(json.dumps(out))
'''


def _python_with_llama_cpp() -> str:
    """Interpreter that can import llama_cpp. The worker venv is where the
    fleet builds it; fall back to whatever is running us."""
    for cand in (os.environ.get("REVIEW_LLAMA_PYTHON"),
                 "/home/solcatcher/hugpy/hugpy-worker/venv/bin/python",
                 sys.executable):
        if not cand or not os.path.isfile(cand):
            continue
        try:
            subprocess.check_output([cand, "-c", "import llama_cpp"],
                                    stderr=subprocess.DEVNULL, timeout=120)
            return cand
        except Exception:
            continue
    return sys.executable


def smoke_test(model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1,
               max_tokens: int = 128, timeout: int = DEFAULT_TIMEOUT,
               probes: list[str] | None = None) -> SmokeResult:
    """Load a GGUF and run the probe set. Never raises — a crash, a hang or an
    unsupported architecture all come back as a recorded failure."""
    r = SmokeResult(model_path=model_path)
    if not os.path.isfile(model_path):
        r.error = f"model file not found: {model_path}"
        return r

    before = _vram_used_bytes()
    spec = {"model_path": model_path, "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers, "max_tokens": max_tokens,
            "probes": probes or PROBES}
    try:
        proc = subprocess.run(
            [_python_with_llama_cpp(), "-c", _CHILD],
            input=json.dumps(spec), text=True, capture_output=True,
            timeout=timeout)
    except subprocess.TimeoutExpired:
        r.error = f"timed out after {timeout}s (load or generation hung)"
        return r
    except Exception as exc:
        r.error = f"could not launch probe process: {type(exc).__name__}: {exc}"
        return r

    marker = "---SMOKE-JSON---"
    blob = proc.stdout.split(marker, 1)[1].strip() if marker in proc.stdout else ""
    if not blob:
        tail = (proc.stderr or proc.stdout or "").strip()[-600:]
        r.error = f"probe process produced no result (exit {proc.returncode}): {tail}"
        return r
    try:
        data = json.loads(blob)
    except Exception as exc:
        r.error = f"unparseable probe result: {exc}"
        return r

    r.ok = bool(data.get("ok"))
    r.error = data.get("error")
    r.load_seconds = data.get("load_seconds")
    r.n_ctx_train = data.get("n_ctx_train")
    r.n_ctx_used = data.get("n_ctx_used")
    r.prompt_tokens_per_sec = data.get("prompt_tokens_per_sec")
    r.gen_tokens_per_sec = data.get("gen_tokens_per_sec")
    r.probes = data.get("probes") or []
    r.coherence = _score_coherence(r.probes)
    # VRAM the child held once loaded, minus what the card was already using.
    # An estimate, not a fact: anything else scheduling on the GPU during the
    # run lands in this delta too.
    loaded = data.get("vram_mib")
    if before is not None and loaded is not None:
        delta = int(loaded) * 1024**2 - before
        r.vram_used_bytes = delta if delta > 0 else None
    r.gpu_offloaded = (n_gpu_layers != 0) and bool(r.vram_used_bytes)
    return r

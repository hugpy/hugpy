"""hugpy-model-audit — decide, per model central holds, whether it is WORKING,
a BROKEN DOWNLOAD, a FAULTY MODEL or MISCONFIGURED, from evidence alone.

The sweep a model must pass before it enters the fleet, and the sweep that
finds what to eliminate among the models already in it. Every verdict carries
its ``why`` (one line) and its ``log`` (the verbatim evidence: parsed header
facts, byte counts, the worker's full load error), so a grade of 0 in the
registry is never "unknown".

Verdicts (per model; per-worker evidence rides along):

  not_downloaded    no model files on central
  broken_download   a file is missing / short / long vs its expected bytes, a
                    GGUF whose tensor data runs past EOF, an ``.incomplete``
                    marker, a safetensors header whose byte ranges disagree
                    with the file size, a shard index naming absent shards
  faulty_model      complete, right-sized, structurally wrong: bad GGUF magic
                    or version, tensor dims that contradict the file's own
                    metadata (llama.cpp ``check_tensor_dims``), unparsable
                    ``config.json`` / safetensors header
  misconfigured     files fine, config wrong: a pin (or the effective pick)
                    that is an mmproj/``clip`` projector, a MoE with no
                    ``n_cpu_moe`` too big for any GPU, a dense model at
                    ``n_gpu_layers=-1`` too big for every GPU, an override
                    stored under an alias key only, a catalog task that the
                    weights contradict (``mislabeled_task``), a fine-tune whose
                    appended <think>/</think> rows were never trained (fix:
                    "output delimiter repair" — hugpy.json ``output_repair``)
  mislabeled_task   the catalog says text-generation, the weights say otherwise
                    (Wan video GGUF, whisper, ...) — detected task reported
  suite_mismatch    a non-text model graded 0 by a text suite: not a failure
  static_ok         every static check passed (not smoke-tested)
  working           static_ok AND a ``--smoke`` chat on a hot worker answered
  unsupported       framework ``comfy``: presence check only

ORDER is load-bearing: size/truncation checks run BEFORE structural checks (a
truncated file also fails structurally), config checks BEFORE smoke (admission
refuses in under a second and says nothing about the model). ``eliminate`` is
true ONLY for faulty_model (complete, structurally wrong — re-downloading
reproduces it). A broken download is "re-provision", full stop.

EXPECTED VALUES COME FROM THE INSTALL MANIFEST, NEVER THE HUB (operator
2026-09-23: "no reason at all to ever call hugging face for anything after a
model is downloaded"). hugpy.json ``manifest.files[{path, bytes, sha256}]`` is
captured once at download time; this audit reads it and makes no network call
to Hugging Face. A model without a manifest gets an ``info`` finding
(``no_manifest``) and is judged from local structure plus the legacy
``quants`` byte counts — never escalated for lack of upstream evidence.
``hugpy-model-manifest-backfill`` writes the missing manifests once.

``--hash`` (opt-in) verifies sha256 of manifest files that carry one and are
under ``--hash-max-bytes`` (default 2 GB). Size + structure checks cannot prove
a large file bit-exact; only a full hash can, and hashing hundreds of GB is
deliberately not the default.

Read-only against the fleet. The only writes: the report files, the optional
``--record`` grade (suite ``integrity``) plus, with it, a derived
``hugpy.json["output_repair"]`` block (dead-delimiter repair), and the optional ``--smoke`` chats,
which only target models already HOT on a worker (a smoke never triggers a
download or a cold load).

Exit codes: 0 no eliminate candidates, 1 some, 2 tool error.
Stdlib only; hugpy_storage / hugpy_engine are imported lazily.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

DEFAULT_CENTRAL = "http://127.0.0.1:7002"
CENTRAL_ENV_VARS = ("HUGPY_BASE_URL", "HUGPY_CENTRAL", "HUGPY_URL")
HUGPY_ENV_FILE = "/srv/hugpy/etc/hugpy.env"
FALLBACK_PROJECTS_HOME = "/mnt/16T_toshiba/llm_storage/projects"
CENTRAL_TIMEOUT = 90.0
HASH_MAX_BYTES = 2 * 10 ** 9
SMOKE_TIMEOUT = 120.0
SUITE = "integrity"
TEXT_SUITES = ("hugpy-native-v2",)

NOT_DOWNLOADED = "not_downloaded"
BROKEN = "broken_download"
FAULTY = "faulty_model"
MISCONFIG = "misconfigured"
STATIC_OK = "static_ok"
WORKING = "working"
UNSUPPORTED = "unsupported"
MISLABELED = "mislabeled_task"
SUITE_MISMATCH = "suite_mismatch"
VERDICTS = (NOT_DOWNLOADED, BROKEN, FAULTY, MISCONFIG, MISLABELED, SUITE_MISMATCH,
            WORKING, STATIC_OK, UNSUPPORTED)
# Severity order when several findings disagree: the first present wins.
_PRECEDENCE = (NOT_DOWNLOADED, BROKEN, FAULTY, MISCONFIG)

GIB = float(2 ** 30)
PROJECTOR_ARCHS = {"clip", "mmproj"}
PROJECTOR_HINTS = ("mmproj", "projector")
# Repacked types removed from llama.cpp (repacking now happens at load time):
# a file carrying them is refused by every current llama-server.
REMOVED_GGML_TYPES = {31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
                      36: "IQ4_NL_4_4", 37: "IQ4_NL_4_8", 38: "IQ4_NL_8_8"}
WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".onnx",
               ".msgpack", ".h5", ".npz", ".sft")
PARTIAL_SUFFIXES = (".incomplete", ".part", ".partial", ".downloading", ".aria2")
IGNORED_NAMES = ("hugpy.json",)

TEXT_TASKS = {"text-generation", "image-text-to-text", "conversational",
              "text2text-generation", "summarization", "text-summarization"}
CHAT_TASKS = {"text-generation", "image-text-to-text", "conversational"}
# GGUF general.architecture -> the task the weights actually serve.
ARCH_TASKS = {
    "wan": "text-to-video", "ltxv": "text-to-video", "hunyuan-video": "text-to-video",
    "mochi": "text-to-video", "cogvideox": "text-to-video",
    "flux": "text-to-image", "sd1": "text-to-image", "sdxl": "text-to-image",
    "sd3": "text-to-image", "lumina2": "text-to-image", "hidream": "text-to-image",
    "chroma": "text-to-image", "qwen_image": "text-to-image",
    "whisper": "automatic-speech-recognition",
    "bert": "feature-extraction", "nomic-bert": "feature-extraction",
    "jina-bert-v2": "feature-extraction",
}
# transformers config.json ``architectures`` suffix -> task (first match wins).
CONFIG_ARCH_TASKS = (
    ("WhisperFor", "automatic-speech-recognition"),
    ("ForSpeechSeq2Seq", "automatic-speech-recognition"),
    ("ForCTC", "automatic-speech-recognition"),
    ("ForImageClassification", "image-classification"),
    ("ForObjectDetection", "object-detection"),
    ("DetrFor", "object-detection"),
    ("ForSemanticSegmentation", "image-segmentation"),
    ("ForDepthEstimation", "depth-estimation"),
    ("ForSequenceClassification", "text-classification"),
    ("ForTokenClassification", "token-classification"),
)
DIFFUSERS_TASKS = (("Wan", "text-to-video"), ("LTX", "text-to-video"),
                   ("Flux", "text-to-image"), ("StableDiffusion", "text-to-image"))


# ── data shapes ──────────────────────────────────────────────────────────────

@dataclass
class Finding:
    check: str
    verdict: str                 # the verdict this finding implies ("info" = none)
    detail: str
    file: Optional[str] = None
    worker: Optional[str] = None
    fix: Optional[str] = None


@dataclass
class ModelReport:
    model_key: str
    framework: str
    hub_id: Optional[str]
    primary_task: Optional[str]
    destination: Optional[str]
    verdict: str = STATIC_OK
    eliminate: bool = False
    why: str = ""
    findings: list = field(default_factory=list)
    log: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    suggested_fix: Optional[str] = None
    detected_task: Optional[str] = None
    recorded_grade: Optional[dict] = None
    evidence: dict = field(default_factory=lambda: {"central_files": [], "workers": {}})
    smoke: dict = field(default_factory=dict)
    # dead-delimiter output repair derived from the weights (see
    # _check_dead_delimiters); written to hugpy.json by record().
    output_repair: Optional[dict] = None
    output_repair_applied: bool = False

    def add(self, f: Finding) -> None:
        self.findings.append(f)


# ── HTTP ─────────────────────────────────────────────────────────────────────

def _request(url: str, *, method: str = "GET", payload: Optional[dict] = None,
             token: Optional[str] = None, timeout: float = CENTRAL_TIMEOUT):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "hugpy-model-audit")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        body, status = exc.read(), exc.code
    try:
        return status, json.loads(body.decode("utf-8", "replace") or "null")
    except ValueError:
        return status, body.decode("utf-8", "replace")


def fetch_json(url: str, token: Optional[str] = None, timeout: float = CENTRAL_TIMEOUT) -> Any:
    status, body = _request(url, token=token, timeout=timeout)
    if status != 200:
        raise RuntimeError(f"GET {url} -> HTTP {status}")
    return body


def central_url(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    for var in CENTRAL_ENV_VARS:
        if os.environ.get(var):
            return os.environ[var].rstrip("/")
    return DEFAULT_CENTRAL


def api_key() -> Optional[str]:
    """The operator key, looked up exactly like ``py/tooling/pkg_promote.py``:
    env ``PKG_PROMOTE_API_KEY`` / ``HUGPY_API_KEY``, else the file named by
    ``PKG_PROMOTE_API_KEY_FILE`` (default ``~/.config/pkg_src/api_key``)."""
    k = os.environ.get("PKG_PROMOTE_API_KEY") or os.environ.get("HUGPY_API_KEY")
    if k:
        return k.strip()
    path = os.environ.get("PKG_PROMOTE_API_KEY_FILE") or os.path.join(
        os.path.expanduser("~"), ".config", "pkg_src", "api_key")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _env_file_value(name: str, path: str = HUGPY_ENV_FILE) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    except OSError:
        pass
    return None


def projects_home() -> str:
    return (os.environ.get("PROJECTS_HOME") or _env_file_value("PROJECTS_HOME")
            or FALLBACK_PROJECTS_HOME)


# ── load_reports classifier ──────────────────────────────────────────────────

_NEED_RE = re.compile(r"needs\s+~?\s*([\d.]+)\s*GB", re.I)


def split_loader_stderr(text: str) -> tuple:
    """``(head, loader_stderr)`` — the llama-server child's captured stderr
    rides after ``Loader stderr:`` in a load error."""
    if not text:
        return "", None
    i = text.find("Loader stderr:")
    if i < 0:
        return text, None
    return text[:i].rstrip(), text[i + len("Loader stderr:"):].strip()


def classify_load_error(text: Optional[str]) -> Optional[dict]:
    """Map one worker ``load_reports[key].error`` string to
    ``{"class", "reason", "need_bytes"?}``. Classes: ``misconfigured``,
    ``faulty_model`` (caller demotes to broken_download when the size check
    failed), ``fit`` (a VRAM refusal — misconfigured only if the need exceeds
    the card's TOTAL, else contention), ``transient``, ``info``. None for an
    empty string."""
    if not text:
        return None
    t = str(text)
    low = t.lower()
    if "unsupported model architecture" in low and ("'clip'" in low or '"clip"' in low):
        return {"class": MISCONFIG, "reason": "mmproj: loader refused arch 'clip' (a projector was loaded as the model)"}
    if ("check_tensor_dims" in low or "wrong shape" in low or "bad magic" in low
            or "invalid magic" in low or "failed to read magic" in low
            or "gguf_init_from_file" in low or "tensor data is not within the file bounds" in low):
        m = re.search(r"(check_tensor_dims:[^\n]*|tensor '[^']*' has wrong shape[^\n]*|"
                      r"gguf_init_from_file[^\n]*|[^\n]*magic[^\n]*|"
                      r"[^\n]*not within the file bounds[^\n]*)", t, re.I)
        return {"class": FAULTY, "reason": (m.group(1) if m else "loader rejected the file").strip()}
    if "not local" in low and "probe does not download" in low:
        return {"class": "info", "reason": "not local on this worker (probe does not download)"}
    m = _NEED_RE.search(t)
    if m and ("vram" in low or "won't fit" in low or "wont fit" in low or "fit on gpu" in low):
        return {"class": "fit", "reason": "VRAM admission refusal",
                "need_bytes": int(float(m.group(1)) * 1e9)}
    if "partial gpu offload degenerate" in low:
        return {"class": "transient", "reason": "GPU held by protected residents (offload degenerate)"}
    if "workerunreachable" in low or "server disconnected" in low or "timed out" in low:
        return {"class": "transient", "reason": "worker unreachable during load"}
    if "needs the expert split" in low:
        return {"class": "info", "reason": "MoE needs --n-cpu-moe; this worker had no slot child to express it"}
    if "vision model loaded in-process" in low:
        return {"class": "info", "reason": "vision model loaded text-only in-process (no projector)"}
    if "unknown transformers sub-module" in low:
        return {"class": "info", "reason": "worker transformers build lacks the model's Auto class"}
    if "failed to load model from file" in low or "in-process load failed" in low:
        return {"class": "load_failed", "reason": "in-process load failed (no loader stderr captured)"}
    return {"class": "info", "reason": t.strip()}


# ── file walk ────────────────────────────────────────────────────────────────

_CHUNKSUMS_RE = re.compile(r"\.chunksums-\d+\.json$")
_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$", re.I)


def _ignored(rel: str) -> bool:
    base = os.path.basename(rel)
    if base.startswith(IGNORED_NAMES) or base.startswith(".llm_storage"):
        return True
    if _CHUNKSUMS_RE.search(base) or ".bak" in base or base.endswith((".lock", ".metadata")):
        return True
    return base in (".gitattributes", ".gitignore", "CACHEDIR.TAG")


def walk_model_dir(dest: str) -> dict:
    """``{files: {relpath: bytes}, partial: [relpath], cache_incomplete: [(relpath, bytes)]}``.
    ``.cache/`` is skipped for the file list but scanned for ``.incomplete``;
    other dot-dirs (``.git``) are skipped entirely."""
    files: dict = {}
    partial: list = []
    cache_incomplete: list = []
    for root, dirs, fns in os.walk(dest):
        rel_root = os.path.relpath(root, dest)
        top = rel_root.split(os.sep)[0]
        in_cache = top == ".cache"
        if rel_root == ".":
            dirs[:] = [d for d in dirs if not d.startswith(".") or d == ".cache"]
        for fn in fns:
            rel = os.path.normpath(os.path.join(rel_root, fn))
            try:
                size = os.path.getsize(os.path.join(root, fn))
            except OSError:
                continue
            if in_cache:
                if fn.endswith(".incomplete"):
                    cache_incomplete.append((rel, size))
                continue
            if fn.endswith(PARTIAL_SUFFIXES):
                partial.append(rel)
                continue
            if _ignored(rel):
                continue
            files[rel] = size
    return {"files": files, "partial": sorted(partial), "cache_incomplete": sorted(cache_incomplete)}


def hf_short_hash(name: str) -> str:
    """huggingface_hub's ``_short_hash`` — the prefix of a local-dir download's
    ``<hash>.<etag>.incomplete`` is this over ``<basename>.metadata``."""
    import base64
    import hashlib
    return base64.urlsafe_b64encode(hashlib.sha1(name.encode()).digest()).decode()


def map_incomplete(cache_rel: str, candidates) -> Optional[str]:
    """The repo-relative file an ``.incomplete`` belongs to, or None."""
    parts = cache_rel.split(os.sep)
    # .cache/huggingface/download/<subdir...>/<hash>.<etag>[.<x>].incomplete
    sub = os.sep.join(parts[3:-1])
    h = parts[-1].split(".", 1)[0]
    for rel in candidates:
        if os.path.dirname(rel) == sub and hf_short_hash(os.path.basename(rel) + ".metadata") == h:
            return rel
    return None


def is_weight(rel: str) -> bool:
    return rel.lower().endswith(WEIGHT_EXTS)


def is_projector(rel: str, arch: Optional[str] = None) -> bool:
    low = rel.lower()
    if arch and arch.lower() in PROJECTOR_ARCHS:
        return True
    parts = low.split(os.sep)
    return "mmproj" in parts[:-1] or any(h in parts[-1] for h in PROJECTOR_HINTS)


def logical_gguf(rel: str) -> str:
    """``foo-00001-of-00004.gguf`` -> ``foo.gguf`` (basename); else the basename."""
    base = os.path.basename(rel)
    m = _SHARD_RE.match(base)
    return f"{m.group('base')}.gguf" if m else base


# ── expected sizes: the install manifest (hugpy.json) ────────────────────────

def read_sidecar(dest: str) -> dict:
    try:
        with open(os.path.join(dest, "hugpy.json"), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def install_manifest(sidecar: dict) -> Optional[dict]:
    """hugpy.json ``manifest`` when it carries a file list, else None."""
    m = sidecar.get("manifest")
    return m if isinstance(m, dict) and isinstance(m.get("files"), list) and m["files"] else None


def expected_sizes(sidecar: dict) -> tuple:
    """``(per_file {rel: (bytes, source, sha256)}, per_logical_gguf {basename: (bytes, shards, source)})``.
    per_file comes from the install manifest; the legacy ``quants`` byte
    counts are used only when there is no manifest."""
    per_file: dict = {}
    per_logical: dict = {}
    man = install_manifest(sidecar)
    if man:
        for e in man["files"]:
            if isinstance(e, dict) and e.get("path") and e.get("bytes") is not None:
                per_file[os.path.normpath(e["path"])] = (int(e["bytes"]), "install manifest", e.get("sha256"))
        return per_file, per_logical
    for q in sidecar.get("quants") or []:
        if isinstance(q, dict) and q.get("file") and q.get("bytes"):
            per_logical[os.path.basename(q["file"])] = (int(q["bytes"]), q.get("shards"), "hugpy.json quants")
    return per_file, per_logical


def sha256_file(path: str, chunk: int = 8 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _json_broken(path: str) -> bool:
    if not path.endswith(".json"):
        return False
    return _lenient_json(path) is None


def _size_drift(rep: "ModelReport", rels, size_bad: set, how: str) -> None:
    """A size mismatch on a file whose own structure proves it complete means
    the file was replaced after install (another revision), not a broken
    download: demote to info."""
    rels = set(rels)
    for f in rep.findings:
        if f.check == "size" and f.verdict == BROKEN and f.file and (
                f.file in rels or any(logical_gguf(r) == os.path.basename(f.file) for r in rels)):
            f.verdict = "info"
            f.detail = f"changed since install (file internally complete: {how}): {f.detail}"
            f.fix = "optional: re-provision through hugpy so the file and its install manifest agree again"
    size_bad.difference_update(rels)


def _expected_lookup(rel: str, per_file: dict) -> Optional[tuple]:
    if rel in per_file:
        return per_file[rel]
    base = os.path.basename(rel)
    hits = [v for k, v in per_file.items() if os.path.basename(k) == base]
    return hits[0] if len(hits) == 1 else None


# ── structural readers (safetensors, torch zip) ──────────────────────────────

def zero_probe(path: str) -> Optional[str]:
    """None when the file's first 4 KiB hold data; else a verbatim description
    of the unwritten (all-zero) regions — an allocated file whose transfer never
    wrote its content (sparse/preallocated), i.e. a broken download."""
    try:
        st = os.stat(path)
        size = st.st_size
        with open(path, "rb") as fh:
            head = fh.read(4096)
            if not head or head.count(0) != len(head):
                return None
            samples = []
            for off in (0, size // 4, size // 2, max(0, size - 4096)):
                fh.seek(off)
                b = fh.read(4096)
                samples.append(f"@{off}:{'zero' if b.count(0) == len(b) else 'data'}")
            fh.seek(0)
            pos, lead = 0, None
            while pos < min(size, 1 << 30):
                b = fh.read(1 << 22)
                if not b:
                    break
                nz = len(b) - len(b.lstrip(b"\0"))
                if nz < len(b):
                    lead = pos + nz
                    break
                pos += len(b)
    except OSError:
        return None
    alloc = getattr(st, "st_blocks", 0) * 512
    return (f"first {lead if lead is not None else '>' + str(pos)} bytes are zero (unwritten); samples "
            f"{' '.join(samples)}; allocated {alloc} of {size} bytes")


def safetensors_check(path: str) -> dict:
    """``{ok, kind: ok|truncated|oversize|bad_header, header_len, expected_size, file_size, detail}``."""
    size = os.path.getsize(path)
    out = {"ok": False, "file_size": size, "header_len": None, "expected_size": None}
    z = zero_probe(path)
    if z:
        return {**out, "kind": "zeroed", "detail": z}
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8:
                return {**out, "kind": "truncated", "detail": f"file is {size} bytes, shorter than the 8-byte header length"}
            n = int.from_bytes(raw, "little")
            out["header_len"] = n
            if n > 200 * 2 ** 20:
                return {**out, "kind": "bad_header", "detail": f"header length {n} is implausible (> 200 MiB)"}
            if 8 + n > size:
                return {**out, "kind": "truncated", "detail": f"header claims {n} bytes but file is {size} bytes"}
            hdr = json.loads(fh.read(n).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return {**out, "kind": "bad_header", "detail": f"unparsable safetensors header: {exc}"}
    except OSError as exc:
        return {**out, "kind": "bad_header", "detail": f"unreadable: {exc}"}
    end = 0
    for k, v in hdr.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        offs = v.get("data_offsets") or [0, 0]
        try:
            end = max(end, int(offs[1]))
        except (TypeError, ValueError, IndexError):
            return {**out, "kind": "bad_header", "detail": f"tensor {k!r} has bad data_offsets {offs!r}"}
    exp = 8 + n + end
    out["expected_size"] = exp
    if exp > size:
        return {**out, "kind": "truncated", "detail": f"tensor data ends at byte {exp} but file is {size} bytes (short by {exp - size})"}
    if exp < size:
        return {**out, "kind": "oversize", "detail": f"tensor data ends at byte {exp} but file is {size} bytes ({size - exp} trailing bytes)"}
    return {**out, "ok": True, "kind": "ok", "detail": f"header {n} bytes, {len(hdr) - ('__metadata__' in hdr)} tensors, data ends exactly at EOF ({size} bytes)"}


def torch_zip_check(path: str) -> Optional[dict]:
    """A zip-format torch checkpoint must end with an end-of-central-directory
    record; a legacy pickle returns None (not checkable)."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if fh.read(2) != b"PK":
            return None
        fh.seek(max(0, size - 65557))
        tail = fh.read()
    if b"PK\x05\x06" in tail:
        return {"ok": True, "detail": f"zip end-of-central-directory present ({size} bytes)"}
    return {"ok": False, "detail": f"zip checkpoint has no end-of-central-directory record in its last 64 KiB ({size} bytes) — truncated"}


# ── overrides ────────────────────────────────────────────────────────────────

def load_overrides() -> tuple:
    path = os.environ.get("SERVE_OVERRIDES_PATH") or os.path.join(projects_home(), "serve_overrides.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return (data if isinstance(data, dict) else {}), path
    except (OSError, ValueError):
        return {}, path


def pin_for_worker(ov: dict, forms) -> tuple:
    """``(file, source)`` — mirrors ``hugpy_engine.serve.overrides.resolve_override_gguf``:
    per-worker map (id or name, case-insensitive) wins over the model-wide pin."""
    want = {str(f).strip().lower() for f in forms if f}
    for name, val in (ov.get("gguf_file_by_worker") or {}).items():
        if str(name).strip().lower() in want and str(val or "").strip():
            return str(val).strip(), f"gguf_file_by_worker[{name}]"
    if ov.get("gguf_file"):
        return str(ov["gguf_file"]).strip(), "gguf_file"
    return None, None


def _find_local(pin: str, files) -> Optional[str]:
    base = os.path.basename(pin).lower()
    exact = [r for r in files if os.path.basename(r).lower() == base]
    if exact:
        return sorted(exact)[0]
    logical = sorted(r for r in files if r.lower().endswith(".gguf") and logical_gguf(r).lower() == base)
    if logical:
        return logical[0]           # shard 00001 of a split model
    sub = sorted(r for r in files if r.lower().endswith(".gguf") and base in os.path.basename(r).lower())
    return sub[0] if sub else None


# ── the audit ────────────────────────────────────────────────────────────────

def _gb(n) -> str:
    return f"{n / 1e9:.2f} GB" if n is not None else "?"


def _fmt_kv(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items() if v is not None)


class Context:
    """Fleet-wide inputs, fetched once."""

    def __init__(self, central: str, workers: list, overrides: dict, overrides_path: str,
                 grades: dict, catalog_keys: set, hash_check: bool = False,
                 hash_max_bytes: int = HASH_MAX_BYTES):
        self.central = central
        self.workers = workers
        self.overrides = overrides
        self.overrides_path = overrides_path
        self.grades = grades
        self.catalog_keys = catalog_keys
        self.hash_check = hash_check
        self.hash_max_bytes = hash_max_bytes
        self.online = [w for w in workers if w.get("status") == "online"]
        self.gpu_totals = {w["name"]: int(w.get("gpu_total_bytes_known") or 0) for w in workers}
        self.max_gpu = max([self.gpu_totals[w["name"]] for w in self.online] or [0])

    def worker_by_ref(self, ref: str) -> Optional[dict]:
        for w in self.workers:
            if ref in (w.get("name"), w.get("id")):
                return w
        return None


def grade_rows(central: str) -> dict:
    """``{model: [rows with a grade]}`` from ``/llm/model-metrics2``."""
    try:
        body = fetch_json(f"{central}/llm/model-metrics2?limit=5000")
    except Exception:  # noqa: BLE001
        return {}
    out: dict = {}
    for r in (body or {}).get("rows") or []:
        if r.get("grade") is None:
            continue
        out.setdefault(r.get("model_name"), []).append(r)
    return out


def _report_keys(key: str, ctx: Context) -> list:
    keys = [key]
    if "~" in key:
        bare = key.split("~", 1)[1]
        if bare not in ctx.catalog_keys:
            keys.append(bare)
    return keys


PEFT_TASKS = {"CAUSAL_LM": "text-generation", "SEQ_2_SEQ_LM": "text2text-generation",
              "SEQ_CLS": "text-classification", "TOKEN_CLS": "token-classification",
              "QUESTION_ANS": "question-answering", "FEATURE_EXTRACTION": "feature-extraction"}


def _lenient_json(path: str) -> Optional[dict]:
    """json.load, then once more with trailing commas stripped (upstream repo
    descriptors such as Hunyuan's ``config.json`` are not strict JSON)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return None
    for text in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            d = json.loads(text)
            return d if isinstance(d, dict) else {}
        except ValueError:
            continue
    return None


def detect_task_transformers(dest: str, files: dict) -> tuple:
    """``(task | None, evidence str)`` from config.json / model_index.json / adapter_config.json."""
    if "model_index.json" in files:
        try:
            with open(os.path.join(dest, "model_index.json"), "r", encoding="utf-8") as fh:
                cls = str((json.load(fh) or {}).get("_class_name") or "")
        except (OSError, ValueError):
            cls = ""
        for frag, task in DIFFUSERS_TASKS:
            if frag in cls:
                return task, f"model_index.json _class_name={cls}"
        return "diffusers-pipeline", f"model_index.json _class_name={cls or '?'}"
    if "config.json" in files:
        cfg = _lenient_json(os.path.join(dest, "config.json"))
        if cfg is None:
            return None, "config.json unparsable"
        if not (cfg.get("architectures") or cfg.get("model_type") or cfg.get("_class_name")):
            return "non-transformers", f"config.json has no architectures/model_type (keys={sorted(cfg)[:6]})"
        archs = cfg.get("architectures") or []
        mt = str(cfg.get("model_type") or "")
        for a in archs:
            for frag, task in CONFIG_ARCH_TASKS:
                if frag in str(a):
                    return task, f"config.json architectures={archs}"
        if mt == "whisper":
            return "automatic-speech-recognition", f"config.json model_type={mt}"
        if any(str(a).endswith(("ForCausalLM", "LMHeadModel", "ForConditionalGeneration")) for a in archs):
            return "text-generation", f"config.json architectures={archs}"
        if "modules.json" in files or any(str(a) in ("BertModel", "XLMRobertaModel", "NomicBertModel", "NewModel") for a in archs):
            return "feature-extraction", f"config.json architectures={archs}"
        return None, f"config.json architectures={archs} model_type={mt}"
    if "adapter_config.json" in files:
        cfg = _lenient_json(os.path.join(dest, "adapter_config.json")) or {}
        tt = str(cfg.get("task_type") or "")
        task = PEFT_TASKS.get(tt)
        return (task or "adapter"), f"adapter_config.json task_type={tt or '?'} base={cfg.get('base_model_name_or_path')}"
    return None, "no config.json"


def audit_model(row: dict, ctx: Context) -> ModelReport:
    key = row["model_key"]
    fw = row.get("framework") or "?"
    dest = row.get("destination") or ""
    rep = ModelReport(model_key=key, framework=fw, hub_id=row.get("hub_id"),
                      primary_task=row.get("primary_task"), destination=dest)
    log = rep.log
    ov = ctx.overrides.get(key) or {}

    # ── 0. presence ──
    if not dest or not os.path.isdir(dest):
        rep.add(Finding("presence", NOT_DOWNLOADED, f"destination {dest!r} does not exist on central",
                        fix="re-provision through hugpy, or drop the catalog entry"))
        log.append(f"presence: destination {dest!r} missing on central")
        return _worker_evidence(rep, row, ctx, sizes_ok=False)
    walk = walk_model_dir(dest)
    files = walk["files"]
    if not files:
        rep.add(Finding("presence", NOT_DOWNLOADED, f"no model files under {dest} (only markers)",
                        fix="re-provision through hugpy, or drop the catalog entry"))
        log.append(f"presence: {dest} holds no model files")
        return _worker_evidence(rep, row, ctx, sizes_ok=False)
    sidecar = read_sidecar(dest)
    man = install_manifest(sidecar)
    per_file, per_logical = expected_sizes(sidecar)
    if man:
        rep.evidence["manifest"] = {"revision": man.get("revision"), "source": man.get("source"),
                                    "captured_at": man.get("captured_at"), "n_files": len(per_file)}
        log.append(f"install manifest: {len(per_file)} files, source={man.get('source')}, "
                   f"revision={man.get('revision')}, captured {man.get('captured_at')}")
    else:
        rep.add(Finding("no_manifest", "info", ("hugpy.json has no install manifest" if sidecar else "no hugpy.json")
                        + " — audited from local structure"
                        + (" + legacy quants bytes" if per_logical else ""),
                        fix="hugpy-model-manifest-backfill --apply (one-time)"))
        log.append("install manifest: none (expected sizes from legacy quants only)")
    weights = sorted(r for r in files if is_weight(r))
    log.append(f"presence: {len(files)} files ({len(weights)} weight files, {_gb(sum(files.values()))}) under {dest}")

    # ── 1. size / completeness (BEFORE structure) ──
    size_bad: set = set()
    central_files = rep.evidence["central_files"]
    for rel in sorted(files):
        exp = _expected_lookup(rel, per_file)
        ent = {"file": rel, "bytes": files[rel]}
        if exp:
            ent.update(expected=exp[0], expected_src=exp[1])
            if exp[0] != files[rel] and not is_weight(rel) and not _json_broken(os.path.join(dest, rel)):
                ent["status"] = "changed_since_install"
                rep.add(Finding("size", "info", f"{rel}: {files[rel]} bytes, {exp[1]} recorded {exp[0]} — non-weight "
                                "file changed since install (parses fine)", file=rel))
                log.append(f"size {rel}: on disk {files[rel]} != {exp[1]} {exp[0]} (non-weight; changed since install)")
            elif exp[0] != files[rel]:
                size_bad.add(rel)
                ent["status"] = "size_mismatch"
                rep.add(Finding("size", BROKEN, f"{rel}: {files[rel]} bytes on disk, {exp[1]} says {exp[0]} "
                                f"({'short' if files[rel] < exp[0] else 'long'} by {abs(exp[0] - files[rel])})", file=rel,
                                fix="re-provision through hugpy so the file is re-fetched"))
                log.append(f"size {rel}: on disk {files[rel]} != expected {exp[0]} ({exp[1]})")
            elif is_weight(rel):
                ent["status"] = "size_ok"
                log.append(f"size {rel}: {files[rel]} bytes == {exp[1]}")
        if is_weight(rel) or rel.endswith(".json"):
            central_files.append(ent)
    # every file the install manifest records must still be on disk (stat, not
    # the walk: the walk hides markers/.gitattributes the manifest may list)
    missing_listed: list = []
    for mrel, (nbytes, src, _sha) in sorted(per_file.items()):
        if mrel not in files and not os.path.isfile(os.path.join(dest, mrel)):
            missing_listed.append(mrel)
            rep.add(Finding("size", BROKEN, f"{src} lists {mrel} ({nbytes} bytes) but it is not on disk", file=mrel,
                            fix="re-provision through hugpy"))
            log.append(f"presence {mrel}: listed in {src} ({nbytes} bytes) — missing on disk")
    # opt-in sha256 (only where the manifest carries one, under the size cap)
    if ctx.hash_check and man:
        for mrel, (nbytes, src, sha) in sorted(per_file.items()):
            if not sha or mrel in size_bad or mrel in missing_listed:
                continue
            if nbytes > ctx.hash_max_bytes:
                log.append(f"hash {mrel}: skipped ({_gb(nbytes)} > --hash-max-bytes {_gb(ctx.hash_max_bytes)})")
                continue
            try:
                got = sha256_file(os.path.join(dest, mrel))
            except OSError as exc:
                log.append(f"hash {mrel}: unreadable: {exc}")
                continue
            if got != str(sha).lower():
                size_bad.add(mrel)
                rep.add(Finding("hash", BROKEN, f"{mrel}: sha256 {got} != install manifest {sha}", file=mrel,
                                fix="re-provision through hugpy so the file is re-fetched"))
                log.append(f"hash {mrel}: MISMATCH {got} != {sha}")
            else:
                log.append(f"hash {mrel}: sha256 == install manifest")
    # logical GGUF sizes from hugpy.json quants (sum over shards)
    groups: dict = {}
    for rel in files:
        if rel.lower().endswith(".gguf"):
            groups.setdefault(logical_gguf(rel), []).append(rel)
    for name, (nbytes, shards, src) in per_logical.items():
        rels = groups.get(name) or groups.get(os.path.basename(name))
        if not rels:
            rep.add(Finding("size", BROKEN, f"{src} lists {name} ({nbytes} bytes) but it is not on disk", file=name,
                            fix="re-provision through hugpy"))
            log.append(f"size {name}: listed in {src} ({nbytes} bytes, shards={shards}) — missing on disk")
            continue
        tot = sum(files[r] for r in rels)
        if tot != nbytes:
            size_bad.update(rels)
            rep.add(Finding("size", BROKEN, f"{name}: {tot} bytes on disk over {len(rels)} file(s), {src} says {nbytes}",
                            file=name, fix="re-provision through hugpy"))
            log.append(f"size {name}: on disk {tot} ({len(rels)} files) != {nbytes} ({src}, shards={shards})")
        else:
            log.append(f"size {name}: {tot} bytes over {len(rels)} file(s) == {src}")
    # sidecar filename must exist
    sf = sidecar.get("filename")
    if sf and fw in ("gguf", "llama_cpp") and not _find_local(str(sf), files):
        rep.add(Finding("size", BROKEN, f"hugpy.json filename {sf!r} is not on disk", file=str(sf),
                        fix="re-provision through hugpy"))
        log.append(f"presence: hugpy.json filename {sf!r} missing on disk")
    # shard sets
    shard_sets: dict = {}
    for rel in files:
        m = _SHARD_RE.match(os.path.basename(rel))
        if m:
            shard_sets.setdefault((os.path.dirname(rel), m.group("base"), int(m.group("n"))), set()).add(int(m.group("i")))
    missing_shards: list = []
    for (d, base, n), have in shard_sets.items():
        missing = sorted(set(range(1, n + 1)) - have)
        if missing:
            names = [os.path.join(d, f"{base}-{i:05d}-of-{n:05d}.gguf") for i in missing]
            missing_shards += names
            rep.add(Finding("shards", BROKEN, f"split GGUF {base}: {len(missing)} of {n} shards missing: {', '.join(names)}",
                            file=names[0], fix="re-provision through hugpy"))
            log.append(f"shards {base}: have {sorted(have)} of 1..{n}; missing {names}")
    # partial markers
    for rel in walk["partial"]:
        target = rel.rsplit(".", 1)[0]
        if target in files:
            log.append(f"partial marker {rel}: target exists (stale marker)")
        elif is_weight(target) or target.endswith((".index.json", "config.json", "model_index.json")):
            rep.add(Finding("partial", BROKEN, f"{rel}: download never completed ({target} absent)", file=target,
                            fix="re-provision through hugpy"))
            log.append(f"partial marker {rel}: {target} absent -> interrupted download")
        else:
            log.append(f"partial marker {rel}: non-weight target {target} absent (ignored)")
    cand = set(files) | set(per_file) | set(missing_shards)
    for crel, csize in walk["cache_incomplete"]:
        tgt = map_incomplete(crel, cand)
        if tgt and tgt in files and tgt not in size_bad:
            log.append(f"cache {crel} ({csize} bytes) -> {tgt}: final file present (stale .incomplete)")
            rep.evidence.setdefault("stale_incomplete", []).append(crel)
            continue
        what = f"{tgt} absent" if tgt else "unmapped to a present/expected file"
        rep.add(Finding("incomplete", BROKEN if tgt else "info", f"interrupted download: {crel} ({csize} bytes), {what}",
                        file=tgt, fix="re-provision through hugpy (resume the download)"))
        log.append(f"cache {crel} ({csize} bytes): {what}")
    # index files naming absent shards
    for rel in sorted(files):
        if not rel.endswith(".index.json"):
            continue
        try:
            with open(os.path.join(dest, rel), "r", encoding="utf-8") as fh:
                wm = (json.load(fh) or {}).get("weight_map") or {}
        except (OSError, ValueError) as exc:
            if rel not in size_bad:
                rep.add(Finding("index", FAULTY, f"{rel}: unparsable ({exc})", file=rel))
            log.append(f"index {rel}: unparsable: {exc}")
            continue
        d = os.path.dirname(rel)
        need = sorted(set(wm.values()))
        absent = [s for s in need if os.path.normpath(os.path.join(d, s)) not in files]
        if absent:
            rep.add(Finding("index", BROKEN, f"{rel} names {len(absent)} absent shard(s): {', '.join(absent[:6])}"
                            f"{' …' if len(absent) > 6 else ''}", file=rel, fix="re-provision through hugpy"))
            log.append(f"index {rel}: {len(need)} shards named, absent: {absent}")
        else:
            log.append(f"index {rel}: all {len(need)} named shards present")

    # ── 2. structure ──
    gguf_facts: dict = {}
    if fw in ("gguf", "llama_cpp") or any(r.lower().endswith(".gguf") for r in files):
        _check_ggufs(rep, dest, files, size_bad, gguf_facts)
    if fw in ("transformers", "comfy") or any(r.endswith(".safetensors") for r in files):
        _check_transformers(rep, dest, files, size_bad, fw)
    if fw in ("transformers",) and not weights:
        rep.add(Finding("presence", BROKEN, "no weight files on central (config/tokenizer only)",
                        fix="re-provision through hugpy"))
        log.append("presence: no weight files (.safetensors/.bin/.gguf/...) found")

    # ── 3. config ──
    if fw in ("gguf", "llama_cpp"):
        _check_gguf_config(rep, row, ctx, files, gguf_facts)
    _check_override_alias(rep, key, row, ctx)
    if fw in ("gguf", "llama_cpp"):
        _scope_to_served(rep, row, ctx, files, gguf_facts)
    # task
    detected, why_task = None, ""
    if fw in ("gguf", "llama_cpp"):
        archs = {f.get("arch") for r, f in gguf_facts.items() if not is_projector(r, f.get("arch"))}
        archs.discard(None)
        for a in archs:
            if a in ARCH_TASKS:
                detected, why_task = ARCH_TASKS[a], f"GGUF general.architecture={a}"
        if not archs and gguf_facts and all(is_projector(r, f.get("arch")) for r, f in gguf_facts.items()):
            detected, why_task = "projector-only", "every GGUF on central is an mmproj/clip projector"
    elif fw == "transformers":
        detected, why_task = detect_task_transformers(dest, files)
    rep.detected_task = detected
    if why_task:
        log.append(f"task: catalog primary_task={row.get('primary_task')}; detected={detected} ({why_task})")
    if (row.get("primary_task") in CHAT_TASKS and detected and detected not in TEXT_TASKS
            and detected not in ("projector-only", "adapter")):
        rep.add(Finding("task", MISLABELED, f"catalog primary_task={row.get('primary_task')} but weights are {detected} ({why_task})",
                        fix=f"set primary_task to {detected!r} (hugpy.json / catalog); stop grading it with a text suite"))
    return _worker_evidence(rep, row, ctx, sizes_ok=not size_bad)


def _scope_to_served(rep: ModelReport, row: dict, ctx: Context, files: dict, facts: dict) -> None:
    """A GGUF repo ships many quants; only the served ones (catalog effective
    pick, per-worker pins, projectors) decide the verdict. Defects in the other
    variants stay in findings/log as ``info`` ("unserved variant")."""
    ov = ctx.overrides.get(rep.model_key) or {}
    names = [row.get("effective_gguf")] + [pin_for_worker(ov, _worker_forms(w))[0] for w in ctx.workers]
    served = set()
    for n in names:
        if not n:
            continue
        loc = _find_local(n, files)
        served.add(logical_gguf(loc or n).lower())
    langs = {logical_gguf(r).lower() for r in files if r.lower().endswith(".gguf") and not is_projector(r)}
    if not served & langs:
        return                      # nothing resolvable: every variant counts
    rep.evidence["served_variants"] = sorted(served & langs)
    for f in rep.findings:
        if f.verdict in (BROKEN, FAULTY) and f.file and f.file.lower().endswith(".gguf"):
            lg = logical_gguf(f.file).lower()
            if lg not in served and not is_projector(f.file):
                f.detail = f"unserved variant: {f.detail}"
                f.verdict = "info"


def _check_ggufs(rep: ModelReport, dest: str, files: dict, size_bad: set, facts: dict) -> None:
    try:
        from hugpy_storage.gguf_inspect import gguf_integrity, gguf_read_header, GGUFHeaderError
    except ImportError as exc:  # pragma: no cover — tree reader missing
        rep.log.append(f"gguf: reader unavailable ({exc})")
        return
    log = rep.log
    ggufs = sorted(r for r in files if r.lower().endswith(".gguf"))
    shard1: dict = {}
    for rel in ggufs:
        m = _SHARD_RE.match(os.path.basename(rel))
        if m and int(m.group("i")) != 1:
            continue
        try:
            shard1[(os.path.dirname(rel), m.group("base") if m else rel)] = gguf_read_header(os.path.join(dest, rel))
        except (GGUFHeaderError, OSError):
            pass
    for rel in ggufs:
        path = os.path.join(dest, rel)
        m = _SHARD_RE.match(os.path.basename(rel))
        meta = shard1.get((os.path.dirname(rel), m.group("base"))) if m and int(m.group("i")) != 1 else None
        try:
            hdr = gguf_read_header(path)
        except GGUFHeaderError as exc:
            hdr = None
            herr = str(exc)
        except OSError as exc:
            hdr = None
            herr = f"unreadable: {exc}"
        if hdr is None:
            f = gguf_integrity(path)
            facts[rel] = f
            z = zero_probe(path)
            if z:
                rep.add(Finding("gguf_header", BROKEN, f"{rel}: {herr}; {z}", file=rel, fix="re-provision through hugpy"))
                log.append(f"gguf {rel}: header error: {herr}; {z}")
                continue
            if rel in size_bad or f.get("truncated"):
                rep.add(Finding("gguf_header", BROKEN, f"{rel}: header cut short — {herr}", file=rel,
                                fix="re-provision through hugpy"))
            else:
                rep.add(Finding("gguf_header", FAULTY, f"{rel}: {herr}", file=rel))
            log.append(f"gguf {rel}: header error: {herr} (file_size={files[rel]})")
            continue
        f = gguf_integrity(path, header=hdr, meta_from=meta)
        facts[rel] = f
        removed = sorted({REMOVED_GGML_TYPES[t['type']] for t in hdr["tensors"] if t["type"] in REMOVED_GGML_TYPES})
        log.append(f"gguf {rel}: magic=GGUF version={hdr['version']} arch={f.get('arch')} n_tensors={hdr['n_tensors']} "
                   f"n_kv={hdr['n_kv']} alignment={hdr['alignment']} data_offset={hdr['data_offset']} "
                   f"data_end={f.get('data_end')} file_size={hdr['file_size']}"
                   + (f" unknown_type_tensors={f['unknown_type_tensors']}" if f.get("unknown_type_tensors") else "")
                   + (f" embedding_length={f.get('embedding_length')} vocab={f.get('vocab_size')}" if f.get("embedding_length") else "")
                   + (f" expert_count={f.get('expert_count')}" if f.get("expert_count") else ""))
        if f.get("truncated"):
            short = f["data_end"] - hdr["file_size"]
            rep.add(Finding("gguf_truncated", BROKEN, f"{rel}: tensor data runs to byte {f['data_end']} but file is "
                            f"{hdr['file_size']} bytes (truncated by {short})", file=rel, fix="re-provision through hugpy"))
            continue
        if rel in size_bad:
            if f.get("unknown_type_tensors"):
                continue        # extent unknowable: the size verdict stands
            _size_drift(rep, [rel], size_bad, f"data_end {f['data_end']} == file_size")
            log.append(f"gguf {rel}: size differs from the expected source but tensor data ends exactly at EOF "
                       "-> complete file of another revision")
        for inc in f.get("inconsistencies") or []:
            rep.add(Finding("gguf_tensor_dims", FAULTY, f"{rel}: check_tensor_dims: {inc} "
                            f"(metadata embedding_length={f.get('embedding_length')}, vocab={f.get('vocab_size')})", file=rel))
        if removed:
            rep.add(Finding("gguf_types", FAULTY, f"{rel}: uses tensor types removed from llama.cpp ({', '.join(removed)}); "
                            "every current llama-server refuses it", file=rel))
        if not m and not is_projector(rel, f.get("arch")):
            _check_dead_delimiters(rep, rel, path, hdr)


# ── untrained (collapsed) reasoning delimiters → output repair ──────────────
#
# A fine-tune that ADDS reasoning delimiter tokens (``<think>``/``</think>``)
# and never trains their embedding rows ships them at the resize-init value:
# open and close collapse onto ONE row, the same row as the vocabulary's other
# never-trained entries. The logit of a token is ``hidden · output_row``, so
# when the model's trained behaviour is "emit <think>", an ordinary token whose
# row points the same way but is marginally LONGER outscores the delimiter
# itself and greedy decoding prints it instead (seen live:
# LFM2.5-350M-home-assistant-sft printing "tvåvingeart"/"fjärilsart" where its
# SFT emits <think>/</think>).
#
# Operator ruling 2026-09-23: such a model is SERVABLE — verdict
# ``misconfigured`` with fix "output delimiter repair", never eliminate. The
# audit derives the repair from the weights and records it as
# ``hugpy.json["output_repair"]`` (written by :func:`record`, i.e. ``--record``
# and the admission runner); central's output path (hugpy_engine.output_repair)
# maps the glitch tokens back to <think>/</think>. Once the marker carries the
# block matching this file, the finding drops to ``info`` ("repair applied").
#
# GLITCH SET (the tokens the model can emit in the delimiters' place) — every
# NORMAL token whose output row (a) lies in the collapsed cluster, cos >= 0.99
# with the dead <think> row, and (b) OUTSCORES the delimiter along its own
# direction by >= 0.1% of its norm (projection onto the unit <think> row minus
# the <think> norm). For the LFM2.5 SFT this selects exactly the two observed
# strings (margins 0.00076 / 0.00065 on norm 0.2749); the next candidates
# (' has_', 'ü', ' int64') sit at <= 0.00004, i.e. quantization noise, and must
# NOT be filtered. "cos >= 0.999 alone" would be wrong in both directions: it
# selects 552 rows incl. ' python', "n't", 'ö', 'ø', ' well-known' (live text)
# and misses the two real glitches (cos 0.99878 / 0.99876).
#
# Scope, kept narrow so base models are never caught: only CONTROL/USER_DEFINED
# REASONING pairs (think/thinking/reasoning), only pairs that ARE the tail of
# the vocabulary (the resize_token_embeddings signature: the fine-tune appended
# them), and only when open == close. A *base* checkpoint whose pretrained
# vocab merely reserves an untrained <think> pair (Qwen3-8B-Base: ids
# 151667/151668 of 151936, collapsed) is not caught — it never learned to emit
# it. Tool-call pairs are deliberately excluded: 2026-09-23 fleet sweep — every
# Qwen2.5-Coder GGUF (and a qwen3 *base*) ships a collapsed <tool_call>/
# </tool_call> pair and serves text correctly; base LFM2 likewise carries
# collapsed <|tool_response_*|>/<|image_*|>/<|review_*|> pairs it never emits.
# Every trained reasoning pair in the same sweep (Qwen3/3.5/3.8, LFM2.5-VL,
# DAN-L3-R1, Keye-VL, Step3-VL) has open/close cos 0.14-0.79.
OUTPUT_REPAIR_KIND = "dead_think_delimiters"
OUTPUT_REPAIR_FIX = "output delimiter repair"
_DELIM_PAIR_RE = re.compile(r"^(?:<(?P<a>think|thinking|reasoning)>|<\|(?P<b>think|thinking|reasoning)_start\|>)$")
_DELIM_COS = 0.999          # open vs close row: identical up to quantization noise
_DELIM_NORM_TOL = 0.01
_GLITCH_COS = 0.99          # (a) in the collapsed cluster
_GLITCH_MARGIN = 0.001      # (b) outscores the delimiter by >= 0.1% of its norm
_GLITCH_MAX = 64            # a sanity cap: more than this is not a delimiter stand-in


def _f16(b: bytes, off: int) -> float:
    import struct
    return struct.unpack_from("<e", b, off)[0]


def _k_scale_min(j: int, q: bytes) -> tuple:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def _dequant_row(raw: bytes, gtype: int, n: int) -> Optional[list]:
    """Dequantize ONE row (ggml reference layouts). None for an unsupported type."""
    import struct
    if gtype == 0:
        return list(struct.unpack(f"<{n}f", raw))
    if gtype == 1:
        return list(struct.unpack(f"<{n}e", raw))
    if gtype == 30:
        return [struct.unpack("<f", b"\x00\x00" + raw[i:i + 2])[0] for i in range(0, 2 * n, 2)]
    out: list = []
    if gtype == 8:                                   # Q8_0: 32 vals / 34 B
        for o in range(0, len(raw), 34):
            d = _f16(raw, o)
            out += [d * v for v in struct.unpack_from("<32b", raw, o + 2)]
        return out
    if gtype == 12:                                  # Q4_K: 256 vals / 144 B
        for o in range(0, len(raw), 144):
            d, dmin = _f16(raw, o), _f16(raw, o + 2)
            sc, qs = raw[o + 4:o + 16], raw[o + 16:o + 144]
            for k in range(4):
                s1, m1 = _k_scale_min(2 * k, sc)
                s2, m2 = _k_scale_min(2 * k + 1, sc)
                q = qs[32 * k:32 * k + 32]
                out += [d * s1 * (x & 0xF) - dmin * m1 for x in q]
                out += [d * s2 * (x >> 4) - dmin * m2 for x in q]
        return out
    if gtype == 13:                                  # Q5_K: 256 vals / 176 B
        for o in range(0, len(raw), 176):
            d, dmin = _f16(raw, o), _f16(raw, o + 2)
            sc, qh, qs = raw[o + 4:o + 16], raw[o + 16:o + 48], raw[o + 48:o + 176]
            for k in range(4):
                s1, m1 = _k_scale_min(2 * k, sc)
                s2, m2 = _k_scale_min(2 * k + 1, sc)
                u1, u2 = 1 << (2 * k), 2 << (2 * k)
                q = qs[32 * k:32 * k + 32]
                out += [d * s1 * ((x & 0xF) + (16 if qh[l] & u1 else 0)) - dmin * m1 for l, x in enumerate(q)]
                out += [d * s2 * ((x >> 4) + (16 if qh[l] & u2 else 0)) - dmin * m2 for l, x in enumerate(q)]
        return out
    if gtype == 14:                                  # Q6_K: 256 vals / 210 B
        for o in range(0, len(raw), 210):
            ql, qh = raw[o:o + 128], raw[o + 128:o + 192]
            sc = struct.unpack_from("<16b", raw, o + 192)
            d = _f16(raw, o + 208)
            y = [0.0] * 256
            for h in range(2):
                L, H, S, Y = 64 * h, 32 * h, 8 * h, 128 * h
                for l in range(32):
                    i = l // 16
                    a, b, c = ql[L + l], ql[L + l + 32], qh[H + l]
                    y[Y + l] = d * sc[S + i] * (((a & 0xF) | ((c & 3) << 4)) - 32)
                    y[Y + l + 32] = d * sc[S + i + 2] * (((b & 0xF) | (((c >> 2) & 3) << 4)) - 32)
                    y[Y + l + 64] = d * sc[S + i + 4] * (((a >> 4) | (((c >> 4) & 3) << 4)) - 32)
                    y[Y + l + 96] = d * sc[S + i + 6] * (((b >> 4) | (((c >> 6) & 3) << 4)) - 32)
            out += y
        return out
    return None


def _cos_norm(a: list, b: list) -> tuple:
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if not na or not nb:
        return 0.0, na, nb
    return sum(x * y for x, y in zip(a, b)) / (na * nb), na, nb


def _bytes_to_unicode() -> dict:
    """GPT-2 byte-level BPE alphabet (inverse used to decode token strings)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _token_text(tok: str, model: Optional[str]) -> str:
    """The text a token decodes to (what the client sees), from the GGUF's own
    tokenizer model: gpt2 byte-level BPE or llama SentencePiece."""
    if model == "gpt2":
        inv = _bytes_to_unicode()
        try:
            return bytes(inv[c] for c in tok).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            return tok
    if model == "llama":
        return tok.replace("\u2581", " ")
    return tok


def _dequant_rows_np(np, raw: bytes, gtype: int, n_rows: int, ne0: int):
    """Vectorized dequantization of ``n_rows`` contiguous rows (ggml reference
    layouts). None for an unsupported type."""
    b = np.frombuffer(raw, dtype=np.uint8)
    if gtype == 0:
        return b.view("<f4").reshape(n_rows, ne0).astype(np.float64)
    if gtype == 1:
        return b.view("<f2").reshape(n_rows, ne0).astype(np.float64)
    if gtype == 30:
        return (b.view("<u2").astype(np.uint32) << 16).view("<f4").reshape(n_rows, ne0).astype(np.float64)
    if gtype == 8:
        B = b.reshape(-1, 34)
        d = B[:, :2].copy().view("<f2").astype(np.float64)
        q = B[:, 2:].view(np.int8).astype(np.float64)
        return (d * q).reshape(n_rows, ne0)
    if gtype == 14:                                                   # Q6_K
        B = b.reshape(-1, 210)
        nb = B.shape[0]
        ql = B[:, :128].reshape(nb, 2, 64).astype(np.int32)
        qh = B[:, 128:192].reshape(nb, 2, 32).astype(np.int32)
        sc = B[:, 192:208].copy().view(np.int8).reshape(nb, 2, 8).astype(np.float64)
        d = B[:, 208:210].copy().view("<f2").astype(np.float64).reshape(nb, 1)
        y = np.empty((nb, 256))
        lsc = np.arange(32) // 16
        for h in range(2):
            a, bb, c = ql[:, h, :32], ql[:, h, 32:], qh[:, h, :]
            qs = ((a & 15) | ((c & 3) << 4), (bb & 15) | (((c >> 2) & 3) << 4),
                  (a >> 4) | (((c >> 4) & 3) << 4), (bb >> 4) | (((c >> 6) & 3) << 4))
            for k in range(4):
                y[:, 128 * h + 32 * k:128 * h + 32 * k + 32] = d * sc[:, h, lsc + 2 * k] * (qs[k] - 32)
        return y.reshape(n_rows, ne0)
    if gtype in (12, 13):                                             # Q4_K / Q5_K
        bs = 144 if gtype == 12 else 176
        B = b.reshape(-1, bs)
        nb = B.shape[0]
        d = B[:, 0:2].copy().view("<f2").astype(np.float64).reshape(nb, 1)
        dmin = B[:, 2:4].copy().view("<f2").astype(np.float64).reshape(nb, 1)
        q = B[:, 4:16].astype(np.int32)
        scs, mns = [], []
        for j in range(8):
            if j < 4:
                scs.append(q[:, j] & 63)
                mns.append(q[:, j + 4] & 63)
            else:
                scs.append((q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4))
                mns.append((q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4))
        if gtype == 12:
            qh, qs = None, B[:, 16:144].astype(np.int32)
        else:
            qh, qs = B[:, 16:48].astype(np.int32), B[:, 48:176].astype(np.int32)
        y = np.empty((nb, 256))
        for k in range(4):
            lo, hi = qs[:, 32 * k:32 * k + 32] & 0xF, qs[:, 32 * k:32 * k + 32] >> 4
            if qh is not None:
                lo = lo + np.where(qh & (1 << (2 * k)), 16, 0)
                hi = hi + np.where(qh & (2 << (2 * k)), 16, 0)
            y[:, 64 * k:64 * k + 32] = d * scs[2 * k].reshape(nb, 1) * lo - dmin * mns[2 * k].reshape(nb, 1)
            y[:, 64 * k + 32:64 * k + 64] = d * scs[2 * k + 1].reshape(nb, 1) * hi - dmin * mns[2 * k + 1].reshape(nb, 1)
        return y.reshape(n_rows, ne0)
    return None


def _glitch_scan(path: str, hdr: dict, head: dict, row_bytes: int, ne0: int, n_vocab: int,
                 types: list, anchor: list, row) -> Optional[list]:
    """[(id, cos, margin)] of NORMAL tokens meeting the glitch-set criterion
    (block comment above), strongest margin first. numpy when importable
    (chunked, bounded memory), else a per-row stdlib pass. None when the head
    type cannot be dequantized."""
    na = sum(x * x for x in anchor) ** 0.5
    if not na:
        return None
    unit = [x / na for x in anchor]
    out = []
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        u = np.asarray(unit)
        chunk = max(1, (64 << 20) // max(1, ne0 * 8))
        base = hdr["data_offset"] + head["offset"]
        with open(path, "rb") as fh:
            for start in range(0, n_vocab, chunk):
                n = min(chunk, n_vocab - start)
                fh.seek(base + start * row_bytes)
                M = _dequant_rows_np(np, fh.read(n * row_bytes), head["type"], n, ne0)
                if M is None:
                    return None
                proj = M @ u
                norms = np.linalg.norm(M, axis=1)
                cos = np.divide(proj, norms, out=np.zeros_like(proj), where=norms > 0)
                for k in np.nonzero((cos >= _GLITCH_COS) & (proj - na >= _GLITCH_MARGIN * na))[0]:
                    i = start + int(k)
                    if types[i] == 1:
                        out.append((i, float(cos[k]), float(proj[k] - na)))
    else:
        for i in range(n_vocab):
            if types[i] != 1:
                continue
            r = row(i)
            if r is None:
                return None
            p = sum(x * y for x, y in zip(r, unit))
            nr = sum(x * x for x in r) ** 0.5
            if nr and p / nr >= _GLITCH_COS and p - na >= _GLITCH_MARGIN * na:
                out.append((i, p / nr, p - na))
    out.sort(key=lambda t: -t[2])
    return out


def _check_dead_delimiters(rep: ModelReport, rel: str, path: str, hdr: dict) -> None:
    """``misconfigured`` + a derived ``output_repair`` block when a fine-tune-
    appended reasoning delimiter pair has collapsed (untrained) OUTPUT rows —
    see the block comment above. ``info`` when the model's hugpy.json already
    carries the matching repair."""
    kv = hdr.get("kv") or {}
    if not isinstance(kv.get("tokenizer.ggml.token_type"), dict):
        return
    try:
        from hugpy_storage.gguf_inspect import _gguf_metadata, gguf_tensor_nbytes
    except ImportError:
        return
    md = _gguf_metadata(path, (".ggml.tokens", ".ggml.token_type", ".ggml.model"))
    toks, types = md.get(".ggml.tokens"), md.get(".ggml.token_type")
    tok_model = md.get(".ggml.model")
    if not toks or not types or len(toks) != len(types):
        return
    last_normal = max((i for i, t in enumerate(types) if t == 1), default=-1)
    ids = {s: i for i, s in enumerate(toks) if types[i] in (3, 4)}
    pairs = []
    for s, i in ids.items():
        mm = _DELIM_PAIR_RE.match(s)
        if not mm:
            continue
        close = f"</{mm.group('a')}>" if mm.group("a") else f"<|{mm.group('b')}_end|>"
        j = ids.get(close)
        # the resize signature: the pair IS the tail of the vocabulary
        if j is not None and min(i, j) > last_normal and max(i, j) == len(toks) - 1:
            pairs.append((s, i, close, j))
    if not pairs:
        return
    tens = {t["name"]: t for t in hdr["tensors"]}
    head = tens.get("output.weight") or tens.get("token_embd.weight")
    if not head or len(head["dims"]) != 2 or head["dims"][1] != len(toks):
        return
    head_name = "output.weight" if "output.weight" in tens else "token_embd.weight (tied: no output.weight)"
    ne0 = int(head["dims"][0])
    row_bytes = gguf_tensor_nbytes(head["type"], [ne0])
    if not row_bytes:
        return

    def row(i: int) -> Optional[list]:
        with open(path, "rb") as fh:
            fh.seek(hdr["data_offset"] + head["offset"] + i * row_bytes)
            return _dequant_row(fh.read(row_bytes), head["type"], ne0)

    for s, i, close, j in pairs:
        a, b = row(i), row(j)
        if a is None or b is None:
            rep.log.append(f"gguf {rel}: delimiter check skipped — {head_name} ggml type {head['type']} not dequantizable here")
            return
        cos, na, nb = _cos_norm(a, b)
        maxd = max(abs(x - y) for x, y in zip(a, b))
        rep.log.append(f"gguf {rel}: delimiter pair {s} (id {i}) / {close} (id {j}) on {head_name}: "
                       f"cos={cos:.5f} max_abs_diff={maxd:.5f} norms={na:.4f},{nb:.4f} last_text_token_id={last_normal}")
        if cos < _DELIM_COS or abs(na - nb) > _DELIM_NORM_TOL * max(na, nb):
            continue
        scan = _glitch_scan(path, hdr, head, row_bytes, ne0, len(toks), types, a, row)
        if scan is None:
            rep.log.append(f"gguf {rel}: glitch-token scan skipped — {head_name} ggml type {head['type']} "
                           "not dequantizable here")
            return
        rep.log.append(f"gguf {rel}: glitch scan (NORMAL rows, cos>={_GLITCH_COS} with {s} and projection "
                       f">= {s} norm*(1+{_GLITCH_MARGIN})): " + (", ".join(
                           f"{k} {toks[k]!r} cos={c:.5f} margin={m:.5f}" for k, c, m in scan[:10]) or "none"))
        measured = {"cos": round(cos, 5), "max_abs_diff": round(maxd, 5), "norm": round(na, 4),
                    "head": head_name, "ggml_type": head["type"],
                    "glitch": [{"id": k, "cos": round(c, 5), "margin": round(m, 5)} for k, c, m in scan]}
        base = (f"{rel}: added delimiter tokens {s} (id {i}) and {close} (id {j}) — CONTROL tokens appended as the "
                f"last entries of the {len(toks)}-token vocabulary — have identical {head_name} rows (cos {cos:.5f}, "
                f"max abs diff {maxd:.5f}, norm {na:.4f}): the rows were never trained, so the model cannot emit its "
                "own delimiters")
        if not scan or len(scan) > _GLITCH_MAX:
            # No deterministic stand-in (or an implausibly large one): nothing
            # safe to filter — the defect stands as a misconfiguration without
            # a derivable repair.
            rep.add(Finding("gguf_dead_delimiters", MISCONFIG,
                            base + f"; glitch scan found {len(scan)} stand-in token(s) — no derivable output repair",
                            file=rel))
            return
        glitch_ids = [k for k, _, _ in scan]
        glitch_texts = [_token_text(toks[k], tok_model) for k in glitch_ids]
        block = {"kind": OUTPUT_REPAIR_KIND, "file": rel,
                 "open": s, "close": close, "open_id": i, "close_id": j,
                 "glitch_token_ids": glitch_ids, "glitch_tokens": glitch_texts,
                 "measured": measured,
                 "criterion": f"NORMAL tokens with {head_name} row cos>={_GLITCH_COS} to the {s} row and projection "
                              f"on it >= its norm by >= {_GLITCH_MARGIN:.1%}",
                 "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rep.output_repair = block
        stand_in = ", ".join(f"{t!r} (id {k}, cos {c:.5f}, +{m:.5f})"
                             for t, (k, c, m) in zip(glitch_texts, scan))
        prior = _marker_output_repair(rep.destination)
        if prior and all(prior.get(f) == block[f] for f in
                         ("kind", "file", "open_id", "close_id", "glitch_token_ids")):
            rep.output_repair_applied = True
            rep.add(Finding("gguf_dead_delimiters", "info",
                            f"output delimiter repair applied (hugpy.json output_repair, recorded {prior.get('at')}): "
                            f"{s}/{close} rows collapsed (cos {cos:.5f}, max abs diff {maxd:.5f}, norm {na:.4f}); "
                            f"{stand_in} served as {s}/{close}", file=rel))
            return
        rep.add(Finding("gguf_dead_delimiters", MISCONFIG,
                        base + f" — it emits {stand_in} in their place (NORMAL tokens whose rows outscore the dead "
                        f"delimiter row along its own direction)", file=rel, fix=OUTPUT_REPAIR_FIX))
        return


def _marker_output_repair(dest: Optional[str]) -> Optional[dict]:
    if not dest:
        return None
    try:
        from hugpy_storage.hugpy_marker import read_output_repair
    except ImportError:
        return _lenient_json_block(dest)
    return read_output_repair(dest)


def _lenient_json_block(dest: str) -> Optional[dict]:
    blk = (_lenient_json(os.path.join(dest, "hugpy.json")) or {}).get("output_repair")
    return blk if isinstance(blk, dict) else None


def _check_transformers(rep: ModelReport, dest: str, files: dict, size_bad: set, fw: str) -> None:
    log = rep.log
    for cfg_name in ("config.json", "model_index.json", "adapter_config.json"):
        if cfg_name in files:
            try:
                with open(os.path.join(dest, cfg_name), "r", encoding="utf-8") as fh:
                    json.load(fh)
                log.append(f"{cfg_name}: parses")
            except (OSError, ValueError) as exc:
                loose = _lenient_json(os.path.join(dest, cfg_name))
                if loose is not None and not (loose.get("architectures") or loose.get("model_type")
                                              or loose.get("_class_name")):
                    rep.add(Finding("config", "info", f"{cfg_name}: not strict JSON ({exc}) and not a transformers "
                                    f"config (keys={sorted(loose)[:6]}) — a repo descriptor; transformers cannot load "
                                    "this repo", file=cfg_name))
                    log.append(f"{cfg_name}: not strict JSON ({exc}); lenient parse keys={sorted(loose)}")
                    continue
                v = BROKEN if cfg_name in size_bad else FAULTY
                rep.add(Finding("config", v, f"{cfg_name}: unparsable ({exc})", file=cfg_name))
                log.append(f"{cfg_name}: unparsable: {exc}")
    for rel in sorted(files):
        low = rel.lower()
        path = os.path.join(dest, rel)
        if low.endswith((".safetensors", ".sft")):
            r = safetensors_check(path)
            log.append(f"safetensors {rel}: {r['kind']}: {r['detail']}")
            if r["kind"] == "ok" and rel in size_bad:
                _size_drift(rep, [rel], size_bad, "header ranges end exactly at EOF")
            if r["kind"] in ("truncated", "oversize", "zeroed"):
                rep.add(Finding("safetensors", BROKEN, f"{rel}: {r['detail']}", file=rel, fix="re-provision through hugpy"))
            elif r["kind"] == "bad_header" and rel not in size_bad:
                rep.add(Finding("safetensors", FAULTY, f"{rel}: {r['detail']}", file=rel))
        elif low.endswith((".bin", ".pt", ".pth", ".ckpt")):
            try:
                r = torch_zip_check(path)
            except OSError as exc:
                r = {"ok": False, "detail": f"unreadable: {exc}"}
            if r is None:
                continue
            log.append(f"torch {rel}: {r['detail']}")
            if r["ok"] and rel in size_bad:
                _size_drift(rep, [rel], size_bad, "zip central directory present")
            if not r["ok"]:
                rep.add(Finding("torch_zip", BROKEN, f"{rel}: {r['detail']}", file=rel, fix="re-provision through hugpy"))


def _worker_forms(w: dict) -> list:
    return [w.get("name"), w.get("id")]


def _check_gguf_config(rep: ModelReport, row: dict, ctx: Context, files: dict, facts: dict) -> None:
    key = rep.model_key
    ov = ctx.overrides.get(key) or {}
    log = rep.log
    if ov:
        log.append(f"override {ctx.overrides_path}[{key}] = {json.dumps(ov, sort_keys=True)}")
    else:
        log.append(f"override: none for key {key!r}")
    lang = {r: f for r, f in facts.items() if not is_projector(r, f.get("arch"))}
    projectors = sorted(r for r in facts if r not in lang)
    if facts and not lang:
        rep.add(Finding("effective_pick", MISCONFIG,
                        f"every GGUF on central is a vision projector ({', '.join(projectors)}); no language-model quant is on central",
                        fix="set hugpy.json filename / the model's quant to a language-model GGUF and re-provision through hugpy"))
    # effective pick + per-worker pins
    eff = row.get("effective_gguf")
    picks = {}
    if eff:
        loc = _find_local(eff, files)
        picks["(catalog effective_gguf)"] = (eff, loc, "catalog effective_gguf")
    for w in ctx.workers:
        pin, src = pin_for_worker(ov, _worker_forms(w))
        if pin:
            picks[w["name"]] = (pin, _find_local(pin, files), f"override {src}")
    for who, (name, loc, src) in picks.items():
        f = facts.get(loc) if loc else None
        arch = (f or {}).get("arch")
        if loc and is_projector(loc, arch):
            where = "catalog" if who.startswith("(") else f"worker {who}"
            rep.add(Finding("effective_pick", MISCONFIG,
                            f"{src} for {where} = {name!r} is a vision projector ({loc}, arch={arch}) — llama-server "
                            "refuses it: unsupported model architecture: 'clip'",
                            worker=None if who.startswith("(") else who,
                            fix=(f"serve_overrides.json[{key!r}].{src.split('override ')[-1].split('[')[0]}: point it at a "
                                 "language-model quant (not mmproj/*) or delete the pin" if src.startswith("override")
                                 else "the catalog's effective quant is a projector; pin a language-model quant via serve_overrides gguf_file")))
            log.append(f"pick {who}: {src}={name!r} -> {loc} arch={arch} (projector)")
        elif loc:
            log.append(f"pick {who}: {src}={name!r} -> {loc} arch={arch}")
        elif src.startswith("override"):
            log.append(f"pick {who}: {src}={name!r} not present on central (worker-local only?)")
    # MoE without expert split, too big for any GPU
    eff_loc = _find_local(eff, files) if eff else None
    if eff_loc and eff_loc in lang:
        ec = lang[eff_loc].get("expert_count") or 0
        m = _SHARD_RE.match(os.path.basename(eff_loc))
        eff_bytes = (sum(sz for r, sz in files.items() if logical_gguf(r) == logical_gguf(eff_loc)
                         and os.path.dirname(r) == os.path.dirname(eff_loc)) if m else files[eff_loc])
        moe_eff = any((w.get("moe_effective") or {}).get(k) for w in ctx.workers for k in _report_keys(key, ctx))
        max_mem = max([int(w.get("ram_bar_total") or 0) + ctx.gpu_totals.get(w["name"], 0) for w in ctx.online] or [0])
        if ec and ec > 0 and "n_cpu_moe" not in ov and not moe_eff and eff_bytes > ctx.max_gpu:
            too_big = max_mem and eff_bytes > max_mem
            rep.add(Finding("moe_split", MISCONFIG,
                            f"MoE GGUF (expert_count={ec}) of {_gb(eff_bytes)} with no n_cpu_moe override and no worker "
                            f"applying the expert split; largest online GPU is {_gb(ctx.max_gpu)}"
                            + (f"; it also exceeds every online worker's RAM+VRAM (largest {_gb(max_mem)})" if too_big else ""),
                            fix=(f"unservable on this fleet (needs > {_gb(max_mem)} RAM+VRAM): drop it or place it on a bigger node"
                                 if too_big else
                                 # DENSE BACKBONE FIRST (operator ruling 2026-09-25:
                                 # nothing should suggest evacuating GPU resources
                                 # for no need). A MoE's card holds the dense
                                 # backbone + as many EXPERT layers as fit — NOT
                                 # the whole file — so a static n_cpu_moe:999 (ALL
                                 # experts to CPU) idles the card whenever the
                                 # backbone fits it. Recommend a STATED gpu_mem_gib
                                 # contract sized to the card instead: it flips the
                                 # worker into moe_dense_first_plan, which
                                 # auto-sizes n_cpu_moe to fill VRAM (backbone +
                                 # the experts that fit, rest spilled). 999 is
                                 # reserved for the genuinely-can't-hold-experts
                                 # case — the backbone alone exceeding the card —
                                 # which lands in the too_big / dense branches, not
                                 # here.
                                 f"serve_overrides.json[{key!r}] += {{\"gpu_mem_gib\": "
                                 f"{max(1.0, ctx.max_gpu / (2 ** 30) * 0.85):.1f}, "
                                 f"\"n_gpu_layers\": -1}} (a card-sized budget on "
                                 f"the {_gb(ctx.max_gpu)} GPU; the worker's "
                                 f"moe_dense_first_plan auto-sizes n_cpu_moe to "
                                 f"fill it — do NOT hardcode n_cpu_moe:999, which "
                                 f"strands every expert in RAM)")))
        reserve = 2 ** 29
        if (not ec) and ov.get("n_gpu_layers") == -1 and ctx.online and all(
                eff_bytes + reserve > ctx.gpu_totals[w["name"]] for w in ctx.online):
            rep.add(Finding("fit", MISCONFIG,
                            f"dense GGUF {eff_loc} is {_gb(eff_bytes)} (+0.5 GiB reserve) at n_gpu_layers=-1; exceeds every "
                            f"online GPU (largest {_gb(ctx.max_gpu)})",
                            fix=f"serve_overrides.json[{key!r}]: drop n_gpu_layers=-1 (let autofit spill) or pin a smaller quant"))
    # per-worker: planned gpu-only but the served bytes exceed the card
    for w in ctx.workers:
        name = w["name"]
        total = ctx.gpu_totals.get(name) or 0
        if not total:
            continue
        for k in _report_keys(key, ctx):
            plan = (w.get("planned_split") or {}).get(k) or {}
            ld = (w.get("loaded_detail") or {}).get(k) or {}
            served = max(int(ld.get("weight_bytes") or 0), int(plan.get("size_bytes") or 0))
            if plan.get("mode") == "gpu-only" and "n_cpu_moe" not in ov and served > total:
                rep.add(Finding("fit", MISCONFIG,
                                f"worker {name}: plan is gpu-only (n_gpu_layers=-1) but the served file is {_gb(served)} "
                                f"on a {_gb(total)} GPU (planned_split={json.dumps(plan, sort_keys=True)}, "
                                f"loaded weight_bytes={ld.get('weight_bytes')})", worker=name,
                                fix=f"pin a quant that fits {name} via serve_overrides.json[{key!r}].gguf_file_by_worker"
                                    f"[{w.get('id')}], or set n_gpu_layers to a partial count / alloc_mode max-gpu"))
                break


def _check_override_alias(rep: ModelReport, key: str, row: dict, ctx: Context) -> None:
    if "~" in key or key in ctx.overrides:
        return
    owner = (row.get("hub_id") or "").split("/")[0]
    for ok in ctx.overrides:
        if "~" in ok and ok.split("~", 1)[1] == key and ok not in ctx.catalog_keys:
            same = ok.split("~", 1)[0] == owner
            rep.add(Finding("override_key", MISCONFIG if same else "info",
                            f"override stored only under alias key {ok!r}; the called key {key!r} has none "
                            "(overrides match the exact key)",
                            fix=f"move serve_overrides.json[{ok!r}] to [{key!r}]"))
            rep.log.append(f"override alias: {ok!r} = {json.dumps(ctx.overrides[ok], sort_keys=True)}; bare key has none")


def _worker_evidence(rep: ModelReport, row: dict, ctx: Context, sizes_ok: bool) -> ModelReport:
    key = rep.model_key
    keys = _report_keys(key, ctx)
    cat_workers = {w.get("worker"): w for w in row.get("workers") or []}
    hot = set(row.get("hot_workers") or [])
    faulty_sigs: dict = {}
    for w in ctx.workers:
        name = w["name"]
        lr = None
        for k in keys:
            lr = (w.get("load_reports") or {}).get(k)
            if lr:
                break
        cw = cat_workers.get(name) or {}
        local = any(k in (w.get("models_local") or []) for k in keys)
        loaded = any(k in (w.get("loaded_models") or []) for k in keys)
        if not (lr or cw or local or loaded):
            continue
        ev = {"status": w.get("status"), "gpu_total_bytes_known": w.get("gpu_total_bytes_known"),
              "local": local, "loaded": loaded, "hot": name in hot,
              "on_disk_bytes": cw.get("on_disk_bytes"), "alloc": cw.get("alloc"), "alloc_mode": cw.get("alloc_mode"),
              "planned_split": next(((w.get("planned_split") or {}).get(k) for k in keys if (w.get("planned_split") or {}).get(k)), None),
              "loaded_detail": next(((w.get("loaded_detail") or {}).get(k) for k in keys if (w.get("loaded_detail") or {}).get(k)), None),
              "moe_effective": any((w.get("moe_effective") or {}).get(k) for k in keys)}
        if lr:
            err = lr.get("error")
            head, stderr = split_loader_stderr(err or "")
            # The structured load_failure (whole loader stderr + log_ref)
            # outranks the stderr parsed out of the error wording.
            _lf = lr.get("load_failure") if isinstance(lr.get("load_failure"), dict) else {}
            if _lf.get("loader_stderr"):
                stderr = _lf.get("loader_stderr")
            log_ref = _lf.get("log_ref")
            ev["load_report"] = {k: v for k, v in lr.items() if k != "error"}
            ev["load_error"] = head or None
            ev["loader_stderr"] = stderr
            ev["log_ref"] = log_ref
            cls = classify_load_error(err)
            ev["load_error_class"] = cls
            ts = lr.get("ts")
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "?"
            if err:
                rep.log.append(f"worker {name} load_reports[{key}] ({stamp}) ok={lr.get('ok')} local={lr.get('local')} "
                               f"fit={lr.get('fit')} error: {head}")
                if stderr:
                    rep.log.append(f"worker {name} loader_stderr"
                                   f"{' (log_ref=' + str(log_ref) + ')' if log_ref else ''}:\n{stderr}")
                else:
                    rep.log.append(f"worker {name} loader_stderr: empty (read load_reports[{key}].error "
                                   f"+ .load_failure.loader_stderr, bytes=0"
                                   f"{', log_ref=' + str(log_ref) if log_ref else ''})")
            else:
                rep.log.append(f"worker {name} load_reports[{key}] ({stamp}) ok={lr.get('ok')} path={lr.get('path')}")
            if cls:
                c = cls["class"]
                if c == FAULTY:
                    faulty_sigs[name] = cls["reason"]
                elif c == MISCONFIG:
                    rep.add(Finding("load_report", MISCONFIG, f"worker {name}: {cls['reason']}", worker=name,
                                    fix="point the pin at a language-model quant (not mmproj)"))
                elif c == "fit":
                    total = int(w.get("gpu_total_bytes_known") or 0)
                    need = cls.get("need_bytes") or 0
                    if total and need > total:
                        rep.add(Finding("load_report", MISCONFIG, f"worker {name}: needs {_gb(need)} VRAM, card total "
                                        f"{_gb(total)} — can never fit as configured", worker=name,
                                        fix="pin a smaller quant for this worker or allow a partial/MoE split"))
                    else:
                        rep.add(Finding("load_report", "info", f"worker {name}: VRAM contention (needs {_gb(need)} of "
                                        f"{_gb(total)} total; other residents held the rest) — not a model fault", worker=name))
                elif c in ("info", "transient", "load_failed"):
                    if "not local" not in cls["reason"]:
                        rep.add(Finding("load_report", "info", f"worker {name}: {cls['reason']}", worker=name))
        rep.evidence["workers"][name] = ev
    if faulty_sigs:
        same = len(set(faulty_sigs.values())) == 1 and len(faulty_sigs) >= 2
        for name, reason in faulty_sigs.items():
            if sizes_ok:
                rep.add(Finding("load_report", FAULTY, f"worker {name}: {reason}"
                                + (f" (identical on {len(faulty_sigs)} workers)" if same else ""), worker=name))
            else:
                rep.add(Finding("load_report", BROKEN, f"worker {name}: {reason} (size check failed)", worker=name))
    return rep


# ── verdict ──────────────────────────────────────────────────────────────────

def _ever_loaded(rep: ModelReport, ctx: Context) -> list:
    out = []
    keys = _report_keys(rep.model_key, ctx)
    for w in ctx.workers:
        e = rep.evidence["workers"].get(w["name"]) or {}
        lr = e.get("load_report") or {}
        if (lr.get("ok") and lr.get("path") not in (None, "none")) or e.get("loaded") or any(
                k in (w.get("model_tok_stats") or {}) for k in keys):
            out.append(w["name"])
    return out


def _expected_suite_name(row: dict) -> Optional[str]:
    """The suite hugpy_curation would grade this row with NOW, by its CURRENT
    task (``suites.suite_for_model``), or None when unknown / hugpy_curation is
    not importable here. Central's live suite assignment is the authoritative
    fact: a 0 recorded by a suite that is no longer this model's suite (e.g. a
    text-suite grade taken before the task was corrected to text-to-image) is
    STALE and must not keep the model flagged suite_mismatch."""
    try:
        from hugpy_curation.review.suites import suite_for_model
    except Exception:  # noqa: BLE001 — curation not importable here: unknown
        return None
    try:
        chosen = suite_for_model(row)
    except Exception:  # noqa: BLE001 — unresolvable: unknown
        return None
    return chosen.name if chosen is not None else None


def finalize(rep: ModelReport, row: dict, ctx: Context) -> ModelReport:
    order = (NOT_DOWNLOADED, BROKEN, FAULTY, MISCONFIG, MISLABELED)
    verdict = None
    for v in order:
        if any(f.verdict == v for f in rep.findings):
            verdict = v
            break
    grades = ctx.grades.get(rep.model_key) or []
    if grades:
        best = max(grades, key=lambda r: r.get("grade") or 0)
        rep.recorded_grade = {"grade": best.get("grade"), "suite": best.get("grade_suite"),
                              "worker": best.get("worker"), "quant": best.get("quant"),
                              "n_rows": len(grades)}
    non_text = (rep.framework == "comfy" or (rep.primary_task and rep.primary_task not in TEXT_TASKS)
                or (rep.detected_task and rep.detected_task not in TEXT_TASKS
                    and rep.detected_task != "projector-only"))
    text_zero = [r for r in grades if (r.get("grade") or 0) == 0 and r.get("grade_suite") != SUITE]
    # RECOMPUTE against the CURRENT suite assignment: drop 0-grades recorded by a
    # suite that is no longer this model's suite (a stale mis-grade from before
    # the task was corrected — comfy-dreamshaper-8 now runs hugpy-imagegen-v1, not
    # the text suite that scored it 0). Only a 0 from the suite central would
    # STILL grade it with is a live mismatch; when the expected suite is unknown
    # here, behaviour is unchanged.
    expected_suite = _expected_suite_name(row)
    if expected_suite is not None:
        text_zero = [r for r in text_zero if r.get("grade_suite") == expected_suite]
    if non_text and text_zero and all((r.get("grade") or 0) == 0 for r in grades if r.get("grade_suite") != SUITE):
        rep.tags.append(SUITE_MISMATCH)
        rep.add(Finding("suite", SUITE_MISMATCH, f"{rep.primary_task or rep.detected_task} model graded 0 by text suite "
                        f"{text_zero[0].get('grade_suite')} — the grade measures the suite, not the model",
                        fix="exclude non-text tasks from the text grader"))
    if verdict is None:
        if SUITE_MISMATCH in rep.tags:
            verdict = SUITE_MISMATCH
        elif rep.framework == "comfy":
            verdict = UNSUPPORTED
        else:
            verdict = WORKING if rep.smoke.get("ok") else STATIC_OK
    rep.verdict = verdict
    # eliminate ONLY a faulty model (complete + structurally wrong). A broken
    # download is re-provisioned, never eliminated — whatever upstream says.
    rep.eliminate = verdict == FAULTY
    lead = [f for f in rep.findings if f.verdict == verdict]
    if verdict in (STATIC_OK, WORKING, UNSUPPORTED):
        loaded = _ever_loaded(rep, ctx)
        infos = [f.detail for f in rep.findings if f.verdict == "info" and f.worker]
        for name, ev in rep.evidence["workers"].items():
            cls = ev.get("load_error_class") or {}
            if cls.get("class") == "info" and "not local" in cls.get("reason", ""):
                infos.append(f"worker {name}: last load probe found it not local (the probe does not download)")
        infos += [f.detail for f in rep.findings if f.verdict == "info" and not f.worker]
        base = "static checks passed" if verdict != UNSUPPORTED else "comfy checkpoint: presence + header checks passed"
        if verdict == WORKING:
            rep.why = f"{base}; smoke answered on {rep.smoke.get('worker')} in {rep.smoke.get('seconds')}s"
        elif loaded:
            rep.why = f"{base}; loaded ok on {', '.join(loaded)}" + (f"; {'; '.join(infos[:2])}" if infos else "")
        else:
            rep.why = f"{base}; never loaded on any worker" + (f"; {'; '.join(infos[:2])}" if infos else "")
            rep.log.append("static checks passed; never loaded on any worker")
        if rep.recorded_grade and (rep.recorded_grade.get("grade") or 0) == 0 and verdict != WORKING:
            rep.why += (f"; recorded grade 0 ({rep.recorded_grade.get('suite')}) is not explained by the files — "
                        "the grader got no answer (see worker evidence)")
    else:
        rep.why = lead[0].detail if lead else verdict
    fixes = [f.fix for f in lead if f.fix]
    rep.suggested_fix = "; ".join(dict.fromkeys(fixes)) or None
    if verdict == FAULTY:
        rep.suggested_fix = ("eliminate: the file is complete and matches its source, yet internally inconsistent — "
                             "re-downloading reproduces it; remove the model from the catalog")
    return rep



# ── smoke + record ───────────────────────────────────────────────────────────

def api_key_candidates() -> list:
    """Operator keys in lookup order: :func:`api_key`, then ``HUGPY_API_KEY=``
    lines of the operator's env files (only one ``hp_`` key authorizes /v1)."""
    out = []
    k = api_key()
    if k:
        out.append(k)
    home = os.path.expanduser("~")
    for p in (os.path.join(home, ".env"), os.path.join(home, ".config", "hugpy-station", "env", "station.env"),
              os.path.join(home, ".config", "hugpy-agent", "agent.env")):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith(("HUGPY_API_KEY=", "export HUGPY_API_KEY=")):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if v and v not in out:
                            out.append(v)
        except OSError:
            continue
    return out


def _authed(method: str, url: str, payload: dict, keys: list, timeout: float):
    """POST with the first key that is not refused (401/403)."""
    last = (None, None)
    for k in keys or [None]:
        try:
            st, body = _request(url, method=method, payload=payload, token=k, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return "error", f"{type(exc).__name__}: {exc}"
        if st in (401, 403):
            last = (st, body)
            continue
        return st, body
    return last


def smoke(rep: ModelReport, ctx: Context, keys: list) -> None:
    hot = [n for n, e in rep.evidence["workers"].items() if e.get("loaded") and e.get("hot")]
    if not hot:
        rep.smoke = {"skipped": "not hot on any worker (a smoke never cold-loads)"}
        rep.log.append("smoke: skipped — not hot on any worker")
        return
    for name in hot:
        payload = {"model": rep.model_key, "max_tokens": 4, "temperature": 0,
                   "messages": [{"role": "user", "content": "Reply with: OK"}], "alloc": {"worker": name}}
        t0 = time.time()
        st, body = _authed("POST", f"{ctx.central}/v1/chat/completions", payload, keys, SMOKE_TIMEOUT)
        dt = round(time.time() - t0, 2)
        text = None
        if st == 200 and isinstance(body, dict) and body.get("choices"):
            text = ((body["choices"][0].get("message") or {}).get("content") or "")
            rep.smoke = {"ok": True, "worker": name, "seconds": dt, "http": st, "reply": text}
            rep.log.append(f"smoke {name}: HTTP 200 in {dt}s, reply={text!r}")
            return
        detail = json.dumps(body) if not isinstance(body, str) else body
        transient = st in (429, 502, 503, 504, "error")
        rep.smoke = {"ok": False, "worker": name, "seconds": dt, "http": st, "transient": transient, "body": detail}
        rep.log.append(f"smoke {name}: HTTP {st} in {dt}s: {detail}")
        if not transient:
            cls = classify_load_error(detail)
            if cls and cls["class"] in (FAULTY, MISCONFIG):
                rep.add(Finding("smoke", cls["class"], f"smoke on {name}: {cls['reason']}", worker=name))
            else:
                rep.add(Finding("smoke", "info", f"smoke on {name} failed: HTTP {st}", worker=name))


def apply_output_repair(rep: ModelReport) -> Optional[str]:
    """Write the derived ``output_repair`` block to the model's hugpy.json
    (the fix for a ``misconfigured`` dead-delimiter finding). Returns the path
    written, or None when there is nothing to apply / no marker."""
    if not rep.output_repair or rep.output_repair_applied or not rep.destination:
        return None
    try:
        from hugpy_storage.hugpy_marker import write_output_repair
    except ImportError:
        return None
    path = write_output_repair(rep.destination, rep.output_repair)
    if path:
        rep.output_repair_applied = True
        rep.log.append(f"output repair applied: hugpy.json output_repair written ({path}): "
                       f"kind={rep.output_repair.get('kind')} glitch_token_ids={rep.output_repair.get('glitch_token_ids')}")
    return path


def _mark_repair_applied(rep: ModelReport, ctx: Context) -> None:
    """The fix just landed: the dead-delimiter finding becomes ``info`` and the
    verdict is recomputed, so the caller (admission) proceeds on the repaired
    model in the SAME run. The why keeps the history: misconfigured -> fixed."""
    was = rep.verdict
    changed = False
    for f in rep.findings:
        if f.check == "gguf_dead_delimiters" and f.verdict == MISCONFIG and f.fix == OUTPUT_REPAIR_FIX:
            f.verdict = "info"
            f.detail = f"output delimiter repair applied now (hugpy.json output_repair): {f.detail}"
            f.fix = None
            changed = True
    if not changed or SUITE_MISMATCH in rep.tags:
        return
    finalize(rep, {}, ctx)
    if rep.verdict != was:
        rep.why = (f"{rep.why}; was {was} ({OUTPUT_REPAIR_FIX}) — fix applied: hugpy.json output_repair written "
                   f"(glitch tokens {rep.output_repair.get('glitch_tokens')} served as "
                   f"{rep.output_repair.get('open')}/{rep.output_repair.get('close')})")
        rep.log.append(f"verdict {was} -> {rep.verdict} after {OUTPUT_REPAIR_FIX}")


def record(rep: ModelReport, ctx: Context, keys: list) -> Optional[dict]:
    # The fix for a dead-delimiter misconfiguration is data this audit derived:
    # apply it with the record (``--record`` / the admission runner), so the
    # next audit of the same file reads "repair applied" and serving is clean.
    try:
        if apply_output_repair(rep):
            _mark_repair_applied(rep, ctx)
    except Exception as exc:  # noqa: BLE001 — the grade row stands regardless
        rep.log.append(f"output repair NOT applied: {type(exc).__name__}: {exc}")
    if rep.verdict in (STATIC_OK, WORKING):
        grade = 100.0
    elif rep.verdict in (BROKEN, FAULTY, MISCONFIG):
        grade = 0.0
    else:
        return None
    detail = json.dumps({"verdict": rep.verdict, "eliminate": rep.eliminate, "why": rep.why, "log": rep.log,
                         "suggested_fix": rep.suggested_fix,
                         **({"output_repair": rep.output_repair, "output_repair_applied": rep.output_repair_applied}
                            if rep.output_repair else {})})
    worker = next(iter(rep.evidence["workers"]), "")
    payload = {"model": rep.model_key, "grade": grade, "suite": SUITE, "detail": detail,
               "quant": "", "worker": worker}
    st, body = _authed("POST", f"{ctx.central}/llm/model-grade", payload, keys, CENTRAL_TIMEOUT)
    return {"http": st, "ok": st == 200, "grade": grade, **({"error": body} if st != 200 else {})}


# ── output ───────────────────────────────────────────────────────────────────

def as_dict(rep: ModelReport) -> dict:
    d = asdict(rep)
    d["findings"] = [asdict(f) if not isinstance(f, dict) else f for f in rep.findings]
    return d


def table(reports: list) -> str:
    rows = [("MODEL", "FRAMEWORK", "VERDICT", "ELIM", "FIRST FINDING / WHY")]
    for r in reports:
        rows.append((r.model_key[:60], r.framework, r.verdict, "yes" if r.eliminate else "", r.why[:110]))
    w = [max(len(x[i]) for x in rows) for i in range(4)]
    out = [f"{a:<{w[0]}}  {b:<{w[1]}}  {c:<{w[2]}}  {d:<{w[3]}}  {e}" for a, b, c, d, e in rows]
    return "\n".join(out)


def counts(reports: list) -> dict:
    out: dict = {}
    for r in reports:
        out.setdefault(r.framework, {}).setdefault(r.verdict, 0)
        out[r.framework][r.verdict] += 1
    return out


def markdown(reports: list, meta: dict) -> str:
    L = [f"# hugpy model integrity audit — {time.strftime('%Y-%m-%d %H:%M')}", "",
         f"central {meta['central']}; {len(reports)} models; expected values: install manifest "
         f"({meta.get('with_manifest', 0)} with, {meta.get('without_manifest', 0)} without); "
         f"hash={'on' if meta.get('hash') else 'off'}; "
         f"smoke={'on' if meta['smoke'] else 'off'}; record={'on' if meta['record'] else 'off'}", "",
         "## Verdict counts", "", "| framework | " + " | ".join(VERDICTS) + " |",
         "|---|" + "---|" * len(VERDICTS)]
    for fw, c in sorted(counts(reports).items()):
        L.append(f"| {fw} | " + " | ".join(str(c.get(v, 0)) for v in VERDICTS) + " |")
    L += ["", "## Eliminate", ""]
    L += [f"- `{r.model_key}` ({r.verdict}): {r.why}" for r in reports if r.eliminate] or ["- none"]
    for v, title in ((BROKEN, "Broken downloads"), (MISCONFIG, "Misconfigured"), (MISLABELED, "Mislabeled task"),
                     (NOT_DOWNLOADED, "Not downloaded")):
        sel = [r for r in reports if r.verdict == v]
        L += ["", f"## {title} ({len(sel)})", ""]
        for r in sel:
            L.append(f"- `{r.model_key}`: {r.why}" + (f"  \n  fix: {r.suggested_fix}" if r.suggested_fix else "")
                     + (f"  \n  detected task: {r.detected_task}" if v == MISLABELED else ""))
        if not sel:
            L.append("- none")
    L += ["", "## All models", "", "| model | framework | verdict | elim | why |", "|---|---|---|---|---|"]
    for r in reports:
        L.append(f"| `{r.model_key}` | {r.framework} | {r.verdict} | {'yes' if r.eliminate else ''} | "
                 f"{r.why.replace('|', '/')} |")
    return "\n".join(L) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────────

def run(*, central: Optional[str] = None, only=None, framework: Optional[str] = None,
        do_smoke: bool = False, do_record: bool = False, jobs: int = 8,
        hash_check: bool = False, hash_max_bytes: int = HASH_MAX_BYTES) -> tuple:
    base = central_url(central)
    catalog = fetch_json(f"{base}/api/models?verbose=1")
    if isinstance(catalog, dict):
        catalog = catalog.get("models") or catalog.get("rows") or []
    wk = fetch_json(f"{base}/llm/workers")
    if isinstance(wk, dict):
        wk = wk.get("workers") or []
    overrides, opath = load_overrides()
    ctx = Context(base, wk, overrides, opath, grade_rows(base), {r["model_key"] for r in catalog},
                  hash_check, hash_max_bytes)
    rows = catalog
    if only:
        want = set(only)
        rows = [r for r in rows if r["model_key"] in want]
        missing = want - {r["model_key"] for r in rows}
        if missing:
            raise KeyError(f"not in catalog: {sorted(missing)}")
    if framework:
        fws = ("gguf", "llama_cpp") if framework == "gguf" else (framework,)
        rows = [r for r in rows if r.get("framework") in fws]

    def one(row):
        try:
            rep = audit_model(row, ctx)
        except Exception as exc:  # noqa: BLE001 — one model never sinks the sweep
            rep = ModelReport(row["model_key"], row.get("framework") or "?", row.get("hub_id"),
                              row.get("primary_task"), row.get("destination"))
            rep.add(Finding("tool", "info", f"audit error: {type(exc).__name__}: {exc}"))
            rep.log.append(f"tool error: {type(exc).__name__}: {exc}")
        return rep

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        reports = list(ex.map(one, rows))
    keys = api_key_candidates() if (do_smoke or do_record) else []
    for rep, row in zip(reports, rows):
        if do_smoke and not any(f.verdict in _PRECEDENCE + (MISLABELED,) for f in rep.findings) \
                and rep.framework in ("gguf", "llama_cpp", "transformers"):
            smoke(rep, ctx, keys)
        finalize(rep, row, ctx)
        if do_record:
            rep.recorded_grade = {**(rep.recorded_grade or {}), "integrity_write": record(rep, ctx, keys)}
    reports.sort(key=lambda r: (VERDICTS.index(r.verdict) if r.verdict in VERDICTS else 99, r.framework, r.model_key))
    with_manifest = sum(1 for r in reports if "manifest" in r.evidence)
    meta = {"central": base, "expected_values": "install manifest (hugpy.json)",
            "with_manifest": with_manifest, "without_manifest": len(reports) - with_manifest,
            "hash": hash_check, "hash_max_bytes": hash_max_bytes, "smoke": do_smoke, "record": do_record,
            "generated_at": time.time(), "n_models": len(reports),
            "workers": {w["name"]: {"status": w.get("status"), "gpu_total_bytes_known": w.get("gpu_total_bytes_known")}
                        for w in wk}, "overrides_path": opath}
    return reports, meta




def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hugpy-model-audit",
        description="Per model central holds: working / broken download / faulty model / misconfigured, "
                    "each with its verbatim evidence log. Read-only unless --smoke/--record.")
    p.add_argument("--central", help=f"central base URL (default: HUGPY_BASE_URL or {DEFAULT_CENTRAL})")
    p.add_argument("--only", nargs="+", metavar="KEY", help="audit only these model keys")
    p.add_argument("--framework", choices=("gguf", "transformers", "comfy"), help="audit one framework")
    p.add_argument("--hub", action="store_true", help=argparse.SUPPRESS)   # removed: hard error below
    p.add_argument("--hash", action="store_true",
                   help="verify sha256 of install-manifest files that carry one and are under --hash-max-bytes "
                        "(off by default: only a full hash proves a large file bit-exact, and it is slow)")
    p.add_argument("--hash-max-bytes", type=int, default=HASH_MAX_BYTES, metavar="N",
                   help=f"size cap for --hash (default {HASH_MAX_BYTES})")
    p.add_argument("--smoke", action="store_true", help="4-token chat on models that pass static checks AND are hot")
    p.add_argument("--record", action="store_true", help="write the verdict as grade (suite 'integrity') + log as detail")
    p.add_argument("--json", metavar="PATH", help="JSON report path (default: $PROJECTS_HOME/model_audit.json)")
    p.add_argument("--md", metavar="PATH", help="markdown summary path (default: $PROJECTS_HOME/model_audit.md)")
    p.add_argument("--workers", action="store_true", help="also print per-worker load evidence in the table")
    p.add_argument("--jobs", type=int, default=8, help="parallel model checks (default 8)")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.hub:
        print("hugpy-model-audit: --hub is not supported: expected values are captured at install "
              "(hugpy.json manifest); the audit never calls Hugging Face. "
              "Models without a manifest: hugpy-model-manifest-backfill.", file=sys.stderr)
        return 2
    try:
        reports, meta = run(central=args.central, only=args.only, framework=args.framework,
                            do_smoke=args.smoke, do_record=args.record, jobs=args.jobs,
                            hash_check=args.hash, hash_max_bytes=args.hash_max_bytes)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"hugpy-model-audit: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(table(reports))
    if args.workers:
        for r in reports:
            for name, ev in r.evidence["workers"].items():
                c = (ev.get("load_error_class") or {}).get("class", "")
                print(f"  {r.model_key} @ {name}: local={ev['local']} loaded={ev['loaded']} {c} "
                      f"{(ev.get('load_error') or '')[:120]}")
    c = counts(reports)
    print("\n" + "; ".join(f"{fw}: " + ", ".join(f"{v}={n}" for v, n in sorted(d.items())) for fw, d in sorted(c.items())))
    ph = projects_home()
    jpath = args.json or os.path.join(ph, "model_audit.json")
    mpath = args.md or os.path.join(ph, "model_audit.md")
    doc = {**meta, "counts": c, "eliminate": [r.model_key for r in reports if r.eliminate],
           "models": {r.model_key: as_dict(r) for r in reports}}
    try:
        with open(jpath + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, default=str)
        os.replace(jpath + ".tmp", jpath)
        with open(mpath + ".tmp", "w", encoding="utf-8") as fh:
            fh.write(markdown(reports, meta))
        os.replace(mpath + ".tmp", mpath)
        print(f"report: {jpath}\nsummary: {mpath}")
    except OSError as exc:
        print(f"hugpy-model-audit: cannot write report: {exc}", file=sys.stderr)
        return 2
    return 1 if any(r.eliminate for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Host-wide, read-only model discovery for a Hugpy worker.

External weights keep their ownership and location.  Hugpy writes a small
``hugpy.json`` pointer in its own worker inventory, never into a foreign model
directory.  The inventory is metadata; it is not part of the disposable cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

_SKIP = {"proc", "sys", "dev", "run", "tmp", "lost+found", ".git",
         "node_modules", "__pycache__", ".venv", "venv", "site-packages",
         "var/log", "var/cache", "_archive", "dedupes", "incoming", "legacy",
         "llm_backups"}


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _roots() -> list[str]:
    """Return explicitly configured external-model roots for this worker.

    Hugpy-managed model discovery is independent of this optional augmentation.
    Never walk a worker's whole filesystem merely because the worker started.
    """
    raw = os.environ.get("HUGPY_WORKER_MODEL_SCAN_ROOTS")
    roots = raw.split(os.pathsep) if raw else []
    return [os.path.realpath(p) for p in roots if p and os.path.isdir(p)]


def _model_root(path: str, files: list[str], dirs: list[str]) -> bool:
    if "hugpy.json" in files or "model_index.json" in files:
        return True
    if any(f.lower().endswith(".gguf") and not f.lower().startswith("mmproj")
           for f in files):
        return True
    if "config.json" in files and any(
            f.lower().endswith((".safetensors", ".bin", ".pt")) for f in files):
        return True
    # A GGUF repository may keep each quant's shards one level below its root.
    # Inspect quant-looking children only, so an owner directory containing
    # several repos is not mistaken for one model.
    for child in dirs:
        low = child.lower()
        if not (low.startswith(("q2", "q3", "q4", "q5", "q6", "q8", "iq",
                                "f16", "bf16", "fp16")) or
                os.path.basename(path).lower().endswith("gguf")):
            continue
        try:
            if any(name.lower().endswith(".gguf") for name in os.listdir(os.path.join(path, child))):
                return True
        except OSError:
            pass
    return False


def _complete_gguf_entrypoint(path: str) -> bool:
    """Whether a GGUF entrypoint has every shard llama.cpp will require.

    ``model_looks_downloaded`` intentionally treats a partial directory as
    present so a downloader can resume it.  Host discovery has the opposite
    contract: it must never advertise an unloadable model to fleet routing.
    """
    from hugpy_engine.gguf_election import parse_shard

    entry = Path(path)
    parsed = parse_shard(entry.name)
    if not parsed:
        return entry.is_file()
    stem, _idx, total = parsed
    return all((entry.parent / f"{stem}-{number:05d}-of-{total:05d}.gguf").is_file()
               for number in range(1, total + 1))


def _host_candidates() -> tuple[list[str], list[str]]:
    from hugpy_platform.constants import MODELS_HOME
    managed = os.path.realpath(str(MODELS_HOME))
    inventory = os.path.realpath(str(_inventory_dir()))
    found: list[str] = []
    manifests: list[str] = []
    seen: set[str] = set()
    for root in _roots():
        for path, dirs, files in os.walk(root, followlinks=False):
            real = os.path.realpath(path)
            if _under(real, managed) or _under(real, inventory):
                dirs[:] = []
                continue
            rel = os.path.relpath(path, root)
            dirs[:] = [d for d in dirs if d not in _SKIP
                       and os.path.join(rel, d).strip("./") not in _SKIP
                       and not os.path.islink(os.path.join(path, d))]
            if f"{os.sep}manifests{os.sep}" in path:
                manifests.extend(os.path.join(path, f) for f in files
                                 if not os.path.islink(os.path.join(path, f)))
            if _model_root(path, files, dirs) and real not in seen:
                found.append(real)
                seen.add(real)
                dirs[:] = []
    return found, manifests


def model_directories() -> list[str]:
    return _host_candidates()[0]


def _inventory_dir() -> Path:
    override = os.environ.get("HUGPY_WORKER_MODEL_INVENTORY_DIR")
    return Path(override or (Path.home() / ".hugpy" / "worker-models"))


def _identity(path: str, marker: dict) -> tuple[str, str]:
    hub = str(marker.get("hub_id") or "").strip("/")
    if not hub:
        parts = Path(path).parts
        cache = next((p for p in parts if p.startswith("models--") and
                      p.count("--") >= 2), "")
        if cache:
            hub = cache[len("models--"):].replace("--", "/", 1)
    name = str(marker.get("name") or (hub.rsplit("/", 1)[-1] if hub else
                                     Path(path).name))
    return name, hub or f"local/{name}"


def _ollama_tags() -> dict[str, dict]:
    """Only advertise manifests that the local Ollama server actually serves."""
    url = os.environ.get("HUGPY_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=3) as response:
            data = json.load(response)
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            return {}
        return {r["name"]: r for r in data.get("models", [])
                if isinstance(r, dict) and isinstance(r.get("name"), str)}
    except (OSError, ValueError, KeyError):
        return {}


def _ollama_rows(paths: list[str]) -> list[tuple[str, dict]]:
    tags = _ollama_tags()
    if not tags:
        return []
    result = []
    for path in paths:
        try:
            manifest_at = Path(path).parts.index("manifests")
            parts = Path(path).parts[manifest_at + 1:]
            if len(parts) < 4:
                continue
            name = "/".join(parts[2:-1]) + ":" + parts[-1]
            if parts[1] != "library":
                name = parts[0] + "/" + parts[1] + "/" + name
            tag = tags.get(name)
            if not tag:
                continue
            raw = Path(path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != tag.get("digest"):
                continue
            manifest = json.loads(raw)
            layers = manifest.get("layers") or []
            model_layers = [layer for layer in layers
                            if layer.get("mediaType") == "application/vnd.ollama.image.model"]
            if not model_layers:
                continue
            # The manifests tree and blobs tree are siblings in Ollama's store.
            store = Path(path).parents[len(parts)]
            for layer in [manifest.get("config"), *layers]:
                if not isinstance(layer, dict):
                    raise ValueError("invalid Ollama manifest layer")
                digest = layer["digest"]
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                    raise ValueError("invalid Ollama blob digest")
                blob = store / "blobs" / digest.replace(":", "-")
                if not blob.is_file() or blob.stat().st_size != layer["size"]:
                    raise ValueError("missing Ollama model blob")
                with blob.open("rb") as fh:
                    fh.read(1)  # worker must be able to read the weights
            safe = re.sub(r"[^A-Za-z0-9._-]+", "~", name)
            # The worker adapter currently implements chat/completion. Keep
            # other Ollama capabilities out of routing until they have a wire.
            if "completion" not in (tag.get("capabilities") or []):
                continue
            tasks = ["text-generation"]
            result.append((path, {
                "model_key": "ollama~" + safe, "name": name,
                # The remote relay borrows GGUF's chat request/result types.
                "hub_id": "ollama/" + safe, "framework": "gguf",
                "tasks": tasks, "primary_task": tasks[0],
                "ollama_model": name, "dir": path,
                "external_location": path,
                "size_bytes": sum(int(layer["size"]) for layer in model_layers),
            }))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    return result


def scan_system_models() -> dict[str, dict]:
    """Return discoverable host models and stamp external location pointers."""
    from hugpy_engine.apis.get_module import build_resolver_chain, enrich
    from hugpy_engine.config.main import get_gguf_file
    from hugpy_storage.hugpy_marker import read_hugpy_marker
    from hugpy_storage.model_presence import model_looks_downloaded
    from hugpy_engine.config.models.models_config import MODEL_REGISTRY

    chain = build_resolver_chain(use_hub=False)
    pending: list[tuple[str, dict]] = []
    directories, manifests = _host_candidates()
    for path in directories:
        marker = read_hugpy_marker(path) or {}
        name, hub = _identity(path, marker)
        meta, _sources = enrich(path, hub, chain)
        row = meta.to_dict()
        row.update({k: v for k, v in marker.items()
                    if k in {"name", "hub_id", "framework", "tasks", "primary_task",
                             "filename", "model_max_length"} and v is not None})
        row.update(name=name, hub_id=hub, dir=path, external_location=path)
        gguf = get_gguf_file(path, None)
        if gguf:
            if not _complete_gguf_entrypoint(gguf):
                continue
            row["framework"] = "gguf"
            row["filename"] = os.path.relpath(gguf, path)
            if not row.get("tasks"):
                row["tasks"] = ["text-generation"]
            if not row.get("primary_task"):
                row["primary_task"] = row["tasks"][0]
            row["size_bytes"] = os.path.getsize(gguf)
            row["effective_bytes"] = row["size_bytes"]
        else:
            weight_bytes = 0
            for base, _dirs, files in os.walk(path, followlinks=False):
                for filename in files:
                    if filename.lower().endswith((".safetensors", ".bin", ".pt")):
                        try:
                            weight_bytes += os.path.getsize(os.path.join(base, filename))
                        except OSError:
                            pass
            if weight_bytes:
                row["size_bytes"] = weight_bytes
        if not model_looks_downloaded(path, row):
            continue
        pending.append((path, row))
    pending.extend(_ollama_rows(manifests))

    # A host can retain more than one live-looking copy of the same Hugging Face
    # repo.  They are one catalog identity, not independently routable models:
    # publishing both makes the later path acquire a hash suffix while an older
    # path keeps the canonical key.  Keep the deterministic first non-staging
    # candidate; archive/staging trees were pruned above.
    unique: dict[tuple[str, str], tuple[str, dict]] = {}
    for path, row in pending:
        identity = (str(row.get("hub_id") or "").strip("/").lower(),
                    str(row.get("name") or "").lower())
        unique.setdefault(identity, (path, row))
    pending = list(unique.values())

    counts: dict[str, int] = {}
    for _, row in pending:
        counts[row["name"]] = counts.get(row["name"], 0) + 1
    out: dict[str, dict] = {}
    inventory = _inventory_dir()
    inventory.mkdir(parents=True, exist_ok=True)
    for path, row in pending:
        name = row["name"]
        owner = row["hub_id"].split("/", 1)[0]
        key = row.get("model_key") or (f"{owner}~{name}" if counts[name] > 1 else name)
        current = MODEL_REGISTRY.get(key)
        if current and str(current.hub_id or "").strip("/").lower() != row["hub_id"].lower():
            key = f"{owner}~{name}"
        if key in out or (key in MODEL_REGISTRY and
                          str(MODEL_REGISTRY[key].hub_id or "").strip("/").lower()
                          != row["hub_id"].lower()):
            key += "~" + hashlib.sha256(path.encode()).hexdigest()[:8]
        row["model_key"] = key
        out[key] = row
        pointer = {k: row.get(k) for k in ("model_key", "name", "hub_id", "framework",
                                          "tasks", "primary_task", "filename", "ollama_model")}
        pointer["location"] = path
        target = inventory / hashlib.sha256(path.encode()).hexdigest() / "hugpy.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(pointer, sort_keys=True, indent=2) + "\n"
        if not target.exists() or target.read_text() != payload:
            tmp = target.with_suffix(".tmp")
            tmp.write_text(payload)
            os.replace(tmp, target)
    live = {hashlib.sha256(path.encode()).hexdigest() for path, _ in pending}
    for entry in inventory.iterdir():
        if entry.is_dir() and entry.name not in live:
            marker = entry / "hugpy.json"
            marker.unlink(missing_ok=True)
            try:
                entry.rmdir()
            except OSError:
                pass
    return out

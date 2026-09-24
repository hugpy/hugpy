"""hugpy-model-resweep — archive a model's central directory and re-download it
THROUGH HUGPY, so the post-download admission gate judges the fresh copy.

Operator rulings (2026-09-23): "archive or live, nothing in between"; "no
removing originals" (archive = move, never delete); the only Hub traffic is
the download itself.

Per model (``--from`` the sweep JSON, default
``$PROJECTS_HOME/redownload_sweep_2026-09-23.json``, or ``--only KEY ...``):

  1. REFUSE when the model is loaded / serving / seated on any worker
     (``/api/models?verbose=1`` ``workers[]``), unless ``--force``.
  2. MOVE central's model directory (the catalog ``destination``) with one
     ``rename`` — atomic, same filesystem, never a copy+delete — to
     ``ARCHIVE/MODELS_RESWEEP-<date>/<framework>/<owner>/<name>/`` and append a
     MANIFEST.md row (original, archived, why, restore command). A Civitai
     checkpoint's drop-file (``<root>/checkpoints/<file>`` + its
     ``.civitai.json``) moves beside it (``<name>.checkpoint/``): it is the
     same model's central copy, and leaving it would make /civitai/download
     answer "already".
  3. Refresh the catalog the way hugpy does when files vanish: the model's
     detail read (``GET /api/models/<key>``) is the explicit re-derive path —
     it rewrites the persisted status to not_installed for every surface.
     Worker copies are left to hugpy's own storage reconcile; this tool never
     touches a worker.
  4. REQUEST the download through hugpy's own path — HF: ``POST
     /llm/repos/download`` (the console's add-models route; it enqueues the
     job the downloader daemon runs); Civitai: ``POST /civitai/download`` with
     the id/version recorded locally (sidecar / metadata store).
     GGUF quant rule: the catalog's pinned quant when it is a language quant
     that fits; otherwise the largest LANGUAGE quant (never ``mmproj/`` / a
     projector) that fits the largest online GPU (+0.5 GiB reserve, the
     audit's fit rule), else the largest that fits one worker's RAM+VRAM
     (expert/CPU offload). The file listing comes from the LOCAL metadata
     store the discovery/download path filled — no Hub call to choose.
  5. WAIT (bounded, ``--timeout``) for the admission verdict the install hook
     queues, and (6) print it.

Dry-run by default; ``--apply`` performs. Refuses to apply before the
admission gate is live on central (``GET /llm/admission`` answering, hugpy-ops
installed there) — a re-download nobody judges is not a resweep.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import sys
import time
import urllib.parse
from typing import Any, Optional

from hugpy_ops import model_audit as ma

DEFAULT_SWEEP = "redownload_sweep_2026-09-23.json"
ARCHIVE_ROOT = "/mnt/16T_toshiba/llm_storage/ARCHIVE"
FALLBACK_ROOT = "/mnt/16T_toshiba/llm_storage"
RESERVE = 2 ** 29
RAM_HEADROOM = 0.90
DEFAULT_TIMEOUT = 6 * 3600.0


# ── pure planning helpers ────────────────────────────────────────────────────

def is_language_gguf(rel: str) -> bool:
    low = rel.lower()
    if not low.endswith(".gguf"):
        return False
    if low.startswith("mmproj/") or "/mmproj/" in low or ma.is_projector(rel):
        return False
    return os.path.basename(low) != "imatrix.gguf" and "imatrix" not in os.path.basename(low)


def gguf_groups(siblings: list) -> list:
    """Language GGUF quants in a repo listing, shards summed:
    ``[{"quant": logical name, "files": [...], "bytes": n}]``."""
    groups: dict = {}
    for s in siblings or []:
        rel = s.get("rfilename") or s.get("path") or ""
        if not is_language_gguf(rel):
            continue
        k = (os.path.dirname(rel), ma.logical_gguf(rel))
        g = groups.setdefault(k, {"quant": k[1], "files": [], "bytes": 0, "sized": True})
        g["files"].append(rel)
        if s.get("size") is None:
            g["sized"] = False
        g["bytes"] += int(s.get("size") or 0)
    return sorted(groups.values(), key=lambda g: g["bytes"])


def fleet_budgets(workers: list) -> dict:
    online = [w for w in workers or [] if w.get("status") == "online"]
    gpu = max([int(w.get("gpu_total_bytes_known") or 0) for w in online] or [0])
    best_mem, best_name = 0, None
    for w in online:
        m = int(w.get("ram_bar_total") or 0) + int(w.get("gpu_total_bytes_known") or 0)
        if m > best_mem:
            best_mem, best_name = m, w.get("name")
    return {"gpu": gpu, "mem": best_mem, "mem_worker": best_name,
            "gpu_worker": next((w.get("name") for w in online
                                if int(w.get("gpu_total_bytes_known") or 0) == gpu), None)}


def choose_quant(siblings: list, workers: list, pinned: Optional[str] = None) -> dict:
    """The language quant to download (see module doc, step 4)."""
    groups = [g for g in gguf_groups(siblings) if g["sized"]]
    b = fleet_budgets(workers)
    if not groups:
        return {"error": "no sized language GGUF in the cached repo listing"}
    if pinned and is_language_gguf(pinned):
        for g in groups:
            if pinned in g["files"] or os.path.basename(pinned) in {os.path.basename(f) for f in g["files"]}:
                if g["bytes"] + RESERVE <= b["gpu"]:
                    return {**g, "rule": f"catalog pin {pinned} is a language quant that fits "
                            f"{b['gpu_worker']} ({ma._gb(b['gpu'])} GPU)"}
    fits = [g for g in groups if g["bytes"] + RESERVE <= b["gpu"]]
    if fits:
        g = fits[-1]
        return {**g, "rule": f"largest language quant fitting the largest online GPU "
                f"({b['gpu_worker']}, {ma._gb(b['gpu'])} incl. 0.5 GiB reserve)"}
    mem_fits = [g for g in groups if g["bytes"] <= b["mem"] * RAM_HEADROOM]
    if mem_fits:
        g = mem_fits[-1]
        return {**g, "rule": f"no quant fits any GPU (largest {ma._gb(b['gpu'])}); largest fitting "
                f"{b['mem_worker']} RAM+VRAM ({ma._gb(b['mem'])}, 90%) — needs expert/CPU offload"}
    return {"error": f"no language quant fits the fleet (smallest {ma._gb(groups[0]['bytes'])}; "
                     f"largest GPU {ma._gb(b['gpu'])}, largest RAM+VRAM {ma._gb(b['mem'])})"}


def loaded_on(row: Optional[dict]) -> list:
    """Workers where the model is loaded / serving / seated right now."""
    out = []
    for w in (row or {}).get("workers") or []:
        if w.get("loaded") or w.get("serving") or w.get("seat"):
            out.append(w.get("worker") or w.get("worker_id") or "?")
    return out


def archive_target(archive_base: str, entry: dict) -> str:
    hub = (entry.get("hub_id") or "").strip("/")
    owner, _, name = hub.partition("/")
    if not name:
        owner, name = "_", hub or entry["model_key"]
    return os.path.join(archive_base, entry.get("framework") or "unknown", owner, name)


def manifest_row(model_key: str, original: str, archived: str, why: str,
                 extra: Optional[list] = None) -> str:
    restore = f'mv "{archived}" "{original}"'
    for src, dst in extra or []:
        restore += f' && mv "{dst}" "{src}"'
    return (f"| `{model_key}` | `{original}` | `{archived}` | {why} | "
            f"`{restore}` (then `curl -s $CENTRAL/api/models/{model_key}`) |")


MANIFEST_HEADER = (
    "# MODELS_RESWEEP — central model dirs archived before a hugpy re-download\n\n"
    "Moved with one `rename` each (same filesystem; nothing copied, nothing deleted) by "
    "`hugpy-model-resweep --apply`, then re-requested through hugpy's own download path so the "
    "post-download admission gate judges the fresh copy. Restore = the row's command (stop the "
    "fresh download first if it is still running; then the detail read refreshes the catalog).\n\n"
    "| model | original | archived | why | restore |\n|---|---|---|---|---|\n")


def append_manifest(archive_base: str, row: str) -> str:
    os.makedirs(archive_base, exist_ok=True)
    path = os.path.join(archive_base, "MANIFEST.md")
    fresh = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as fh:
        if fresh:
            fh.write(MANIFEST_HEADER)
        fh.write(row + "\n")
    return path


def move(src: str, dst: str) -> None:
    """ONE rename. Refuses an existing target and a cross-device move —
    archive or live, nothing in between."""
    if os.path.lexists(dst):
        raise FileExistsError(f"archive target already exists: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.rename(src, dst)


# ── local facts (no network) ─────────────────────────────────────────────────

def metadata_db_path(explicit: Optional[str] = None) -> Optional[str]:
    """The metadata store central's server fills: explicit, env, the hugpy env
    file, the service account's HUGPY_HOME, else this user's default."""
    for cand in (explicit, os.environ.get("HUGPY_MODEL_METADATA_DB"),
                 ma._env_file_value("HUGPY_MODEL_METADATA_DB"),
                 os.path.join(os.path.expanduser("~hugpy"), ".hugpy", "state", "model_metadata.db")):
        if cand and os.path.isfile(cand):
            return cand
    try:
        from hugpy_storage.model_metadata import default_db_path
        cand = default_db_path()
        return cand if os.path.isfile(cand) else None
    except Exception:  # noqa: BLE001
        return None


def storage_root() -> str:
    """Central's storage root: env, the hugpy env file, then the constant
    (which on an operator shell may be a WORKER's root — hence the order)."""
    v = os.environ.get("DEFAULT_ROOT") or ma._env_file_value("DEFAULT_ROOT")
    if v:
        return v
    from hugpy_platform.constants import DEFAULT_ROOT
    return str(DEFAULT_ROOT or FALLBACK_ROOT)


def _ro_payload(db_path: Optional[str], table: str, col: str, key: str) -> Optional[dict]:
    """One JSON payload from the metadata store, opened READ-ONLY (the store
    belongs to central's server account; the resweep never writes it)."""
    if not db_path:
        return None
    import sqlite3
    for uri in (f"file:{db_path}?mode=ro", f"file:{db_path}?immutable=1"):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=10)
            try:
                row = conn.execute(f"SELECT payload FROM {table} WHERE {col}=?", (key,)).fetchone()
            finally:
                conn.close()
            return json.loads(row[0]) if row else None
        except (sqlite3.Error, ValueError):
            continue
    return None


def cached_repo_info(hub_id: str, db_path: Optional[str]) -> Optional[dict]:
    return _ro_payload(db_path, "repo_info", "hub_id", hub_id)


def civitai_source(filename: str, checkpoints_dir: str, db_path: Optional[str]) -> Optional[dict]:
    """``{civitai_id, version_id, name, base_model}`` from the checkpoint's
    ``.civitai.json`` sidecar or the local metadata store — never Civitai."""
    try:
        with open(os.path.join(checkpoints_dir, filename + ".civitai.json"), encoding="utf-8") as fh:
            sc = json.load(fh)
        if sc.get("version_id"):
            return {k: sc.get(k) for k in ("civitai_id", "version_id", "name", "base_model")}
    except (OSError, ValueError):
        pass
    from hugpy_storage.model_metadata import checkpoint_stem
    meta = _ro_payload(db_path, "civitai_meta", "stem", checkpoint_stem(filename))
    if meta and meta.get("version_id"):
        return {k: meta.get(k) for k in ("civitai_id", "version_id", "name", "base_model")}
    return None


def checkpoint_filename(entry: dict, row: Optional[dict]) -> Optional[str]:
    fn = (row or {}).get("filename")
    if fn:
        return os.path.basename(fn)
    from hugpy_storage.hugpy_marker import read_hugpy_marker
    m = read_hugpy_marker(entry.get("destination") or "") or {}
    return os.path.basename(m["filename"]) if m.get("filename") else None


# ── plan ─────────────────────────────────────────────────────────────────────

def plan_one(entry: dict, row: Optional[dict], workers: list, *, archive_base: str,
             checkpoints_dir: str, db_path: Optional[str], force: bool = False) -> dict:
    key = entry["model_key"]
    dest = (row or {}).get("destination") or entry.get("destination")
    p: dict = {"model_key": key, "why": entry.get("why"), "framework": entry.get("framework"),
               "original": dest, "archived": archive_target(archive_base, entry), "moves": []}
    busy = loaded_on(row)
    p["loaded_on"] = busy
    if busy and not force:
        p["refused"] = f"loaded/serving on {', '.join(busy)} — unload it first or pass --force"
    if dest and os.path.lexists(dest):
        p["moves"].append((dest, p["archived"]))
    else:
        p["note"] = f"no directory at {dest!r} on central — nothing to archive"
    if entry.get("framework") == "comfy":
        fn = checkpoint_filename(entry, row)
        src = civitai_source(fn, checkpoints_dir, db_path) if fn else None
        if not fn or not src:
            p["error"] = f"no Civitai id/version recorded locally for {fn or key!r}"
            return p
        for f in (fn, fn + ".civitai.json"):
            cp = os.path.join(checkpoints_dir, f)
            if os.path.lexists(cp):
                p["moves"].append((cp, os.path.join(p["archived"] + ".checkpoint", f)))
        p["source"] = "civitai"
        p["request"] = {"path": "/civitai/download", "body": {
            "download_url": f"https://civitai.com/api/download/models/{src['version_id']}",
            "filename": fn, **src}}
        p["choice"] = f"civitai model {src.get('civitai_id')} version {src.get('version_id')} -> {fn}"
        return p
    hub = (row or {}).get("hub_id") or entry.get("hub_id")
    body = {"hub_id": hub, "framework": (row or {}).get("framework") or entry.get("framework"),
            "task": (row or {}).get("primary_task") or "text-generation",
            "name": hub.split("/")[-1], "register": False}
    if body["framework"] == "gguf":
        info = cached_repo_info(hub, db_path)
        if not info:
            p["error"] = (f"no cached file listing for {hub} in the local metadata store "
                          f"({db_path or 'default'}) — open it once in Add-models; the resweep "
                          "does not ask the Hub to choose")
            return p
        pick = choose_quant(info.get("siblings") or [], workers, pinned=(row or {}).get("filename"))
        if pick.get("error"):
            p["error"] = pick["error"]
            return p
        if len(pick["files"]) == 1:
            body["filename"] = pick["files"][0]
        else:
            body["include"] = pick["files"]
        body["total_bytes"] = pick["bytes"]
        p["choice"] = f"{pick['quant']} ({ma._gb(pick['bytes'])}): {pick['rule']}"
    else:
        for k in ("filename", "include"):
            if (row or {}).get(k):
                body[k] = row[k]
        p["choice"] = "catalog files (" + (body.get("filename") or str(body.get("include") or "full snapshot")) + ")"
    p["source"] = "huggingface"
    p["request"] = {"path": "/llm/repos/download", "body": body}
    return p


# ── central ──────────────────────────────────────────────────────────────────

class Central:
    def __init__(self, base: Optional[str] = None, tokens: Optional[list] = None):
        from hugpy_ops.admission import operator_tokens
        self.base = ma.central_url(base)
        self.tokens = tokens if tokens is not None else operator_tokens()

    def get(self, path: str):
        return ma._request(self.base + path, token=(self.tokens or [None])[0])

    def post(self, path: str, body: dict):
        return ma._authed("POST", self.base + path, body, self.tokens, ma.CENTRAL_TIMEOUT)


def admission_live(central: Central) -> tuple:
    """``(ok, why)`` — the gate must be installed locally AND live on central."""
    for mod in ("hugpy_storage.admission", "hugpy_ops.admission"):
        if importlib.util.find_spec(mod) is None:
            return False, f"{mod} is not installed here — promote the admission build first"
    try:
        st, body = central.get("/llm/admission?status=pending")
    except Exception as exc:  # noqa: BLE001
        return False, f"central unreachable: {exc}"
    if st != 200 or not isinstance(body, dict) or "models" not in body:
        shape = "non-JSON (the console's SPA fallback)" if isinstance(body, str) else type(body).__name__
        return False, (f"central's GET /llm/admission is not the admission route (HTTP {st}, {shape}) — "
                       "the running server predates the gate; it arrives via verify -> known-good -> promote")
    if not (body.get("runner") or {}).get("installed"):
        return False, "central's server has no hugpy-ops: admission jobs would never run"
    return True, "admission gate live on central"


def wait_admission(central: Central, key: str, since_iso: str, timeout: float,
                   sleep=time.sleep, clock=time.time, poll: float = 30.0,
                   download_check=None) -> dict:
    deadline = clock() + timeout
    started = clock()
    kicked = False
    while clock() < deadline:
        st, body = central.get("/llm/admission/" + urllib.parse.quote(key, safe="~"))
        block = (body or {}).get("admission") if st == 200 and isinstance(body, dict) else None
        fresh = bool(block) and (block.get("at") or "") > since_iso
        if fresh and block.get("status") in ("admitted", "held"):
            return block
        if download_check is not None:
            failed = download_check()
            if failed:
                return {"status": "download_failed", "reason": failed}
        # Lost install hook (live 2026-09-23: "admission hook failed ... database
        # is locked" left dreamshaper-8 with NO job while this loop waited the
        # full timeout). If the download has had two polls to land and no fresh
        # pending/verdict exists, request admission ourselves ONCE; if central
        # can't queue it either, say so and move on instead of timing out.
        if not fresh and not kicked and clock() - started >= 2 * poll:
            kicked = True
            rst, rbody = central.post("/llm/admission/" + urllib.parse.quote(key, safe="~") + "/rerun",
                                      {"reason": "resweep: install hook produced no admission job"})
            if rst not in (200, 201, 202):
                return {"status": "no_admission_job",
                        "reason": f"install hook queued no job and rerun -> HTTP {rst}: {str(rbody)[:300]}"}
        sleep(poll)
    return {"status": "timeout", "reason": f"no admission verdict within {int(timeout)}s"
            + (" (admission re-queued by the sweep; check GET /llm/admission/<key>)" if kicked else "")}


def _download_check(central: Central, plan: dict, resp: Any):
    if plan["source"] == "huggingface" and isinstance(resp, dict) and resp.get("id"):
        jid = resp["id"]

        def check():
            st, row = central.get(f"/jobs/{jid}")
            if st == 200 and isinstance(row, dict) and row.get("status") in ("failed", "cancelled", "expired"):
                return f"download job {jid} {row.get('status')}: {row.get('error') or row.get('message')}"
            return None
        return check
    fn = plan["request"]["body"].get("filename")

    def check_civitai():
        st, rows = central.get("/civitai/downloads")
        r = (rows or {}).get(fn) if isinstance(rows, dict) else None
        if r and r.get("status") == "failed":
            return f"civitai download of {fn} failed: {r.get('error')}"
        return None
    return check_civitai


# ── CLI ──────────────────────────────────────────────────────────────────────

def load_entries(path: Optional[str], only: Optional[list]) -> list:
    path = path or os.path.join(ma.projects_home(), DEFAULT_SWEEP)
    with open(path, "r", encoding="utf-8") as fh:
        entries = json.load(fh)
    if only:
        want = set(only)
        entries = [e for e in entries if e.get("model_key") in want]
        missing = want - {e.get("model_key") for e in entries}
        entries += [{"model_key": k, "why": "--only"} for k in sorted(missing)]
    return entries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hugpy-model-resweep", description=__doc__.split("\n\n")[0])
    p.add_argument("--from", dest="source", help=f"sweep JSON (default $PROJECTS_HOME/{DEFAULT_SWEEP})")
    p.add_argument("--only", nargs="+", metavar="KEY", help="only these model keys")
    p.add_argument("--apply", action="store_true", help="archive + request downloads + wait")
    p.add_argument("--force", action="store_true", help="proceed even when a worker has it loaded")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per-model wait for admission (s)")
    p.add_argument("--central", help="central base URL")
    p.add_argument("--archive-root", default=ARCHIVE_ROOT)
    p.add_argument("--date", default=_dt.date.today().isoformat())
    p.add_argument("--metadata-db", help="the metadata store central fills (default: autodetect)")
    p.add_argument("--json", action="store_true", help="print the plan as JSON")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    central = Central(args.central)
    ok, why = admission_live(central)
    if args.apply and not ok:
        print(f"hugpy-model-resweep: refusing --apply: {why}", file=sys.stderr)
        return 2
    try:
        entries = load_entries(args.source, args.only)
        catalog = ma.fetch_json(central.base + "/api/models?verbose=1")
        workers = ma.fetch_json(central.base + "/llm/workers")
    except Exception as exc:  # noqa: BLE001
        print(f"hugpy-model-resweep: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    catalog = catalog if isinstance(catalog, list) else (catalog or {}).get("models") or []
    workers = workers if isinstance(workers, list) else (workers or {}).get("workers") or []
    by_key = {r.get("model_key"): r for r in catalog}
    checkpoints = os.path.join(storage_root(), "checkpoints")
    base = os.path.join(args.archive_root, f"MODELS_RESWEEP-{args.date}")
    db = metadata_db_path(args.metadata_db)
    plans = [plan_one(e, by_key.get(e["model_key"]), workers, archive_base=base,
                      checkpoints_dir=checkpoints, db_path=db, force=args.force) for e in entries]
    print(f"admission gate: {'OK' if ok else 'NOT LIVE'} — {why}")
    print(f"archive: {base}   metadata store: {db or '(default)'}   mode: {'APPLY' if args.apply else 'dry-run'}")
    if args.json:
        print(json.dumps(plans, indent=1, default=str))
    for p in plans:
        state = "REFUSED" if p.get("refused") else "ERROR" if p.get("error") else "plan"
        print(f"\n[{state}] {p['model_key']} ({p['framework']}, {p['why']})")
        for src, dst in p["moves"]:
            print(f"  move  {src}\n     -> {dst}")
        if p.get("note"):
            print(f"  note  {p['note']}")
        if p.get("choice"):
            print(f"  get   {p.get('source')}: {p['choice']}")
        if p.get("refused") or p.get("error"):
            print(f"  why   {p.get('refused') or p.get('error')}")
    if not args.apply:
        return 0
    results = []
    for p in plans:
        if p.get("refused") or p.get("error"):
            results.append((p["model_key"], "skipped", p.get("refused") or p.get("error")))
            continue
        started = _dt.datetime.now(_dt.timezone.utc).isoformat()
        try:
            for src, dst in p["moves"]:
                move(src, dst)
            if p["moves"]:
                append_manifest(base, manifest_row(p["model_key"], p["moves"][0][0], p["moves"][0][1],
                                                   p["why"] or "", p["moves"][1:]))
            central.get("/api/models/" + urllib.parse.quote(p["model_key"], safe="~"))
            st, resp = central.post(p["request"]["path"], p["request"]["body"])
            if st not in (200, 201, 202):
                raise RuntimeError(f"{p['request']['path']} -> HTTP {st}: {str(resp)[:300]}")
            print(f"{p['model_key']}: archived, download requested ({p['source']}); waiting for admission…")
            verdict = wait_admission(central, p["model_key"], started, args.timeout,
                                     download_check=_download_check(central, p, resp))
            results.append((p["model_key"], verdict.get("status"), verdict.get("reason")))
        except Exception as exc:  # noqa: BLE001 — one model never sinks the sweep
            results.append((p["model_key"], "error", f"{type(exc).__name__}: {exc}"))
        print(f"{results[-1][0]}: {results[-1][1]} — {results[-1][2]}")
    print("\nADMISSION RESULTS")
    for k, s, r in results:
        print(f"  {s:<16} {k}: {r}")
    return 0 if all(s in ("admitted",) for _, s, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

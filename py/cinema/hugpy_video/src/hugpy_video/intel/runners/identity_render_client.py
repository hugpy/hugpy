"""Shared HTTP-client helpers for the IDENTITY RENDER SERVICE relays (k94).

Three bus runners relay work to the remote GPU ``IDENTITY_RENDER_URL`` service and
used to carry their own copies of the same submit / poll / download / persist code:

  * ``identity_render_relay.py``        — ``mesh_build`` / ``mesh_and_turntable``
  * ``identity_video_extract_relay.py`` — ``video_extract`` (char360 only)
  * ``identity_from_video.py``          — ``video_characters_glb`` (char360 + GLB, k94)

This module is the ONE place those code paths now live (factored, not copied — keeper
task k94). Every helper is a pure function over an injected ``requests_mod`` so a test
can stand up a localhost mock service (the existing relay tests do exactly that) and so
importing this module stays boot-cheap: nothing heavy is imported at module top.

Remote service contract (FIXED — the service is built to exactly this):
  * ``POST /jobs``                       -> 202 {job_id}
  * ``GET  /jobs/<id>``                  -> {job_id, status, error?, files?,
                                           stage?, progress?, log_tail?, updated?}
  * ``GET  /jobs/<id>/files/<path>``     -> raw bytes
  * ``DELETE /jobs/<id>``                -> best-effort cleanup
Auth: header ``X-Identity-Render-Token: <IDENTITY_RENDER_TOKEN>`` (missing/wrong -> 401).

Errors are DATA: every helper that can fail returns a ``RelayError`` (code, message,
retryable) for the caller to turn into its own ``JobResult(ok=False, …)`` — this module
never raises for an expected failure and never builds a JobResult itself (each runner
owns its mesh-state / profile side effects on failure). No pathlib; os.path only.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Per-request HTTP budget as a (connect, read) TUPLE: the read side stays generous (a
# POST may carry b64 reference images — a few MB; a file GET may be a large GLB) but the
# CONNECT side is short, so a down/firewalled service is detected in seconds, not a 120s
# hang per attempt (ae's firewall DROPs unknown ports — no RST — so a flat timeout eats
# the whole budget just discovering "nobody home"; keeper 2026-07-14).
HTTP_TIMEOUT_S = (10.0, 120.0)
# Poll cadence + whole-job deadline. The bus registers the identity relays with a 14400s
# timeout; poll a touch under it so a runner returns clean errors-as-data rather than
# being killed mid-poll. Overridable via env (shared by every relay — same service, same
# cadence — so tuning/tests move them together).
POLL_INTERVAL_S = float(os.getenv("IDENTITY_RENDER_POLL_INTERVAL_S", "5") or "5")
POLL_DEADLINE_S = float(os.getenv("IDENTITY_RENDER_DEADLINE_S", "14100") or "14100")


@dataclass(frozen=True)
class RelayError:
    """An expected relay failure as data — the caller wraps it in its own JobError."""
    code: str
    message: str
    retryable: bool


def service_config() -> tuple[str, str]:
    """``(url, token)`` from the environment, each "" when unset. The caller decides how
    to phrase the not-configured error (each relay has its own honest wording)."""
    url = (os.getenv("IDENTITY_RENDER_URL", "") or "").strip().rstrip("/")
    token = (os.getenv("IDENTITY_RENDER_TOKEN", "") or "").strip()
    return url, token


def not_configured_error(what: str) -> RelayError:
    """The shared honest not-configured text. ``what`` names the relayed work
    ("mesh builds", "char360 video-extracts", …)."""
    return RelayError(
        "not_configured",
        "the identity render service is not configured on this host — set "
        "IDENTITY_RENDER_URL and IDENTITY_RENDER_TOKEN (central has no GPU; "
        f"{what} are relayed to a remote GPU render service).",
        False)


def auth_headers(token: str) -> dict:
    return {"X-Identity-Render-Token": token}


# --------------------------------------------------------------------------- #
# filesystem helpers
# --------------------------------------------------------------------------- #
def atomic_write_bytes(dest: str, data: bytes) -> None:
    """Write *data* to *dest* atomically (unique temp in the dest dir + os.replace),
    mirroring identity_profiles' copy idiom so a crashed download never leaves a
    half-written artifact at the final name."""
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{dest}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)


def safe_rel(rel: str) -> str:
    """Reduce EVERY component of a service-supplied relative name to its basename and
    re-join with "/" so a hostile ``../`` can never escape a staging dir (defense-in-
    depth even though the service is trusted). "" when nothing survives."""
    return "/".join(
        os.path.basename(part)
        for part in (rel or "").replace("\\", "/").split("/") if part)


def dest_for(mesh_dir: str, turntable_dir: str, fpath: str) -> str:
    """Map a service file name to its durable destination under an identity's mesh dir.

    ``*.glb`` and ``*.json`` land at the mesh root; ``*.mp4`` and any ``frames/…`` land
    under ``turntable/`` (frames keep the ``frames/`` subdir). Every component is reduced
    to ``os.path.basename`` so a hostile ``../`` in a service-supplied name can never
    escape the mesh dir."""
    norm = (fpath or "").replace("\\", "/").lstrip("/")
    base = os.path.basename(norm)
    low = norm.lower()
    if low.endswith(".glb"):
        return os.path.join(mesh_dir, base)
    if low.endswith(".json"):
        return os.path.join(mesh_dir, base)
    if "frames/" in norm or (low.endswith(".png") and "frame" in low):
        return os.path.join(turntable_dir, "frames", base)
    # mp4 (turntable video) and anything else -> the turntable bucket.
    return os.path.join(turntable_dir, base)


# --------------------------------------------------------------------------- #
# HTTP: submit / poll / download / delete
# --------------------------------------------------------------------------- #
def submit_job(requests_mod, url: str, headers: dict, payload: dict):
    """POST ``payload`` to ``<url>/jobs``. Returns ``(remote_id, None)`` on a 202 with a
    non-empty ``job_id``, else ``(None, RelayError)``."""
    try:
        resp = requests_mod.post(f"{url}/jobs", json=payload, headers=headers,
                                 timeout=HTTP_TIMEOUT_S)
    except requests_mod.RequestException as exc:
        return None, RelayError(
            "render_unreachable",
            f"could not reach the identity render service at {url}: {exc}", True)
    if resp.status_code == 401:
        return None, RelayError(
            "render_unauthorized",
            "the identity render service rejected the token (HTTP 401)", False)
    if resp.status_code != 202:
        body = (resp.text or "")[:300]
        return None, RelayError(
            "render_rejected",
            f"the render service rejected the job (HTTP {resp.status_code}): {body}", True)
    try:
        remote_id = resp.json()["job_id"]
    except (ValueError, KeyError, TypeError):
        return None, RelayError(
            "render_bad_response",
            "the render service accepted the job but returned no job_id", True)
    if not isinstance(remote_id, str) or not remote_id:
        return None, RelayError(
            "render_bad_response", "the render service returned an empty job_id", True)
    return remote_id, None


def delete_remote(requests_mod, url: str, headers: dict, remote_id: str) -> None:
    """Best-effort ``DELETE /jobs/<id>`` — never fails the caller."""
    try:
        requests_mod.delete(f"{url}/jobs/{remote_id}", headers=headers, timeout=30.0)
    except requests_mod.RequestException:
        pass


def mirror_progress(pbody: dict, job_id: str, set_progress: Callable, source: str) -> None:
    """Mirror the service's ADDITIVE live-progress fields (stage / progress / log_tail /
    updated) into the media bus so GET /video/jobs/<id> shows a long — or wedged — remote
    job's stage + log instead of progress 0. Best-effort + wrapped: any field MAY be absent
    (an older service), and a DB hiccup here never fails the relay."""
    if not (isinstance(pbody, dict)
            and any(k in pbody for k in ("stage", "progress", "log_tail"))):
        return
    try:
        blob = {"source": source, "remote_updated": pbody.get("updated")}
        for k in ("stage", "progress", "log_tail"):
            if k in pbody:
                blob[k] = pbody.get(k)
        set_progress(job_id, blob)
    except Exception:  # noqa: BLE001 — progress mirror is best-effort only
        logger.debug("identity relay: progress stamp failed for %s", job_id, exc_info=True)


def poll_job(requests_mod, url: str, headers: dict, remote_id: str, job_id: str, *,
             label: str, is_cancelling: Callable[[str], bool],
             set_progress: Callable, progress_source: str,
             on_cancel: Optional[Callable[[], None]] = None):
    """Poll ``GET /jobs/<remote_id>`` every ``POLL_INTERVAL_S`` until done / error /
    cooperative cancel / ``POLL_DEADLINE_S``. Returns ``(files, None)`` on done, else
    ``(None, RelayError)`` — code ``"cancelled"`` for a user cancel (``on_cancel`` is
    invoked first so a caller can record its own state), ``"render_timeout"`` for the
    deadline, ``"render_failed"`` for a service-reported error. Transient poll hiccups
    (transport errors, non-JSON bodies) are tolerated until the deadline. ``label`` is
    woven into the messages (``"identity mesh build for profile 'x'"``)."""
    deadline = time.time() + POLL_DEADLINE_S
    while True:
        if is_cancelling(job_id):
            delete_remote(requests_mod, url, headers, remote_id)
            if on_cancel is not None:
                on_cancel()
            return None, RelayError("cancelled", f"{label} cancelled by user", False)
        if time.time() > deadline:
            delete_remote(requests_mod, url, headers, remote_id)
            return None, RelayError(
                "render_timeout",
                f"{label} did not finish within the deadline ({int(POLL_DEADLINE_S)}s)",
                True)
        try:
            pr = requests_mod.get(f"{url}/jobs/{remote_id}", headers=headers,
                                  timeout=HTTP_TIMEOUT_S)
            if pr.status_code == 200:
                pbody = pr.json()
                mirror_progress(pbody, job_id, set_progress, progress_source)
                status = pbody.get("status") if isinstance(pbody, dict) else None
                if status == "done":
                    return (pbody.get("files") or []), None
                if status == "error":
                    delete_remote(requests_mod, url, headers, remote_id)
                    msg = pbody.get("error") or "the render service reported an error"
                    return None, RelayError("render_failed", f"{label} failed: {msg}", True)
                # queued / running -> keep polling
        except requests_mod.RequestException:
            pass  # transient poll hiccup — keep polling until the deadline
        except ValueError:
            pass  # non-JSON status body — treat as transient
        time.sleep(POLL_INTERVAL_S)


def download_file(requests_mod, url: str, headers: dict, remote_id: str,
                  rel_name: str) -> Optional[bytes]:
    """GET one job-relative file from the service; return its bytes, or None on any
    non-200 / transport error (logged, never raised — a missing file is the caller's
    call to skip or fail on)."""
    try:
        fr = requests_mod.get(f"{url}/jobs/{remote_id}/files/{rel_name}",
                              headers=headers, timeout=HTTP_TIMEOUT_S)
    except requests_mod.RequestException:
        logger.warning("identity relay: failed to download %r", rel_name)
        return None
    if fr.status_code != 200:
        logger.warning("identity relay: file %r -> HTTP %s", rel_name, fr.status_code)
        return None
    return fr.content


@dataclass
class PersistedMesh:
    """What ``persist_mesh_files`` landed under an identity's mesh dir."""
    glb_path: Optional[str] = None
    mesh_json_path: Optional[str] = None
    video_path: Optional[str] = None
    frame_paths: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.frame_paths is None:
            self.frame_paths = []


def persist_mesh_files(requests_mod, url: str, headers: dict, remote_id: str,
                       files, mesh_dir: str, turntable_dir: str,
                       strip_prefix: str = "") -> PersistedMesh:
    """Download every name in ``files`` and persist it via ``dest_for`` under
    ``mesh_dir`` / ``turntable_dir``, classifying the results (GLB / mesh json / mp4 /
    PNG frames, the frames sorted into angular order). A file that fails to download
    or persist is skipped with a warning — the caller decides whether a missing GLB is
    fatal. ``strip_prefix`` (e.g. ``"char_00/"``) is removed from each name before the
    destination is mapped, so a per-character subdir on the service flattens into the
    identity's own mesh dir."""
    out = PersistedMesh()
    for f in files or ():
        if not isinstance(f, str) or not f.strip():
            continue
        data = download_file(requests_mod, url, headers, remote_id, f)
        if data is None:
            continue
        local_name = f[len(strip_prefix):] if strip_prefix and f.startswith(strip_prefix) else f
        dest = dest_for(mesh_dir, turntable_dir, local_name)
        try:
            atomic_write_bytes(dest, data)
        except OSError:
            logger.warning("identity relay: could not persist %r -> %s", f, dest)
            continue
        low = dest.lower()
        if low.endswith(".glb"):
            out.glb_path = dest
        elif low.endswith(".json"):
            out.mesh_json_path = dest
        elif low.endswith(".mp4"):
            out.video_path = dest
        elif low.endswith(".png"):
            out.frame_paths.append(dest)
    out.frame_paths.sort()  # frame_0000.png, frame_0001.png … == angular order
    return out

r"""Pure ``(mlt, render)`` runner (k22) — headless Kdenlive/MLT render via ``melt``.

``run_mlt_render(spec, job_id) -> JobResult`` renders an operator-authored ``.kdenlive`` /
``.mlt`` project server-side and writes ONE video output back under the studio
``edits/renders/`` tree. It is CPU-only (no GPU reservation template) and mirrors the
pure-runner discipline of ``ffmpeg_frames`` / ``ffmpeg_crop``:

  * EXPECTED failures (melt absent, project missing / outside the jail, an unresolvable clip
    path, melt nonzero, no output) are returned as ``JobResult(ok=False, JobError(...))`` —
    DATA, never a raise. Only ``media_bus.run_claimed`` catches an UNEXPECTED raise.

PATH-MAP (design point 1): a ``.kdenlive`` project embeds the AUTHORING machine's absolute
paths (Windows UNC ``\\host\studio\...`` and mapped-drive ``Z:\...`` forms). Before rendering
the runner REWRITES those to this VM's storage jail
(``/mnt/llm_storage/video_intel/studio/...``) — see ``rewrite_project_paths``. Anchoring on
the SHARE NAME (``studio`` / ``studio-edits``) makes host casing / a differing hostname
irrelevant. AFTER rewriting, every file-backed clip resource is CHECKED to exist; any
unresolvable resource makes the job an HONEST ``unresolved_resources`` error listing exactly
what is missing — NEVER a silent render with blanks.

ATOMIC OUTPUT (design point 6): melt renders to a unique ``.part`` sibling; the final name is
produced by ``os.replace`` ONLY when melt exits 0, so a Samba poller / an editor watching the
tree never observes a partial.

CANCEL (design point 4): melt runs as a child ``Popen``; the poll loop honors a cooperative
``is_cancelling(job_id)`` by terminating (then killing) the process and removing the ``.part``.

PROGRESS (design point 4): melt's ``progress=1`` consumer prints ``… percentage: NN`` to
stderr; a reader thread parses it into the media bus (``set_progress``) with a rolling log
tail, best-effort.

FONTS (design point 3): fonts-dejavu + fonts-liberation are installed VM-side so title clips
render. Operator-side CUSTOM fonts must be installed on THIS VM to match — an absent font
falls back silently in melt (a known limitation, surfaced in the runner log, not fatal).

No pathlib anywhere. os.path only. Heavy stdlib (subprocess/xml) is imported at module top —
it is all stdlib, so the import stays boot-cheap (runners/__init__ imports this at boot).
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from typing import List, Optional, Tuple

from hugpy_platform.constants import DEFAULT_ROOT

from hugpy_video.intel.mlt_render_schema import MltRenderSpec
from hugpy_video.intel.result_schema import JobError, JobResult

logger = logging.getLogger(__name__)

# The studio tree (the Samba [studio] share's backing dir) + its writable edits subtree.
# Kept local (os.path.join off DEFAULT_ROOT) to avoid importing the studio spine here —
# these are stable, and studio.job derives STUDIO_ROOT identically.
STUDIO_ROOT = os.path.join(DEFAULT_ROOT, "video_intel", "studio")
EDITS_ROOT = os.path.join(STUDIO_ROOT, "edits")
RENDERS_ROOT = os.path.join(EDITS_ROOT, "renders")

# Share-name -> jail root. Anchoring the rewrite on the SHARE NAME (not the host) makes any
# host casing / a differing hostname irrelevant. Longest name first so the alternation below
# matches "studio-edits" before "studio".
_SHARE_ROOTS = (
    ("studio-edits", EDITS_ROOT),
    ("studio", STUDIO_ROOT),
)

# UNC form: \\host\share\tail  OR  //host/share/tail (either slash, one or two leading).
# host is any run of non-separator chars (so any casing / a hostname all match); share is one
# of the known names (case-insensitive); tail runs until an XML/quote delimiter. Group order:
# (host, share, tail).
_UNC_RE = re.compile(
    r"[\\/]{1,2}(?P<host>[^\\/\"<>]+)[\\/](?P<share>studio-edits|studio)(?P<tail>[\\/][^\"<>]*)",
    re.IGNORECASE,
)

# melt's progress line, e.g. "Current Frame:   9, percentage:   90".
_PCT_RE = re.compile(r"percentage:\s*(\d+)")

# How long to wait for a terminated melt to exit before SIGKILL (cancel path).
_KILL_GRACE_S = 5.0

# Startup probe memo (log the melt version ONCE per process).
_probe_lock = threading.Lock()
_probe_done = False


def probe_melt() -> Optional[str]:
    """Return melt's version string (e.g. ``"melt 7.22.0"``) or None if melt is absent /
    unrunnable. Logs the version ONCE per process (design point 2 startup probe)."""
    global _probe_done
    try:
        out = subprocess.run(["melt", "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    line = (out.stdout or "").splitlines()[0].strip() if out.stdout else ""
    with _probe_lock:
        if not _probe_done:
            logger.info("mlt_render: melt available -> %s", line or "(version unknown)")
            _probe_done = True
    return line or "melt (version unknown)"


def _convert_tail(tail: str) -> str:
    """Turn a matched Windows path tail (leading separator + rest) into a clean POSIX
    relative path: drop the leading separator, convert backslashes to '/', drop empty/'.'
    components (a jail-escape '..' is impossible here — the tail is under the share root)."""
    parts = [p for p in tail.replace("\\", "/").split("/") if p not in ("", ".")]
    return "/".join(parts)


def rewrite_project_paths(
    xml_text: str,
    drive_letter: Optional[str] = None,
    drive_root: Optional[str] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    r"""Rewrite authoring-machine share paths in an MLT/Kdenlive project to this VM's jail.

    Handles (design point 1):
      * UNC ``\\host\studio\...`` / ``//host/studio/...`` (either slash, any host casing) and
        the writable ``studio-edits`` share -> the matching jail root.
      * an OPTIONAL mapped-drive form (``drive_letter`` e.g. ``"Z:"``) -> ``drive_root``
        (defaults to the ``edits/`` root — the writable share the operator maps).

    Returns ``(new_xml, rewrites)`` where ``rewrites`` is a list of ``(old, new)`` pairs (for
    logging / tests). Pure string work — no filesystem access."""
    rewrites: List[Tuple[str, str]] = []
    share_map = {name.lower(): root for name, root in _SHARE_ROOTS}

    def _unc_sub(m: "re.Match") -> str:
        root = share_map[m.group("share").lower()]
        rel = _convert_tail(m.group("tail"))
        new = os.path.join(root, rel) if rel else root
        rewrites.append((m.group(0), new))
        return new

    out = _UNC_RE.sub(_unc_sub, xml_text)

    if drive_letter:
        letter = drive_letter.rstrip(":")
        root = drive_root or EDITS_ROOT
        drive_re = re.compile(
            r"(?i)" + re.escape(letter) + r":(?P<tail>[\\/][^\"<>]*)")

        def _drive_sub(m: "re.Match") -> str:
            rel = _convert_tail(m.group("tail"))
            new = os.path.join(root, rel) if rel else root
            rewrites.append((m.group(0), new))
            return new
        out = drive_re.sub(_drive_sub, out)

    return out, rewrites


def _looks_like_file_resource(res: str) -> bool:
    """True if a producer ``resource`` value should reference a real file on disk (so a miss
    is an error). Skips synthetic producers (color ``#rrggbb`` / ``color:``), text producers,
    and non-path service resources. A value whose pre-``?`` part is an absolute POSIX path, OR
    that still carries an UNRESOLVED Windows prefix, is a file candidate."""
    if not res or not isinstance(res, str):
        return False
    head = res.split("?", 1)[0].strip()
    if not head:
        return False
    low = head.lower()
    if head.startswith("#") or low.startswith(("color:", "colour:", "qtext:", "pixbuf:",
                                                "consumer:", "blipflash:", "noise:", "count:",
                                                "tone:", "frei0r.", "channelcopy")):
        return False
    return True


def _resource_missing(res: str) -> Optional[str]:
    """Return the offending path if ``res`` is a file candidate that does NOT resolve to an
    existing file, else None. An UNRESOLVED Windows-ish prefix (still ``\\``, drive-letter, or
    ``//``) counts as missing (the path-map could not map it)."""
    head = res.split("?", 1)[0].strip()
    if ("\\" in head) or re.match(r"^[A-Za-z]:", head) or head.startswith("//"):
        return head  # never got mapped to a POSIX jail path
    if head.startswith("/"):
        return None if os.path.isfile(head) else head
    # a bare relative resource (rare) — not something we can vouch for; treat as missing so
    # the operator sees it rather than a silent blank.
    return head


def _collect_unresolved(xml_text: str) -> List[str]:
    """Parse the (already path-mapped) project and return every file-backed clip resource
    that does not resolve. Best-effort: a parse failure returns [] (melt itself will then be
    the authority — we never block a render on our own parser hiccup)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.debug("mlt_render: resource pre-check could not parse project XML", exc_info=True)
        return []
    missing: List[str] = []
    seen = set()
    for prop in root.iter("property"):
        if prop.get("name") != "resource":
            continue
        res = (prop.text or "").strip()
        if not _looks_like_file_resource(res):
            continue
        bad = _resource_missing(res)
        if bad and bad not in seen:
            seen.add(bad)
            missing.append(bad)
    return missing


def _resolve_output_path(spec: MltRenderSpec, job_id: str) -> str:
    """Compute the FINAL absolute output path under ``edits/renders/`` from spec.output_rel
    (already structurally validated as a non-escaping relative path). Empty -> derive
    ``<project-stem>_<job8>.<container>``. The result is guaranteed to sit under RENDERS_ROOT
    (the caller re-checks the jail)."""
    if spec.output_rel:
        rel = spec.output_rel
    else:
        stem = os.path.splitext(os.path.basename(spec.project_path))[0] or "render"
        rel = f"{stem}_{job_id[:8]}.{spec.container}"
    return os.path.normpath(os.path.join(RENDERS_ROOT, rel))


from hugpy_platform.filesystem import is_within as _is_within


def _write_custom_profile(dst_dir: str, width: int, height: int,
                          fps: Optional[float]) -> str:
    """Write a melt profile file for an explicit width/height(/fps) override and return its
    path. fps None -> 25/1 (a sane default only reached when width/height are overridden but
    fps is not)."""
    num, den = (25, 1)
    if fps is not None:
        # represent fps as a rational: integer fps -> N/1; else scale by 1000.
        if float(fps).is_integer():
            num, den = int(fps), 1
        else:
            num, den = int(round(fps * 1000)), 1000
    prof = os.path.join(dst_dir, "profile.mltprofile")
    with open(prof, "w") as f:
        f.write(
            "description=hugpy-mlt-render\n"
            f"frame_rate_num={num}\nframe_rate_den={den}\n"
            f"width={width}\nheight={height}\nprogressive=1\n"
            "sample_aspect_num=1\nsample_aspect_den=1\n"
            f"display_aspect_num={width}\ndisplay_aspect_den={height}\n"
            "colorspace=709\n"
        )
    return prof


def run_mlt_render(spec: MltRenderSpec, job_id: str) -> JobResult:
    from hugpy_video.intel.media_bus import is_cancelling, set_progress
    from hugpy_video.intel.media_store import ingest

    def _fail(code: str, message: str, retryable: bool = False) -> JobResult:
        return JobResult(job_id=job_id, ok=False,
                         error=JobError(code=code, message=message, retryable=retryable))

    # ---- startup probe (design point 2): melt present? ----
    version = probe_melt()
    if version is None:
        return _fail("melt_missing",
                     "melt is not installed / not runnable on this host — install the "
                     "'melt' package (apt) so mlt_render can render Kdenlive/MLT projects.")

    # ---- jail the project path (design point 5): must live under the studio tree ----
    project = spec.project_path
    if not os.path.isfile(project):
        return _fail("missing_project", f"project file does not exist: {project}")
    if not _is_within(project, STUDIO_ROOT):
        return _fail("project_outside_jail",
                     f"project must live under the studio tree ({STUDIO_ROOT}); got {project}")

    # ---- resolve + jail the output path under edits/renders/ ----
    out_path = _resolve_output_path(spec, job_id)
    if not _is_within(os.path.dirname(out_path) or RENDERS_ROOT, RENDERS_ROOT) \
            or not _is_within(out_path, RENDERS_ROOT):
        return _fail("output_outside_jail",
                     f"resolved output escapes the renders jail ({RENDERS_ROOT}): {out_path}")

    # ---- read + PATH-MAP the project (design point 1) ----
    try:
        with open(project, "r", encoding="utf-8", errors="replace") as f:
            xml_text = f.read()
    except OSError as exc:
        return _fail("project_unreadable", f"could not read project: {exc}")

    drive_root = EDITS_ROOT if spec.drive_letter else None
    mapped_xml, rewrites = rewrite_project_paths(
        xml_text, drive_letter=spec.drive_letter, drive_root=drive_root)
    if rewrites:
        logger.info("mlt_render[%s]: rewrote %d authoring path(s) -> jail", job_id, len(rewrites))

    # ---- HONEST unresolved-resource gate (design point 1): never render with blanks ----
    missing = _collect_unresolved(mapped_xml)
    if missing:
        listed = "; ".join(missing[:20])
        more = "" if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
        return _fail("unresolved_resources",
                     f"{len(missing)} clip resource(s) referenced by the project could not be "
                     f"resolved under the studio jail after path-mapping: {listed}{more}. "
                     "Save the referenced media into the studio share and re-render.")

    # ---- stage the rewritten project + build the melt command ----
    stage_dir = os.path.join(RENDERS_ROOT, f".stage_{job_id[:8]}_{secrets.token_hex(3)}")
    try:
        os.makedirs(RENDERS_ROOT, exist_ok=True)
        os.makedirs(stage_dir, exist_ok=True)
    except OSError as exc:
        return _fail("stage_failed", f"could not prepare the render staging dir: {exc}",
                     retryable=True)

    part_path = f"{out_path}.{os.getpid()}.{secrets.token_hex(4)}.part"
    staged_project = os.path.join(stage_dir, "project" + os.path.splitext(project)[1].lower())

    def _cleanup() -> None:
        for p in (part_path, staged_project):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        try:
            # profile file + now-empty stage dir
            prof = os.path.join(stage_dir, "profile.mltprofile")
            if os.path.exists(prof):
                os.remove(prof)
            os.rmdir(stage_dir)
        except OSError:
            pass

    try:
        with open(staged_project, "w", encoding="utf-8") as f:
            f.write(mapped_xml)
    except OSError as exc:
        _cleanup()
        return _fail("stage_failed", f"could not stage the rewritten project: {exc}",
                     retryable=True)

    cmd = ["melt", staged_project]
    # profile override precedence: named profile > explicit width/height custom profile >
    # (nothing -> respect the project's embedded <profile>).
    if spec.profile:
        cmd += ["-profile", spec.profile]
    elif spec.width and spec.height:
        prof = _write_custom_profile(stage_dir, spec.width, spec.height, spec.fps)
        cmd += ["-profile", prof]
    cmd += ["-consumer", f"avformat:{part_path}",
            f"f={spec.container}", f"vcodec={spec.vcodec}", f"acodec={spec.acodec}",
            "progress=1"]
    if spec.vb:
        cmd.append(f"vb={spec.vb}")

    # ---- run melt as a child; parse progress + honor cancel (design point 4) ----
    log_tail: "deque[str]" = deque(maxlen=40)
    last_pct = {"v": -1}

    def _reader(stream) -> None:
        try:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\n")
                if line:
                    log_tail.append(line)
                m = _PCT_RE.search(line)
                if m:
                    last_pct["v"] = int(m.group(1))
        except Exception:  # noqa: BLE001 — a reader hiccup never fails the render
            pass
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True)
    except OSError as exc:
        _cleanup()
        return _fail("melt_spawn_failed", f"could not launch melt: {exc}", retryable=True)

    reader = threading.Thread(target=_reader, args=(proc.stderr,), daemon=True)
    reader.start()

    cancelled = False
    emitted_pct = -1
    while True:
        if proc.poll() is not None:
            break
        if is_cancelling(job_id):
            cancelled = True
            proc.terminate()
            try:
                proc.wait(timeout=_KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=_KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    pass
            break
        pct = last_pct["v"]
        if pct >= 0 and pct != emitted_pct:
            emitted_pct = pct
            try:
                set_progress(job_id, {"source": "mlt_render", "stage": "render",
                                      "progress": pct / 100.0,
                                      "log_tail": list(log_tail)[-8:]})
            except Exception:  # noqa: BLE001 — progress mirror is best-effort
                pass
        time.sleep(0.4)

    reader.join(timeout=2.0)
    rc = proc.returncode

    if cancelled:
        _cleanup()
        return JobResult(job_id=job_id, ok=False, error=JobError(
            code="cancelled", message="mlt_render cancelled by user", retryable=False))

    if rc != 0:
        tail = "\n".join(list(log_tail)[-20:])
        _cleanup()
        return _fail("melt_failed",
                     f"melt exited {rc}.\ncmd: {' '.join(cmd)}\nstderr tail:\n{tail}")

    if not os.path.isfile(part_path) or os.path.getsize(part_path) == 0:
        _cleanup()
        return _fail("missing_output",
                     "melt reported success but produced no (or an empty) output file")

    # ---- ATOMIC publish (design point 6): rename into the tree ONLY on melt exit 0 ----
    try:
        os.replace(part_path, out_path)
    except OSError as exc:
        _cleanup()
        return _fail("publish_failed", f"could not publish the render output: {exc}",
                     retryable=True)
    # stage dir cleanup (part already moved out)
    try:
        prof = os.path.join(stage_dir, "profile.mltprofile")
        if os.path.exists(prof):
            os.remove(prof)
        if os.path.exists(staged_project):
            os.remove(staged_project)
        os.rmdir(stage_dir)
    except OSError:
        pass

    logger.info("mlt_render[%s]: rendered -> %s", job_id, out_path)

    # Ingest the output so its dims/mime/duration are authoritatively resolved (§9.2) and it
    # surfaces as a MediaRef output. The path is under DEFAULT_ROOT (jailed), so /video/media
    # can serve it and GET /video/jobs/<id> -> result.outputs[0] carries the handle.
    try:
        ref = ingest(out_path)
    except Exception as exc:  # noqa: BLE001 — an ingest hiccup shouldn't lose a good render
        logger.warning("mlt_render[%s]: output produced but ingest failed: %s", job_id, exc)
        return _fail("ingest_failed",
                     f"render produced {out_path} but its metadata could not be resolved: {exc}",
                     retryable=True)
    return JobResult(job_id=job_id, ok=True, outputs=(ref,))

r"""Headless Kdenlive/MLT render (k22) — runner + route tests.

Covers the new ``mlt_render`` media_bus job (video_intel/mlt_render_schema.py +
runners/mlt_render.py) and its POST /video/mlt/render route:

  * PATH-MAP (rewrite_project_paths) — UNC ``\\host\studio\...`` (back- AND
    forward-slash, any host casing), the writable ``studio-edits`` share, and an
    optional mapped-drive form all rewrite to this VM's storage jail.
  * UNRESOLVED RESOURCES — a project referencing a clip that does not exist under the
    jail after mapping fails as HONEST ``unresolved_resources`` error-as-data listing
    the missing path (never a silent render with blanks).
  * SPEC VALIDATION / JAILING — make_mlt_render rejects a non-project extension, a
    relative project path, and an output_rel jail-escape; the runner refuses a project
    outside the studio tree and a missing project (all error-as-data / local raise).
  * ATOMIC OUTPUT (guarded on melt) — a REAL tiny render lands its output under
    edits/renders/ via an atomic rename, leaves NO ``.part`` sibling, and returns a
    MediaRef output.
  * CANCEL (guarded on melt) — a long render is killed when is_cancelling() flips, the
    result is ``cancelled``, and no partial/final output survives.
  * ROUTE GATING — a video-share guest is 403'd (operator-only); an operator enqueues
    (200 {job_id}); a project outside the jail 400s.

Script style (plain asserts, __main__ + numbered PASS lines) so it runs BOTH as
``venv/bin/python tests/test_mlt_render.py`` and via ``pytest tests/test_mlt_render.py -q``.
The melt-backed checks self-SKIP if melt is absent; everything else always runs.

Isolation: media_bus.DB_PATH is repointed to a PRIVATE temp sqlite db; synthetic
projects live in a temp dir UNDER the real studio tree (the runner requires a project
under STUDIO_ROOT) and outputs are unique-named under edits/renders/ — all cleaned up.
"""
from __future__ import annotations
import pytest

import importlib
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

logging.disable(logging.INFO)

os.environ.setdefault("PROJECTS_HOME", tempfile.mkdtemp(prefix="hugpy-mlt-test-"))
os.environ.pop("HUGPY_OPERATOR_TOKEN", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flask import Flask  # noqa: E402

oa = importlib.import_module("hugpy_server.app.operator_auth")
va = importlib.import_module("hugpy_server.app.video_auth")
vr = importlib.import_module("hugpy_server.app.routes.video_routes")

from hugpy_video.intel import media_bus
from hugpy_video.intel.mlt_render_schema import make_mlt_render
from hugpy_video.intel.runners import mlt_render as mr

_MELT = shutil.which("melt") is not None

_MINI_TMPL = """<?xml version="1.0" encoding="utf-8"?>
<mlt LC_NUMERIC="C" version="7.22.0">
  <profile description="HD 1080p 25" width="1920" height="1080" progressive="1"
    frame_rate_num="25" frame_rate_den="1" sample_aspect_num="1" sample_aspect_den="1"
    display_aspect_num="16" display_aspect_den="9" colorspace="709"/>
  <producer id="p0" in="0" out="{out}">
    <property name="mlt_service">color</property>
    <property name="resource">#112233</property>
    <property name="length">{length}</property>
    <filter id="fx"><property name="mlt_service">dynamictext</property>
      <property name="argument">k22</property><property name="family">DejaVu Sans</property>
      <property name="size">96</property></filter>
  </producer>
  <playlist id="main"><entry producer="p0" in="0" out="{out}"/></playlist>
  <tractor id="t0" in="0" out="{out}"><track producer="main"/></tractor>
</mlt>
"""


def _private_bus():
    tmpdir = tempfile.mkdtemp(prefix="hugpy_mlt_bus_")
    media_bus.DB_PATH = os.path.join(tmpdir, "media_jobs.db")
    media_bus._initialized = False
    return tmpdir


def _write_project(text: str) -> str:
    """Write a synthetic project into a temp dir UNDER the real studio tree (the runner
    jails project_path to STUDIO_ROOT) and return its path. Registered for cleanup."""
    d = tempfile.mkdtemp(prefix=".k22_test_", dir=mr.STUDIO_ROOT)
    _CLEANUP.append(d)
    p = os.path.join(d, "proj.kdenlive")
    with open(p, "w") as f:
        f.write(text)
    return p


_CLEANUP: list = []


def _cleanup_all():
    for d in _CLEANUP:
        shutil.rmtree(d, ignore_errors=True)
    _CLEANUP.clear()


# --------------------------------------------------------------------------- #
# 1) PATH-MAP — rewrite_project_paths
# --------------------------------------------------------------------------- #
def test_pathmap_unc_backslash():
    xml = (r'<property name="resource">\\192.168.1.250\studio\clips\ab\v.mp4</property>')
    out, rw = mr.rewrite_project_paths(xml)
    assert mr.STUDIO_ROOT + "/clips/ab/v.mp4" in out, out
    assert len(rw) == 1


def test_pathmap_unc_forwardslash_any_host_casing():
    xml = '<property name="resource">//MyHost/Studio/clips/x.mov</property>'
    out, _ = mr.rewrite_project_paths(xml)
    assert mr.STUDIO_ROOT + "/clips/x.mov" in out, out


def test_pathmap_studio_edits_share():
    xml = r'<property name="resource">\\192.168.1.250\studio-edits\proj\a.png</property>'
    out, _ = mr.rewrite_project_paths(xml)
    assert mr.EDITS_ROOT + "/proj/a.png" in out, out
    assert "studio-edits" not in out


def test_pathmap_drive_letter():
    xml = r'<property name="resource">Z:\proj\clip.mp4</property>'
    out, rw = mr.rewrite_project_paths(xml, drive_letter="Z:", drive_root=mr.EDITS_ROOT)
    assert mr.EDITS_ROOT + "/proj/clip.mp4" in out, out
    assert len(rw) == 1
    # without the drive_letter hint, a drive path is LEFT ALONE (later flagged unresolved)
    out2, rw2 = mr.rewrite_project_paths(xml)
    assert rw2 == [] and "Z:" in out2


# --------------------------------------------------------------------------- #
# 2) UNRESOLVED RESOURCES — honest error-as-data
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("melt") is None, reason="needs the melt (MLT) binary: run_mlt_render probes melt before any other check, so without it every path returns melt_missing")
def test_unresolved_resource_errors_as_data():
    _private_bus()
    # A project referencing a UNC clip that does NOT exist under the jail after mapping.
    xml = _MINI_TMPL.format(out=4, length=5).replace(
        '<property name="resource">#112233</property>',
        r'<property name="resource">\\192.168.1.250\studio\clips\NOPE\missing.mp4</property>')
    proj = _write_project(xml)
    spec = make_mlt_render(project_path=proj, output_rel="unresolved_test.mp4")
    res = mr.run_mlt_render(spec, "job-unresolved")
    assert res.ok is False and res.error.code == "unresolved_resources", res
    assert "missing.mp4" in res.error.message, res.error.message


# --------------------------------------------------------------------------- #
# 3) SPEC VALIDATION / JAILING
# --------------------------------------------------------------------------- #
def test_spec_validation():
    # bad extension
    try:
        make_mlt_render(project_path="/mnt/llm_storage/video_intel/studio/edits/x.txt")
        assert False, "expected bad-extension raise"
    except ValueError:
        pass
    # relative project
    try:
        make_mlt_render(project_path="proj.kdenlive")
        assert False, "expected non-absolute raise"
    except ValueError:
        pass
    # output_rel jail escape
    try:
        make_mlt_render(project_path="/mnt/llm_storage/video_intel/studio/a.kdenlive",
                        output_rel="../../etc/evil.mp4")
        assert False, "expected output_rel escape raise"
    except ValueError:
        pass
    # width without height
    try:
        make_mlt_render(project_path="/mnt/llm_storage/video_intel/studio/a.kdenlive",
                        width=1920)
        assert False, "expected width-without-height raise"
    except ValueError:
        pass


@pytest.mark.skipif(shutil.which("melt") is None, reason="needs the melt (MLT) binary: run_mlt_render probes melt before any other check, so without it every path returns melt_missing")
def test_runner_project_outside_jail():
    _private_bus()
    tmp = tempfile.mkdtemp(prefix="hugpy_mlt_outside_")
    _CLEANUP.append(tmp)
    p = os.path.join(tmp, "proj.kdenlive")
    with open(p, "w") as f:
        f.write(_MINI_TMPL.format(out=2, length=3))
    spec = make_mlt_render(project_path=p, output_rel="outside.mp4")
    res = mr.run_mlt_render(spec, "job-outside")
    assert res.ok is False and res.error.code == "project_outside_jail", res


@pytest.mark.skipif(shutil.which("melt") is None, reason="needs the melt (MLT) binary: run_mlt_render probes melt before any other check, so without it every path returns melt_missing")
def test_runner_missing_project():
    _private_bus()
    spec = make_mlt_render(
        project_path=os.path.join(mr.STUDIO_ROOT, "edits", "does_not_exist.kdenlive"),
        output_rel="missing.mp4")
    res = mr.run_mlt_render(spec, "job-missing")
    assert res.ok is False and res.error.code == "missing_project", res


# --------------------------------------------------------------------------- #
# 4) ATOMIC OUTPUT — real tiny render (guarded on melt)
# --------------------------------------------------------------------------- #
def test_atomic_render_output():
    if not _MELT:
        print("SKIP atomic render (melt absent)")
        return
    _private_bus()
    proj = _write_project(_MINI_TMPL.format(out=6, length=7))
    out_name = f"k22_atomic_{os.getpid()}.mp4"
    spec = make_mlt_render(project_path=proj, output_rel=out_name)
    res = mr.run_mlt_render(spec, "job-atomic")
    assert res.ok is True, res
    out_path = os.path.join(mr.RENDERS_ROOT, out_name)
    _CLEANUP_FILES.append(out_path)
    assert os.path.isfile(out_path), out_path
    # atomic: no .part sibling left behind
    leftovers = [f for f in os.listdir(mr.RENDERS_ROOT)
                 if f.startswith(out_name) and f.endswith(".part")]
    assert not leftovers, leftovers
    # a MediaRef output points at the produced file
    assert res.outputs and res.outputs[0].uri == os.path.realpath(out_path) or \
        res.outputs[0].uri == out_path, res.outputs


_CLEANUP_FILES: list = []


# --------------------------------------------------------------------------- #
# 5) CANCEL kills melt (guarded on melt)
# --------------------------------------------------------------------------- #
def test_cancel_kills_melt():
    if not _MELT:
        print("SKIP cancel (melt absent)")
        return
    _private_bus()
    # A long render (8000 frames of 1080p + a text filter) so melt is certainly still
    # running at the first cancel poll.
    proj = _write_project(_MINI_TMPL.format(out=8000, length=8001))
    out_name = f"k22_cancel_{os.getpid()}.mp4"
    spec = make_mlt_render(project_path=proj, output_rel=out_name)

    orig = media_bus.is_cancelling
    media_bus.is_cancelling = lambda job_id: True  # cancel on the first poll
    try:
        t0 = time.time()
        res = mr.run_mlt_render(spec, "job-cancel")
    finally:
        media_bus.is_cancelling = orig
    dt = time.time() - t0
    assert res.ok is False and res.error.code == "cancelled", res
    # killed promptly (nowhere near the ~30s+ full render)
    assert dt < 15, f"cancel took {dt:.1f}s (should be prompt)"
    out_path = os.path.join(mr.RENDERS_ROOT, out_name)
    assert not os.path.exists(out_path), "cancelled render must leave no final output"
    leftovers = [f for f in os.listdir(mr.RENDERS_ROOT)
                 if f.startswith(out_name) and f.endswith(".part")]
    assert not leftovers, f"cancelled render left a partial: {leftovers}"


# --------------------------------------------------------------------------- #
# 6) ROUTE GATING
# --------------------------------------------------------------------------- #
class _ApiPrefixMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == "/api" or path.startswith("/api/"):
            environ["PATH_INFO"] = path[len("/api"):] or "/"
        return self.wsgi_app(environ, start_response)


def _build_gated_app():
    app = Flask(__name__)
    app.register_blueprint(vr.video_bp)
    oa.install_operator_gate(app)
    va.install_video_gate(app)
    app.wsgi_app = _ApiPrefixMiddleware(app.wsgi_app)
    return app


_CLIENT = _build_gated_app().test_client()


def test_route_share_guest_denied():
    _private_bus()
    os.environ["HUGPY_AUTH_MODE"] = "external"
    os.environ.pop("HUGPY_OPERATOR_TOKEN", None)
    oa._SESSION_CACHE.clear()
    orig_sess, orig_share = oa._validate_session_external, va._video_share_principal
    try:
        oa._validate_session_external = lambda: False
        va._video_share_principal = lambda request: {"scope": "video"}
        r = _CLIENT.post("/api/video/mlt/render", json={"project_path": "/x/y.kdenlive"})
        assert r.status_code == 403, (r.status_code, r.get_data(as_text=True))
    finally:
        oa._validate_session_external, va._video_share_principal = orig_sess, orig_share


def test_route_operator_enqueues():
    _private_bus()
    os.environ["HUGPY_AUTH_MODE"] = "external"
    os.environ.pop("HUGPY_OPERATOR_TOKEN", None)
    oa._SESSION_CACHE.clear()
    orig_sess, orig_share = oa._validate_session_external, va._video_share_principal
    proj = _write_project(_MINI_TMPL.format(out=4, length=5))
    try:
        oa._validate_session_external = lambda: True
        va._video_share_principal = lambda request: None
        # outside-jail project 400s
        r_bad = _CLIENT.post("/api/video/mlt/render",
                             json={"project_path": "/etc/passwd.kdenlive"})
        assert r_bad.status_code == 400, r_bad.get_data(as_text=True)
        # a real jailed project enqueues
        r = _CLIENT.post("/api/video/mlt/render",
                         json={"project_path": proj, "output_rel": "route_enq.mp4"})
        assert r.status_code == 200, r.get_data(as_text=True)
        assert r.get_json().get("job_id"), r.get_json()
    finally:
        oa._validate_session_external, va._video_share_principal = orig_sess, orig_share


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    try:
        for i, t in enumerate(tests, 1):
            t()
            print(f"PASS {i:2d}: {t.__name__}")
            passed += 1
    finally:
        for f in _CLEANUP_FILES:
            try:
                os.remove(f)
            except OSError:
                pass
        _cleanup_all()
    print(f"\n{passed}/{len(tests)} passed")


if __name__ == "__main__":
    _run_all()

"""environment_report carries the build identity (WP2).

* ``build_report`` gains ``build`` = ``hugpy_platform.buildinfo.build_info()``.
* ``compact_digest`` (the heartbeat rider) gains ``build`` = the report's
  ``build.identity`` — lifted, not recomputed — and None when a report has no
  build (an older worker's cached report, the standalone seeding copy).
* ``build`` is a digested field, but its ``generated_at`` is not: a fresh
  computation of the same build must not move the digest.
* The probe is guarded: a platform without ``buildinfo`` reads None.
"""
from __future__ import annotations

import builtins

import pytest

from hugpy_fleet.worker import environment_report as er


@pytest.fixture
def cheap_probes(monkeypatch):
    """No child interpreters, no binary execs: only the build probe is real."""
    monkeypatch.setattr(er, "venvs_report", lambda: {"main": {"python": "p", "python_version": "3",
                                                              "packages": {"hugpy-fleet": "1"},
                                                              "error": None}})
    monkeypatch.setattr(er, "binaries_report", lambda: {})
    monkeypatch.setattr(er, "nvidia_report", lambda: {"driver": None, "cuda": None, "gpus": []})
    monkeypatch.setattr(er, "mounts_report", lambda: {})


def test_build_report_carries_build_document(cheap_probes):
    from hugpy_platform import buildinfo
    rep = er.build_report("w1")
    assert isinstance(rep["build"], dict)
    assert set(rep["build"]) >= {"identity", "distributions", "lockstep", "workspace"}
    assert rep["build"]["identity"]["version"] == buildinfo.build_identity()["version"]
    assert rep["pkg_version"] == er.pkg_version(), "pkg_version stays"


def test_compact_digest_carries_identity_from_the_report(cheap_probes):
    rep = er.build_report("w1")
    digest = er.compact_digest(rep)
    assert digest["build"] == rep["build"]["identity"]
    assert set(digest["build"]) >= {"version", "sha", "dirty", "editable", "source", "distribution"}
    assert digest["pkg_version"] == rep["pkg_version"]


def test_compact_digest_lifts_not_recomputes(monkeypatch):
    fake_identity = {"version": "9.9.9", "sha": "abc", "dirty": False,
                     "editable": False, "source": None, "distribution": "hugpy-fleet"}
    rep = {"report_digest": "d", "generated_at": "t", "python": "3", "pkg_version": "9.9.9",
           "venvs": {"main": {"packages": {}}}, "binaries": {}, "nvidia": {},
           "build": {"identity": fake_identity, "generated_at": "t"}}
    from hugpy_platform import buildinfo
    monkeypatch.setattr(buildinfo, "build_identity", lambda: (_ for _ in ()).throw(AssertionError("recomputed")))
    assert er.compact_digest(rep)["build"] == fake_identity


def test_compact_digest_without_build_is_none():
    rep = {"report_digest": "d", "venvs": {}, "binaries": {}, "nvidia": {}}
    assert er.compact_digest(rep)["build"] is None
    rep["build"] = "not-a-dict"
    assert er.compact_digest(rep)["build"] is None


def test_build_is_digested_but_its_timestamp_is_not():
    assert "build" in er.DIGEST_FIELDS
    base = {k: None for k in er.DIGEST_FIELDS}
    a = dict(base, build={"identity": {"sha": "1"}, "generated_at": "2026-01-01T00:00:00Z"})
    b = dict(base, build={"identity": {"sha": "1"}, "generated_at": "2026-01-02T00:00:00Z"})
    c = dict(base, build={"identity": {"sha": "2"}, "generated_at": "2026-01-01T00:00:00Z"})
    assert er.report_digest(a) == er.report_digest(b), "same build, later timestamp: same digest"
    assert er.report_digest(a) != er.report_digest(c), "a different sha moves the digest"
    assert er.report_digest(base) != er.report_digest(a), "gaining a build moves it too"


def test_build_probe_is_guarded_when_platform_predates_buildinfo(monkeypatch):
    real_import = builtins.__import__

    def no_buildinfo(name, *args, **kwargs):
        if name == "hugpy_platform.buildinfo":
            raise ImportError("older hugpy_platform")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_buildinfo)
    assert er.build_info_here() is None

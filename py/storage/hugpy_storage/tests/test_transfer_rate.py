"""Measured central->worker transfer rate (heartbeat ``load_bytes_per_s``).

The benchmark's cold-load budget is 2 x bytes / rate + 30 s when the worker
reports a rate, else a fixed 300 s cap (fleet_grading._cold_budget). These pin
the EMA the rate comes from and that a completed central provision feeds it.
"""
from __future__ import annotations

import types

import pytest

from hugpy_storage import provision as prov

MB = 1 << 20


@pytest.fixture(autouse=True)
def fresh_rate(monkeypatch):
    monkeypatch.setattr(prov, "_RATE", {"bps": None, "samples": 0, "last_bps": None, "last_at": None})


def test_no_sample_no_rate():
    assert prov.transfer_rate() is None


def test_tiny_or_instant_provisions_are_not_samples():
    assert prov.record_transfer(10 * MB, 5) is None          # moved ~nothing (files on disk)
    assert prov.record_transfer(500 * MB, 0.2) is None       # too short to time
    assert prov.record_transfer(None, None) is None
    assert prov.transfer_rate() is None


def test_ema_over_completed_provisions():
    assert prov.record_transfer(1000 * MB, 10) == pytest.approx(100 * MB)   # first sample = itself
    # second: 0.3 * 200 MB/s + 0.7 * 100 MB/s = 130 MB/s
    assert prov.record_transfer(2000 * MB, 10) == pytest.approx(130 * MB)
    assert prov.transfer_rate() == pytest.approx(130 * MB, rel=1e-6)
    st = prov.transfer_rate_stats()
    assert st["samples"] == 2 and st["last_bps"] == pytest.approx(200 * MB)


def test_central_provision_records_rate_and_bytes(monkeypatch):
    events = []

    class _Ev:
        def emit_provision_start(self, *a, **k):
            events.append(("start", a, k))

        def emit_provision_done(self, *a, **k):
            events.append(("done", a, k))

    clock = {"now": 100.0}
    monkeypatch.setattr(prov, "time", types.SimpleNamespace(time=lambda: clock["now"]))

    def _fetch(url, key, progress=None):
        progress(400 * MB, 1000 * MB, "files")
        progress(1000 * MB, 1000 * MB, "files")
        clock["now"] += 10.0
        return True

    seen = []
    monkeypatch.setattr(prov, "fetch_from_central", _fetch)
    monkeypatch.setattr(prov, "publish_catalog_changed", lambda *a, **k: None)
    ok = prov._provision_sources("M", "http://central", lambda d, t, n=None: seen.append((d, t, n)),
                                 _Ev(), "/dest")
    assert ok is True
    assert prov.transfer_rate() == pytest.approx(100 * MB)             # 1000 MB / 10 s
    done = [e for e in events if e[0] == "done"][0]
    assert done[2]["bytes_"] == 1000 * MB and done[2]["duration_ms"] == 10000
    # the caller's progress callback still sees every call, source marker included
    assert seen[0] == (0, 0, "source=central") and seen[-1] == (1000 * MB, 1000 * MB, "files")


def test_provision_without_caller_progress_still_measures(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(prov, "time", types.SimpleNamespace(time=lambda: clock["now"]))

    def _fetch(url, key, progress=None):
        progress(800 * MB, 800 * MB, "files")
        clock["now"] += 4.0
        return True

    monkeypatch.setattr(prov, "fetch_from_central", _fetch)
    monkeypatch.setattr(prov, "publish_catalog_changed", lambda *a, **k: None)
    assert prov._provision_sources("M", "http://central", None, None, None) is True
    assert prov.transfer_rate() == pytest.approx(200 * MB)

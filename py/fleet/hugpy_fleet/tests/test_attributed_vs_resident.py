"""Attributed-vs-resident split in the /llm/workers storage payload (2026-07-17).

The operator scare: assignment/pin ATTRIBUTES a model to a worker without
downloading it (lazy download, 7f0e6e8/2a3baeb). The fleet gauge read
``cache_used/budget`` and an over-subscribed ATTRIBUTION set made a box with
NOTHING transferring look like a runaway download storm. This locks the payload
shape so the two figures are distinct and the disk-pressure gauge is derived
from RESIDENT bytes only — attribution can never masquerade as disk pressure.

Runs under pytest: venv/bin/python -m pytest tests/test_attributed_vs_resident.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib  # noqa: E402

W = importlib.import_module(
    "hugpy_fleet.central.workers")

GiB = 1 << 30


def _worker(**over):
    w = {
        "id": "w", "name": "w", "url": "http://w",
        "last_seen": time.time(),
        "disk": {"free_bytes": 100 * GiB, "total_bytes": 500 * GiB},
        "model_last_picked": {},
        "loaded_models": [], "loading": [], "provisioning": [],
        # ATTRIBUTED set (operator designations) — three models assigned.
        "models": ["assigned-a", "assigned-b", "assigned-c"],
        "config": {}, "limits": {},
        # RESIDENT survey — only ONE of them is actually on disk yet (lazy).
        "storage": {
            "cache_used_bytes": 5 * GiB,
            "disk_free": 100 * GiB,
            "models": [
                {"model_key": "assigned-a", "bytes": 5 * GiB, "assigned": True},
            ],
        },
    }
    w.update(over)
    return w


def test_payload_has_both_blocks():
    s = W.storage_proposal(_worker())
    # attributed
    for k in ("attributed_total_bytes", "attributed_count",
              "attributed_unknown_count", "attributed_over_budget_bytes"):
        assert k in s, k
    # resident
    for k in ("hot_bytes", "hot_model_bytes", "hot_source"):
        assert k in s, k
    # gauge
    for k in ("gauge_used_bytes", "gauge_budget_bytes", "gauge_basis",
              "gauge_over_budget"):
        assert k in s, k
    # legacy keys still present (back-compat with an older UI)
    assert "cache_used_bytes" in s and "allocated_total_bytes" in s


def test_gauge_uses_resident_not_attributed():
    # cache_used (resident) is 5 GiB; the attributed set is 3 models but only one
    # is resident, so the gauge must read the resident 5 GiB, NOT the assigned set.
    s = W.storage_proposal(_worker())
    assert s["gauge_used_bytes"] == 5 * GiB
    assert s["hot_bytes"] == 5 * GiB
    assert s["gauge_basis"] == "hot"
    assert s["hot_source"] == "measured"   # cache_used_bytes present
    assert s["attributed_count"] == 3


def test_attribution_alone_never_reads_as_disk_pressure():
    # Attribute a HUGE set but keep the disk nearly empty: an over-subscribed
    # ATTRIBUTION must not flip the disk-pressure gauge on. Over-subscription is
    # surfaced structurally (attributed_over_budget_bytes), separately.
    w = _worker(
        # 100 allocated - 50 reserve (carved out, 2026-08-22) = 50 GiB budget
        limits={"disk_cache_gib": 100},
        disk={"free_bytes": 400 * GiB, "total_bytes": 500 * GiB},
    )
    w["storage"]["cache_used_bytes"] = 2 * GiB      # almost nothing on disk
    w["storage"]["disk_free"] = 400 * GiB
    s = W.storage_proposal(w)
    # gauge (disk pressure) is calm — resident 2 GiB under the 50 GiB cap.
    assert s["gauge_used_bytes"] == 2 * GiB
    assert s["gauge_over_budget"] is False
    assert s["over_budget"] is False
    # hot_source falls back correctly if cache_used is absent.
    del w["storage"]["cache_used_bytes"]
    s2 = W.storage_proposal(w)
    assert s2["hot_source"] == "summed"
    # summed resident = on-disk model rows only (the single 5 GiB row)
    assert s2["hot_bytes"] == s2["hot_model_bytes"]


def test_pre_feature_worker_degrades():
    # No storage survey at all (old agent) -> storage_proposal returns the
    # monitoring-only shape; resident is unknown, gauge budget None, no crash.
    w = _worker()
    del w["storage"]
    s = W.storage_proposal(w)
    assert s["reported"] is False
    assert s["hot_source"] == "unknown"
    assert s["hot_bytes"] is None
    # orphaned fields present even in the degraded shape (zeros = feature-off).
    assert s["orphaned_bytes"] == 0 and s["orphaned_count"] == 0
    assert s["orphaned_items"] == []


def test_orphaned_residue_passes_through():
    # The worker reports orphaned (unattributed-on-disk) residue — a stalled .part
    # set + a leftover dir. Central must surface it VERBATIM as a THIRD class,
    # distinct from attributed and resident-attributed, so drive junk is visible.
    w = _worker()
    w["storage"]["orphaned_bytes"] = 6 * GiB
    w["storage"]["orphaned_count"] = 2
    w["storage"]["orphaned_items"] = [
        {"path": "transformers/Qwen/Qwen2.5-VL-3B", "bytes": 5_700_000_000,
         "kind": "partial"},
        {"path": "transformers/old/leftover", "bytes": 300_000_000,
         "kind": "stale-dir"},
    ]
    s = W.storage_proposal(w)
    assert s["orphaned_bytes"] == 6 * GiB
    assert s["orphaned_count"] == 2
    assert len(s["orphaned_items"]) == 2
    assert s["orphaned_items"][0]["kind"] == "partial"
    # orphaned is NEITHER the gauge (resident) NOR the attributed set.
    assert s["gauge_used_bytes"] == 5 * GiB           # resident, unchanged
    assert s["attributed_count"] == 3                 # assignment set, unchanged

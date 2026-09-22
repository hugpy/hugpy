"""Wire DTOs round-trip payloads without dropping unknown fields."""

from __future__ import annotations

import hugpy_fleet
from hugpy_fleet import wire


def test_enrollment_round_trip_keeps_extra_and_omits_none():
    payload = {"name": "box", "url": "http://box:9100", "token": "t", "gpus": [{"name": "rtx"}],
               "future_field": {"x": 1}}
    dto = wire.EnrollmentRequest.from_payload(payload)
    assert dto.name == "box" and dto.token == "t" and dto.gpus == [{"name": "rtx"}]
    assert dto.extra == {"future_field": {"x": 1}}
    out = dto.to_payload()
    assert out["future_field"] == {"x": 1}
    assert "pool" not in out            # None omitted: an older peer never sees it
    assert out["role"] == "worker"


def test_heartbeat_fields_match_central_signature():
    import inspect
    from dataclasses import fields
    from hugpy_fleet.central.workers import WorkerStore
    sig = inspect.signature(WorkerStore.heartbeat)
    central = {n for n in sig.parameters if n not in ("self", "worker_id")}
    dto = {f.name for f in fields(wire.WorkerHeartbeat)} - {"extra"}
    assert central <= dto, sorted(central - dto)


def test_operation_path_and_verbs():
    op = wire.WorkerOperation(verb="evict", model_key="m")
    assert op.path() == "/ops/evict"
    assert "evict" in wire.OPERATION_VERBS and "restart" in wire.OPERATION_VERBS


def test_package_exports_dtos_lightly():
    assert wire.WorkerHeartbeat is hugpy_fleet.WorkerHeartbeat
    for name in hugpy_fleet.__all__:
        assert getattr(hugpy_fleet, name) is not None
    assert callable(hugpy_fleet.install) and callable(hugpy_fleet.uninstall)

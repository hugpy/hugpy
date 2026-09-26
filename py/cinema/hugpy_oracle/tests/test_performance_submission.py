"""The HTTP enqueue path must reject bad nested contracts before reserving a job."""

import pytest

from hugpy_oracle.relay.performance_relay import (
    PerformanceSpecError,
    probe,
    validate_performance_spec,
)


def _body():
    return {
        "goal": {"objective": "A scene", "raw_prompt": "A scene"},
        "dialogue": {"locked": True, "lines": [
            {"line_id": "line-1", "speaker": "narrator", "text": "Hello"}]},
        "casting": [["narrator", {"voice_id": "narrator", "kind": "synthetic"}]],
        "raw_request_ref": "test:raw-request",
        "stop_after": "segments",
    }


def test_accepts_constructible_staged_performance():
    spec = validate_performance_spec(_body())
    assert spec.stop_after == "segments"
    assert spec.casting[0][0] == "narrator"


@pytest.mark.parametrize("change", [
    {"stop_after": "unknown"},
    {"dialogue": {"locked": True, "lines": [{"line_id": "1", "speaker": "missing", "text": "Hi"}]}},
    {"casting": [["narrator", {"voice_id": "narrator", "kind": "reference"}]]},
])
def test_refuses_unconstructible_performance(change):
    body = _body()
    body.update(change)
    with pytest.raises((PerformanceSpecError, ValueError)):
        validate_performance_spec(body)


def test_probe_never_claims_full_readiness_without_clip_judge():
    status = probe()
    if any(gap["seam"] == "judge_clip" for gap in status.get("unbound", [])):
        assert status["ready"] is False

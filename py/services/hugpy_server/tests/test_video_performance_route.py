"""The Oracle performance adapter validates before it touches the media bus."""

from flask import Flask

from hugpy_server.app.routes import video_routes


def _body():
    return {
        "goal": {"objective": "A scene", "raw_prompt": "A scene"},
        "dialogue": {"locked": True, "lines": [
            {"line_id": "one", "speaker": "narrator", "text": "Hello"}]},
        "casting": [["narrator", {"voice_id": "narrator", "kind": "synthetic"}]],
        "raw_request_ref": "test:raw",
        "stop_after": "segments",
        "private": True,
    }


def test_valid_submission_and_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(video_routes, "_video_enqueue",
                        lambda name, spec, private=None: calls.append((name, spec, private)) or "queued")
    app = Flask(__name__)
    app.register_blueprint(video_routes.video_bp)
    with app.test_client() as client:
        result = client.post("/video/jobs/performance", json=_body())
        assert result.status_code == 200 and result.json == {"job_id": "queued"}
        probe = client.get("/video/performance/probe")
        assert probe.status_code == 200 and "unbound" in probe.json
        bad = _body()
        bad["stop_after"] = "unknown"
        refused = client.post("/video/jobs/performance", json=bad)
        assert refused.status_code == 400
    assert len(calls) == 1
    assert calls[0][0] == "video_performance"
    assert calls[0][1].stop_after == "segments"
    assert calls[0][2] is True

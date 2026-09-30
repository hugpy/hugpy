"""test_fire_routes fixes (2026-09-29):
  1. VL rows that also carry text-generation fire; only-non-text rows skip.
  2. the "no task" skip reason is reachable (operator-precedence fix).
  3. GET .../test-fire/current returns the worker's latest job, not a 404.
"""
from __future__ import annotations

import importlib
import time

from flask import Flask

tf = importlib.import_module("hugpy_server.app.routes.test_fire_routes")

CATALOG = {
    "Qwen2.5-VL-3B": {"model_key": "Qwen2.5-VL-3B", "framework": "gguf",
                      "primary_task": "image-text-to-text",
                      "tasks": ["image-text-to-text", "text-generation"]},
    "emb-only": {"model_key": "emb-only", "primary_task": "feature-extraction",
                 "tasks": ["feature-extraction", "sentence-similarity"]},
    "img-only": {"model_key": "img-only", "primary_task": "image-text-to-text",
                 "tasks": ["image-text-to-text"]},
    "no-task": {"model_key": "no-task", "tasks": []},
}
WORKER = {"id": "w1", "name": "box",
          "models": ["Qwen2.5-VL-3B", "emb-only", "img-only", "no-task"]}


def test_vl_text_gen_fires_non_text_skips_and_no_task_reason():
    selected, skipped = tf.select_text_gen_models(WORKER, CATALOG)
    assert [m["model_key"] for m in selected] == ["Qwen2.5-VL-3B"]
    assert selected[0]["tasks"] == ["text-generation"]
    reasons = {s["model_key"]: s["reason"] for s in skipped}
    assert reasons["emb-only"].startswith("not text-gen: feature-extraction")
    assert reasons["img-only"] == "not text-gen: image-text-to-text"
    assert reasons["no-task"] == "not text-gen: no task"


def test_get_current_returns_latest_job(monkeypatch):
    tf.reset_registry()
    try:
        app = Flask(__name__)
        app.register_blueprint(tf.test_fire_bp)
        client = app.test_client()
        assert client.get("/llm/workers/w1/test-fire/current").status_code == 404
        calls = lambda key, prompt, max_tokens, job: {"ok": True, "tokens": 1}
        first = tf.start_job("w1", "box", ["Qwen2.5-VL-3B"], call_fn=calls)
        first.thread.join(5)
        second = tf.start_job("w1", "box", ["Qwen2.5-VL-3B"], call_fn=calls)
        second.thread.join(5)
        r = client.get("/llm/workers/w1/test-fire/current")
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["job_id"] == second.job_id
        assert client.get(f"/llm/workers/w1/test-fire/{first.job_id}").status_code == 200
        assert client.get("/llm/workers/other/test-fire/current").status_code == 404
    finally:
        tf.reset_registry()

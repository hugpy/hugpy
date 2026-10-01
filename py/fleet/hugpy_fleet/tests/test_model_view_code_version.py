"""The persisted model-view cache must not outlive the code that derived it:
a landing that changes planned_split kept serving the old figures for up to an
hour (2026-10-01, Coder-Next planned 16.48 GiB after the planned_need landing)."""
import importlib

W = importlib.import_module("hugpy_fleet.central.workers")


def test_signature_changes_with_code_version(monkeypatch):
    worker = {"models": ["m"], "spill_by_model": {}, "gpus": []}
    a = W._model_view_signature(worker)
    monkeypatch.setattr(W, "_MODEL_VIEW_VERSION", "next")
    assert W._model_view_signature(worker) != a


def test_stale_code_cache_is_not_served(monkeypatch):
    worker = {"models": [], "gpus": []}
    W._refresh_model_view(worker)
    assert W._refresh_model_view(worker) is False
    monkeypatch.setattr(W, "_MODEL_VIEW_VERSION", "next")
    assert W._refresh_model_view(worker) is True

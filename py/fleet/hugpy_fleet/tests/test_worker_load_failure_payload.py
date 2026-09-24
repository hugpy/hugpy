"""The worker's error payloads carry the structured load_failure (2026-09-23):
/infer + /infer/stream error bodies and the /probe load report central stores
as load_reports[key]. Full error text, never truncated below the cause."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

agent = importlib.import_module("hugpy_fleet.worker.agent")
lf = importlib.import_module("hugpy_engine.serve.load_failure")

STDERR = "E llama_model_load: error loading model: check_tensor_dims: " + "x" * 2500


def test_hard_failure_payload_is_structured_and_whole():
    exc = lf.HardLoadFailure("Echo-Mini: hard_load_failure — " + STDERR,
                             loader_stderr=STDERR, path="/m/Echo-Mini.gguf")
    body = agent._with_load_failure({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, exc)
    assert body["load_failure"]["class"] == "hard_load_failure"
    assert body["load_failure"]["path"] == "/m/Echo-Mini.gguf"
    assert body["load_failure"]["loader_stderr"] == STDERR      # whole, never cut
    assert STDERR in body["error"]                     # error text untruncated


def test_generation_error_gets_no_load_failure_but_probe_classifies():
    exc = RuntimeError("stream reset by peer")
    assert "load_failure" not in agent._with_load_failure({}, exc)
    assert agent._with_load_failure({}, exc, classify=True)["load_failure"]["class"] == "other"
    chained = RuntimeError("wrapper")
    chained.__cause__ = lf.ModelLoadFailure("needs ~42 GB VRAM", load_class="vram_fit")
    assert agent._with_load_failure({}, chained)["load_failure"]["class"] == "vram_fit"

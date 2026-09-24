"""Central never offers a projector-only "model", and an explicit worker pin is
still admission-gated (2026-09-23 computron incident).

zerodigest/Qwen3.8-27B-Uncensored-YMQ-MTP-GGUF on central held only
``mmproj/Qwen3.8-27B-Uncensored-vision-{Q4_K_S,Q6_K}.gguf``. The transfer
election treated those (no "mmproj" in the basename) as the QUANT, /manifest
offered them, a worker pinned by ``alloc.worker`` pulled them, called the model
present, and its runner then fetched 134 GB from Hugging Face. Now:

  * the election recognises projectors by header / ``mmproj/`` dir;
  * /manifest refuses a weightless set with 409 ``state=no_weights``;
  * the explicit-pin gate refuses when central holds no weights or the model
    does not fit the named worker (normal placement's static feasibility), and
    exempts a box that already holds the model.
"""
import importlib
import os
import struct
import tempfile

import pytest
from flask import Flask

wr = importlib.import_module("hugpy_server.app.routes.worker_routes")

MIB = 1024 * 1024
GB = 2 ** 30


def _gguf(path, arch, pad=2 * MIB):
    key, val = b"general.architecture", arch.encode()
    hdr = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
           + struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
           + struct.pack("<Q", len(val)) + val)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(hdr + b"\0" * pad)


@pytest.fixture()
def projector_only_dir():
    d = tempfile.mkdtemp(prefix="hugpy-proj-only-")
    _gguf(os.path.join(d, "mmproj", "Qwen3.8-27B-Uncensored-vision-Q4_K_S.gguf"), "clip")
    _gguf(os.path.join(d, "mmproj", "Qwen3.8-27B-Uncensored-vision-Q6_K.gguf"), "clip")
    return d


ROW = {"key": "Qwen3.8-27B-Uncensored-YMQ-MTP-GGUF",
       "hub_id": "zerodigest/Qwen3.8-27B-Uncensored-YMQ-MTP-GGUF",
       "name": "Qwen3.8-27B-Uncensored-YMQ-MTP-GGUF", "framework": "gguf",
       "filename": "mmproj/Qwen3.8-27B-Uncensored-vision-Q6_K.gguf"}
KEY = ROW["key"]


@pytest.fixture()
def registry(monkeypatch):
    def _install(dest):
        monkeypatch.setattr(wr, "get_models_dict",
                            lambda dict_return=True: {KEY: dict(ROW)}, raising=False)
        monkeypatch.setattr(wr, "route_destination", lambda model: dest, raising=False)
    return _install


# ── election: projectors in mmproj/ are not quants ─────────────────────────
def test_mmproj_subdir_files_are_projectors_not_quants():
    entries = [("mmproj/x-vision-Q6_K.gguf", int(0.6 * GB)),
               ("x-YMQ-L-TI.gguf", int(14 * GB)),
               ("README.md", 4000)]
    names = [r for r, _s in wr._elect_gguf_transfer_set(entries, {})]
    assert "x-YMQ-L-TI.gguf" in names            # the real quant is elected
    assert "mmproj/x-vision-Q6_K.gguf" in names  # projector rides along


def test_header_decides_projector_when_root_given(tmp_path):
    _gguf(str(tmp_path / "vision-Q6_K.gguf"), "clip")
    assert wr._is_projector_gguf("vision-Q6_K.gguf", str(tmp_path)) is True
    _gguf(str(tmp_path / "model-Q4_K_M.gguf"), "llama")
    assert wr._is_projector_gguf("model-Q4_K_M.gguf", str(tmp_path)) is False


def test_projector_only_set_has_no_weights(projector_only_dir):
    sel = [("mmproj/Qwen3.8-27B-Uncensored-vision-Q4_K_S.gguf", 2 * MIB + 100),
           ("mmproj/Qwen3.8-27B-Uncensored-vision-Q6_K.gguf", 2 * MIB + 100)]
    assert wr._transfer_weight_files(sel, root=projector_only_dir) == []


# ── /manifest refuses a weightless model ────────────────────────────────────
def test_manifest_409_no_weights(monkeypatch, registry, projector_only_dir):
    registry(projector_only_dir)
    monkeypatch.setattr(wr, "_transfer_authorized", lambda: True)
    app = Flask(__name__)
    app.register_blueprint(wr.worker_bp)
    r = app.test_client().get(f"/llm/models/{KEY}/manifest")
    assert r.status_code == 409
    body = r.get_json()
    assert body["state"] == "no_weights"
    assert f"central holds no weights for {KEY}" in body["reason"]
    assert "mmproj/Qwen3.8-27B-Uncensored-vision-Q6_K.gguf" in body["reason"]


# ── explicit pin gate ───────────────────────────────────────────────────────
def test_pin_refused_when_central_lacks_weights(registry, projector_only_dir, monkeypatch):
    registry(projector_only_dir)
    monkeypatch.setattr(wr, "get_model_config", lambda k: (_ for _ in ()).throw(KeyError(k)))
    reason = wr._explicit_pin_refusal({"id": "w1", "name": "computron"}, KEY)
    assert reason and f"central holds no weights for {KEY}" in reason


def test_pin_refused_when_central_has_no_dir(registry, tmp_path, monkeypatch):
    registry(str(tmp_path / "absent"))
    monkeypatch.setattr(wr, "get_model_config", lambda k: (_ for _ in ()).throw(KeyError(k)))
    reason = wr._explicit_pin_refusal({"id": "w1", "name": "computron"}, KEY)
    assert "no model directory in central's llm_storage" in reason


def _weights_dir():
    d = tempfile.mkdtemp(prefix="hugpy-weights-")
    _gguf(os.path.join(d, "x-YMQ-L-TI.gguf"), "qwen35")
    _gguf(os.path.join(d, "mmproj", "x-vision-Q6_K.gguf"), "clip")
    return d


def test_pin_refused_when_model_does_not_fit_worker(registry, monkeypatch):
    registry(_weights_dir())
    monkeypatch.setattr(wr, "get_model_config", lambda k: (_ for _ in ()).throw(KeyError(k)))
    W = importlib.import_module("hugpy_fleet.central.workers")
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: False)
    monkeypatch.setattr(W, "feasibility_context", lambda wid, mk: {
        "model_bytes": int(14.2 * GB), "gpu_total_bytes": 8 * GB,
        "ram_total_bytes": 4 * GB})
    reason = wr._explicit_pin_refusal({"id": "w1", "name": "computron"}, KEY)
    assert reason and "does not fit computron" in reason and "14.2 GiB" in reason


def test_pin_admitted_when_weights_present_and_fit(registry, monkeypatch):
    registry(_weights_dir())
    monkeypatch.setattr(wr, "get_model_config", lambda k: (_ for _ in ()).throw(KeyError(k)))
    W = importlib.import_module("hugpy_fleet.central.workers")
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: True)
    assert wr._explicit_pin_refusal({"id": "w1", "name": "computron"}, KEY) is None
    # Unknown fit (None) is no opinion — never a refusal.
    monkeypatch.setattr(W, "worker_can_hold", lambda w, mk: None)
    assert wr._explicit_pin_refusal({"id": "w1", "name": "computron"}, KEY) is None


def test_pin_exempt_when_worker_already_holds_model(registry, projector_only_dir, monkeypatch):
    registry(projector_only_dir)
    monkeypatch.setattr(wr, "get_model_config", lambda k: (_ for _ in ()).throw(KeyError(k)))
    w = {"id": "w1", "name": "computron", "loaded_models": [KEY]}
    assert wr._explicit_pin_refusal(w, KEY) is None


def test_gate_is_registered_into_the_engine_seam():
    remote = importlib.import_module("hugpy_engine.resolvers.remote")
    assert remote._worker_pin_gate is wr._explicit_pin_refusal

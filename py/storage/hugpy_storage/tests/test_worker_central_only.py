"""Workers take weights from CENTRAL only; "present" means WEIGHTS (2026-09-23).

computron incident: central's copy of zerodigest/Qwen3.8-27B-Uncensored-YMQ-MTP-GGUF
held only its two ``mmproj/*.gguf`` vision projectors. The worker's provision
pulled them, logged "already complete on disk (2 files)" / "provisioned from
CENTRAL", and the runner's direct ``download_models.ensure_model`` then
snapshot-downloaded the whole 134 GB repo from Hugging Face. These tests pin:

  * on a worker (WORKER_CENTRAL_URL set) the serve-path resolver never reaches
    Hugging Face and raises a precise "central holds no weights" error;
  * a projector-only file set / directory is not a model — neither in central's
    manifest nor on the worker's disk;
  * off a worker (central / standalone) the resolver is unchanged.
"""
from __future__ import annotations

import io
import json
import struct
import urllib.error

import pytest

from hugpy_storage import download_models as dm
from hugpy_storage import provision as prov
from hugpy_storage.model_presence import model_looks_downloaded

MIB = 1024 * 1024


def _gguf(path, arch: str, pad: int = 2 * MIB) -> str:
    """A minimal GGUF v3 file whose header says ``general.architecture=arch``."""
    key = b"general.architecture"
    val = arch.encode()
    hdr = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
           + struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
           + struct.pack("<Q", len(val)) + val)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hdr + b"\0" * pad)
    return str(path)


@pytest.fixture
def no_hf(monkeypatch):
    """Any reach for Hugging Face fails the test."""
    calls = []

    def _boom(*a, **k):
        calls.append(a)
        raise AssertionError("Hugging Face reached on a worker serve path")

    monkeypatch.setattr(dm, "_hf", _boom)
    return calls


# ── worker identity ──────────────────────────────────────────────────────────
def test_worker_marker_is_worker_central_url(monkeypatch):
    monkeypatch.delenv("WORKER_CENTRAL_URL", raising=False)
    assert prov.worker_central_url() is None and not prov.is_worker_process()
    monkeypatch.setenv("WORKER_CENTRAL_URL", "http://192.168.1.100:7002/")
    assert prov.worker_central_url() == "http://192.168.1.100:7002"
    assert prov.is_worker_process()


# ── serve path on a worker: never HF ─────────────────────────────────────────
def test_worker_ensure_model_never_calls_hf_and_names_the_reason(monkeypatch, no_hf):
    monkeypatch.setenv("WORKER_CENTRAL_URL", "http://central")
    seen = {}

    def _present(mk, url, **kw):
        seen["args"] = (mk, url, kw.get("purpose"))
        prov._record_failure(mk, "central-transfer",
                             "central holds no weights for Qwen3.8 (mmproj/x-Q6_K.gguf)")
        return False

    monkeypatch.setattr(prov, "ensure_model_present", _present)
    monkeypatch.setattr(prov, "catalog_get",
                        lambda k: {"filename": "mmproj/x-Q6_K.gguf"})
    monkeypatch.setattr(prov, "model_is_local", lambda k: False)
    # The legacy entry point every runner used to call: on a worker it must
    # route to central-only provisioning, not snapshot_download.
    with pytest.raises(prov.CentralHoldsNoWeights) as ei:
        dm.ensure_model("Qwen3.8")
    msg = str(ei.value)
    assert "central holds no weights for Qwen3.8 (mmproj/x-Q6_K.gguf)" in msg
    assert "Hugging Face" in msg          # says it will NOT go there
    assert seen["args"] == ("Qwen3.8", "http://central", "demand")
    assert no_hf == []


def test_worker_serving_weights_returns_the_local_dir(monkeypatch, no_hf, tmp_path):
    monkeypatch.setenv("WORKER_CENTRAL_URL", "http://central")
    monkeypatch.setattr(prov, "ensure_model_present", lambda *a, **k: True)
    monkeypatch.setattr(prov, "catalog_get", lambda k: None)
    monkeypatch.setattr(prov, "model_is_local", lambda k: True)
    monkeypatch.setattr(prov, "_model_dir", lambda k, cfg=None: str(tmp_path))
    assert prov.ensure_serving_weights("m") == str(tmp_path)
    assert no_hf == []


def test_off_worker_serving_weights_is_download_models_unchanged(monkeypatch):
    monkeypatch.delenv("WORKER_CENTRAL_URL", raising=False)
    monkeypatch.setattr(dm, "ensure_model", lambda key, root=None: f"/central/{key}")
    monkeypatch.setattr(prov, "ensure_model_present",
                        lambda *a, **k: pytest.fail("central must not self-provision"))
    assert prov.ensure_serving_weights("m") == "/central/m"


def test_provision_has_no_hf_path_left():
    assert not hasattr(prov, "fetch_from_hf")
    assert not hasattr(prov, "_hf_fallback_always")


# ── "present" means weights ──────────────────────────────────────────────────
def test_projector_only_dir_is_not_present(tmp_path):
    d = tmp_path / "Qwen3.8-27B-Uncensored-YMQ-MTP-GGUF"
    _gguf(d / "mmproj" / "Qwen3.8-27B-Uncensored-vision-Q4_K_S.gguf", "clip")
    _gguf(d / "mmproj" / "Qwen3.8-27B-Uncensored-vision-Q6_K.gguf", "clip")
    cfg = {"framework": "gguf",
           "filename": "mmproj/Qwen3.8-27B-Uncensored-vision-Q6_K.gguf"}
    assert model_looks_downloaded(str(d), cfg) is False
    files = [{"path": "mmproj/Qwen3.8-27B-Uncensored-vision-Q4_K_S.gguf"},
             {"path": "mmproj/Qwen3.8-27B-Uncensored-vision-Q6_K.gguf"}]
    assert prov._weight_files(files, str(d)) == []
    # A real quant beside them IS weights.
    _gguf(d / "Qwen3.8-27B-Uncensored-YMQ-L-TI.gguf", "qwen35")
    files.append({"path": "Qwen3.8-27B-Uncensored-YMQ-L-TI.gguf"})
    assert prov._weight_files(files, str(d)) == ["Qwen3.8-27B-Uncensored-YMQ-L-TI.gguf"]


def test_projector_header_beats_an_innocent_filename(tmp_path):
    # No "mmproj" anywhere in the path — only the header says clip.
    _gguf(tmp_path / "vision-Q6_K.gguf", "clip")
    assert prov._weight_files([{"path": "vision-Q6_K.gguf"}], str(tmp_path)) == []


def test_manifest_weights_judged_by_name_and_size():
    proj = [{"path": "mmproj/x-vision-Q6_K.gguf", "size": 600 * MIB},
            {"path": "README.md", "size": 4000}]
    assert prov._weight_files(proj) == []
    assert prov._weight_files([{"path": "model.safetensors", "size": 5 * MIB}]) \
        == ["model.safetensors"]
    assert prov._weight_files([{"path": "big.weights", "size": 900 * MIB}]) \
        == ["big.weights"]
    assert prov._weight_files([]) == []


def _stub_manifest(monkeypatch, tmp_path, files):
    monkeypatch.setattr(prov, "_get_json", lambda url, timeout=30.0: {
        "files": files, "filename": "mmproj/x-vision-Q6_K.gguf",
        "total_bytes": sum(f["size"] for f in files), "hub_id": "org/m"})
    monkeypatch.setattr(prov, "_local_destination", lambda meta: str(tmp_path))
    monkeypatch.setattr(prov.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("no byte may be transferred"))


def test_fetch_from_central_refuses_projector_only_manifest(monkeypatch, tmp_path):
    _stub_manifest(monkeypatch, tmp_path,
                   [{"path": "mmproj/x-vision-Q4_K_S.gguf", "size": 500 * MIB},
                    {"path": "mmproj/x-vision-Q6_K.gguf", "size": 600 * MIB}])
    with pytest.raises(prov.CentralHoldsNoWeights, match="central holds no weights for m"):
        prov.fetch_from_central("http://c", "m")
    with pytest.raises(prov.CentralHoldsNoWeights):
        prov.fetch_archive_from_central("http://c", "m")


def test_already_complete_projector_only_is_not_success(monkeypatch, tmp_path):
    # Central's manifest names a "weight" file that is really a projector on
    # disk (header clip): the 'nothing pending' short-circuit must not report
    # the model provisioned.
    p = _gguf(tmp_path / "x-Q6_K.gguf", "clip", pad=2 * MIB)
    size = len(open(p, "rb").read())
    _stub_manifest(monkeypatch, tmp_path, [{"path": "x-Q6_K.gguf", "size": size}])
    with pytest.raises(prov.CentralHoldsNoWeights, match="not a model"):
        prov.fetch_from_central("http://c", "m")


def test_central_409_no_weights_surfaces_the_reason():
    body = json.dumps({"state": "no_weights",
                       "reason": "central holds no weights for m (q)"}).encode()
    exc = urllib.error.HTTPError("http://c/manifest", 409, "Conflict", {},
                                 io.BytesIO(body))
    with pytest.raises(prov.CentralHoldsNoWeights, match=r"central holds no weights for m \(q\)"):
        prov._raise_if_no_weights(exc)
    plain = urllib.error.HTTPError("http://c/manifest", 409, "Conflict", {},
                                   io.BytesIO(b'{"error": "budget"}'))
    prov._raise_if_no_weights(plain)          # an ordinary 409 stays a plain verdict


def test_provision_refuses_without_central(monkeypatch, no_hf):
    monkeypatch.setattr(prov, "_evt", lambda: None)
    assert prov._provision_now("m", None) is False
    assert "no central URL" in (prov.last_failure("m") or {}).get("reason", "")
    assert no_hf == []

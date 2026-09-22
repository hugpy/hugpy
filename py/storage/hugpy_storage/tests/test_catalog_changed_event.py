"""``catalog.changed`` on the control bus after the inventory moves."""
import os

import pytest

from hugpy_control.bus import TOPIC_CATALOG_CHANGED, Bus
from hugpy_storage import events


def test_topic_constant_is_agreed_name():
    assert TOPIC_CATALOG_CHANGED == "catalog.changed"
    assert events.TOPIC_CATALOG_CHANGED == TOPIC_CATALOG_CHANGED


def test_publish_reaches_subscriber_with_payload():
    b = Bus()
    sub = b.subscribe(TOPIC_CATALOG_CHANGED)
    msg = events.publish_catalog_changed("download completed", model_key="m",
                                         hub_id="org/m", destination="/x",
                                         change="download", the_bus=b)
    assert msg is not None and msg.topic == TOPIC_CATALOG_CHANGED
    got = sub.get(timeout=1.0)
    assert got.source == "hugpy_storage"
    assert got.payload["model_key"] == "m"
    assert got.payload["hub_id"] == "org/m"
    assert got.payload["change"] == "download"
    assert got.payload["reason"] == "download completed"


def test_publish_never_raises_on_broken_bus():
    class Broken:
        def publish(self, *a, **k):
            raise RuntimeError("bus down")
    assert events.publish_catalog_changed("x", the_bus=Broken()) is None


def test_invalidate_model_status_cache_publishes(monkeypatch):
    from hugpy_control import bus as busmod
    from hugpy_storage.downloader import engine
    seen = []
    monkeypatch.setattr(busmod.bus, "publish",
                        lambda topic=None, **f: seen.append((topic, f)) or None)
    engine.invalidate_model_status_cache("download completed: m", model_key="m")
    assert seen and seen[-1][0] == TOPIC_CATALOG_CHANGED
    assert seen[-1][1]["payload"]["model_key"] == "m"


def test_wipe_model_publishes_wipe(tmp_path, monkeypatch):
    from hugpy_control import bus as busmod
    from hugpy_storage import provision
    target = tmp_path / "models" / "gguf" / "org" / "m"
    target.mkdir(parents=True)
    (target / "w.gguf").write_bytes(b"GGUF")
    monkeypatch.setenv("HUGPY_MODEL_STORE_REAPABLE", "1")
    monkeypatch.delenv("HUGPY_SHARED_MODEL_STORE", raising=False)
    seen = []
    monkeypatch.setattr(busmod.bus, "publish",
                        lambda topic=None, **f: seen.append((topic, f)) or None)
    assert provision.wipe_model("m", str(target)) is True
    assert not os.path.exists(target)
    assert seen and seen[-1][0] == TOPIC_CATALOG_CHANGED
    assert seen[-1][1]["payload"]["change"] == "wipe"


def test_download_one_publishes_promote(tmp_path, monkeypatch):
    """A completed download_one lands, stamps and announces itself."""
    from hugpy_control import bus as busmod
    from hugpy_storage import download_models as dm

    class _FakeHub:
        @staticmethod
        def snapshot_download(repo_id, local_dir, **kw):
            os.makedirs(local_dir, exist_ok=True)
            with open(os.path.join(local_dir, "config.json"), "w") as fh:
                fh.write("{}")
            with open(os.path.join(local_dir, "model.safetensors"), "wb") as fh:
                fh.write(b"x" * (1024 * 1024 + 1))
    monkeypatch.setattr(dm, "_hf", lambda: _FakeHub)
    seen = []
    monkeypatch.setattr(busmod.bus, "publish",
                        lambda topic=None, **f: seen.append((topic, f)) or None)
    model = {"hub_id": "org/m", "framework": "transformers",
             "primary_task": "text-generation"}
    dm.download_one(model, root=str(tmp_path), model_key="m")
    dest = os.path.join(str(tmp_path), "models", "transformers", "org", "m")
    assert os.path.isfile(os.path.join(dest, "model.safetensors"))
    assert os.path.isfile(os.path.join(dest, "hugpy.json"))
    promotes = [f for t, f in seen if t == TOPIC_CATALOG_CHANGED
                and f["payload"]["change"] == "promote"]
    assert promotes and promotes[-1]["payload"]["destination"] == dest

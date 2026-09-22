"""Provisioner over FAKE registries and a FAKE queue: ``Registries`` injects
the engine rows, the studio zoo, storage's resolver and the comfy catalog;
``provision(apply=True)`` enqueues through an injected ``enqueue_fn`` with an
injected job mirror and disk probe — no store, no comfy, no control DB."""
from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace

import pytest

from hugpy_ops import provisioner
from hugpy_ops.provisioner import Registries, Want, provision, wants


def _touch(path, size=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x" * size)


def _studio_cfg(model_id, weight_uri, source_url="https://huggingface.co/x"):
    return SimpleNamespace(model_id=model_id, weight_uri=weight_uri,
                           source_url=source_url, synthetic=False)


@pytest.fixture
def fake_registries(tmp_path):
    root = tmp_path / "store"
    weights = tmp_path / "studio-weights"
    # comfy: one present, one missing checkpoint; id_lock assets present
    _touch(str(root / "checkpoints" / "present.safetensors"), 8)
    for asset in provisioner.COMFY_SHARED_ASSETS.values():
        if asset.kind != "node_pack":
            _touch(str(root / asset.dest_dir / asset.dest_name), 8)
    # studio: one present, one absent
    _touch(str(weights / "Org" / "StudioHere" / "w.bin"), 8)
    curated = {
        "comfy-present": {"framework": "comfy", "hub_id": "Org/Present",
                          "filename": "present.safetensors"},
        "comfy-missing": {"framework": "comfy", "hub_id": "Org/Missing",
                          "filename": "missing.safetensors"},
        "llm-here": {"framework": "transformers", "hub_id": "Org/Here"},
        "llm-gone": {"framework": "gguf", "hub_id": "Org/Gone",
                     "filename": "gone.gguf"},
    }
    studio = {"studio-here": _studio_cfg("studio-here", "Org/StudioHere"),
              "studio-gone": _studio_cfg("studio-gone", "Org/StudioGone")}

    def resolver(row, root_, require_complete=True):
        return str(root / "models" / "Org" / "Here") if row.get("hub_id") == "Org/Here" else None

    return str(root), Registries(curated=curated, studio=studio, resolver=resolver,
                                 catalog=None, folders=None,
                                 studio_weights_root=str(weights))


def test_wants_scans_all_three_fake_registries(fake_registries):
    root, reg = fake_registries
    found = wants(root=root, registries=reg)
    by_name = {w.name: w for w in found}
    assert set(by_name) == {"comfy-missing", "studio-gone", "llm-gone"}
    assert by_name["comfy-missing"].registry == "comfy"
    assert by_name["studio-gone"].registry == "studio"
    assert by_name["llm-gone"].registry == "tasks"
    assert all(w.resolved for w in found)
    assert by_name["llm-gone"].filename == "gone.gguf"
    assert by_name["llm-gone"].framework == "gguf"


def test_wants_never_touches_the_live_registries(fake_registries, monkeypatch):
    root, reg = fake_registries
    for name in ("curated_rows", "studio_models", "default_resolver"):
        monkeypatch.setattr(provisioner, name,
                            lambda *a, _n=name, **k: pytest.fail(f"live {_n} read"))
    assert len(wants(root=root, registries=reg)) == 3


def test_one_failing_registry_never_hides_the_others(fake_registries):
    root, reg = fake_registries

    class ExplodingDict(dict):
        def items(self):
            raise RuntimeError("studio env absent")

    broken = Registries(curated=reg.curated, studio=ExplodingDict(),
                        resolver=reg.resolver, catalog=None, folders=None,
                        studio_weights_root=reg.studio_weights_root)
    names = {w.name for w in wants(root=root, registries=broken)}
    assert names == {"comfy-missing", "llm-gone"}


def test_provision_apply_enqueues_through_the_injected_queue(fake_registries):
    root, reg = fake_registries
    calls = []

    def fake_enqueue(model_key, model, total_bytes=None, transport="web"):
        calls.append((model_key, model, transport))
        return SimpleNamespace(id=f"job-{len(calls)}")

    out = io.StringIO()
    result = provision(apply=True, root=root, out=out, registries=reg,
                       existing_jobs=[], free_bytes=10**15,
                       enqueue_fn=fake_enqueue)
    assert result["applied"] is True
    assert sorted(r["want"] for r in result["results"]) == ["comfy-missing", "llm-gone", "studio-gone"]
    assert all(r["enqueued"] for r in result["results"])
    assert {c[2] for c in calls} == {"provisioner"}
    assert {c[1]["hub_id"] for c in calls} == {"Org/Missing", "Org/Gone", "Org/StudioGone"}
    text = out.getvalue()
    assert "3 want(s) — 3 resolved, 0 UNRESOLVED" in text
    assert "DRY RUN" not in text


def test_provision_dry_run_enqueues_nothing(fake_registries):
    root, reg = fake_registries
    out = io.StringIO()
    result = provision(apply=False, root=root, out=out, registries=reg,
                       enqueue_fn=lambda *a, **k: pytest.fail("dry run must not enqueue"))
    assert result["applied"] is False and result["results"] == []
    assert "[DRY RUN — nothing enqueued]" in out.getvalue()
    assert {w["name"] for w in result["wants"]} == {"comfy-missing", "studio-gone", "llm-gone"}


def test_provision_dedupes_within_one_pass_and_against_the_mirror(tmp_path):
    root = str(tmp_path / "store")
    # id_lock assets present so only the two registries' rows are wanted
    for asset in provisioner.COMFY_SHARED_ASSETS.values():
        if asset.kind != "node_pack":
            _touch(os.path.join(root, asset.dest_dir, asset.dest_name), 8)
    # two registries naming the SAME hub repo (whole-repo pulls, no filename):
    # only one enqueue per pass — dedupe is on (hub_id, filename)
    curated = {"a-key": {"framework": "transformers", "hub_id": "Org/Same"}}
    studio = {"studio-same": _studio_cfg("studio-same", "Org/Same")}
    reg = Registries(curated=curated, studio=studio,
                     resolver=lambda row, r, require_complete=True: None,
                     catalog=None, folders=None,
                     studio_weights_root=str(tmp_path / "w"))
    calls = []
    fake_enqueue = lambda k, m, total_bytes=None, transport="web": calls.append(k) or SimpleNamespace(id="j")
    result = provision(apply=True, root=root, out=io.StringIO(), registries=reg,
                       existing_jobs=[], free_bytes=10**15, enqueue_fn=fake_enqueue)
    reasons = {r["want"]: r.get("reason", "enqueued") for r in result["results"]}
    assert len(calls) == 1
    assert sorted(reasons.values()) == ["duplicate", "enqueued"]
    # an already-live job in the mirror blocks the whole pass for that key
    calls.clear()
    result = provision(apply=True, root=root, out=io.StringIO(), registries=reg,
                       existing_jobs=[{"model_key": "a-key", "status": "processing"},
                                      {"model_key": "studio-same", "status": "pending"}],
                       free_bytes=10**15, enqueue_fn=fake_enqueue)
    assert calls == [] and {r["reason"] for r in result["results"]} == {"duplicate"}


def test_provision_refuses_past_the_disk_floor(fake_registries):
    root, reg = fake_registries
    result = provision(apply=True, root=root, out=io.StringIO(), registries=reg,
                       existing_jobs=[], free_bytes=0,
                       enqueue_fn=lambda *a, **k: pytest.fail("floor breached"))
    assert {r["reason"] for r in result["results"]} == {"disk-floor"}


def test_main_json_uses_injected_registries_via_monkeypatch(fake_registries, monkeypatch, capsys):
    """The CLI path goes through provision() -> wants(); pinning LIVE_REGISTRIES
    to fakes proves the CLI never needs the real registries to run."""
    root, reg = fake_registries
    monkeypatch.setattr(provisioner, "LIVE_REGISTRIES", reg)
    monkeypatch.setattr(provisioner, "_default_root", lambda: root)
    assert provisioner.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is False
    assert {w["name"] for w in payload["wants"]} == {"comfy-missing", "studio-gone", "llm-gone"}


def test_live_registry_readers_are_public_names():
    """The three registry reads resolve to PUBLIC names of the owning
    packages (no underscore imports across the package boundary)."""
    import inspect
    for fn in (provisioner.curated_rows, provisioner.studio_models, provisioner.default_resolver):
        src = inspect.getsource(fn)
        for line in src.splitlines():
            line = line.strip()
            if line.startswith("from hugpy_"):
                imported = line.split("import", 1)[1]
                assert "_" not in imported.strip().split(",")[0].strip()[:1], line
    assert provisioner.Want("comfy", "n", "absent", "/d", hub_id="a/b").resolved
    assert not Want("studio", "n", "absent", "/d").resolved

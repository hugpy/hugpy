"""Provisioner (k97): declared-but-missing weight detection across the three
registries (comfy first-class), enqueue on the EXISTING transfer plane with
dedupe / unresolved-source / disk-floor refusals, and the sentinel
weight_missing fast path with its own downloads gate (default ON).

Run:  python -m pytest tests/test_provisioner.py -q   (in py/operations/hugpy_ops)
"""
import os
from types import SimpleNamespace

import pytest

from hugpy_ops import provisioner
from hugpy_ops.provisioner import (
    COMFY_SHARED_ASSETS,
    COMFY_WORKFLOW_REQUIREMENTS,
    Want,
    comfy_wants,
    enqueue,
    studio_wants,
    tasks_wants,
)
from hugpy_ops.sentinel import checks, remedies, runner
from hugpy_ops.sentinel.cases import Anomaly, CaseStore
from hugpy_ops.sentinel.settings import SentinelSettings, load_settings


# --------------------------------------------------------------------------
# fixtures: a fake store root + fake registries


@pytest.fixture(autouse=True)
def _no_live_comfy(monkeypatch):
    """Pin the per-run Manager-catalog memo to 'fetched, unavailable' so no
    test ever touches the live comfy — k97 behavior — unless it injects a
    catalog explicitly (k97b tests) or monkeypatches _fetch_json (CLI test,
    which resets the caches itself)."""
    monkeypatch.setattr(provisioner, "_catalog_cache", None)
    monkeypatch.setattr(provisioner, "_folders_cache", None)


def _touch(path, size=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x" * size)


COMFY_ROWS = {
    "comfy-present": {"framework": "comfy", "hub_id": "Org/Present",
                      "filename": "present.safetensors"},
    "comfy-zero": {"framework": "comfy", "hub_id": "Org/Zero",
                   "filename": "zero.safetensors"},
    "comfy-missing": {"framework": "comfy", "hub_id": "Org/Missing",
                      "filename": "missing.safetensors"},
    "not-comfy": {"framework": "transformers", "hub_id": "Org/LLM"},
}


def _comfy_root(tmp_path):
    root = tmp_path / "store"
    _touch(str(root / "checkpoints" / "present.safetensors"), size=8)
    _touch(str(root / "checkpoints" / "zero.safetensors"), size=0)
    # id_lock shared assets present so only the checkpoint wants fire.
    for asset in COMFY_SHARED_ASSETS.values():
        if asset.kind != "node_pack":
            _touch(str(root / asset.dest_dir / asset.dest_name), size=8)
    return str(root)


def test_comfy_wants_zero_byte_and_absent(tmp_path):
    root = _comfy_root(tmp_path)
    out = comfy_wants(rows=COMFY_ROWS, root=root)
    by_name = {w.name: w for w in out}
    assert set(by_name) == {"comfy-zero", "comfy-missing"}
    assert by_name["comfy-zero"].reason == "0-byte"
    assert by_name["comfy-missing"].reason == "absent"
    w = by_name["comfy-missing"]
    assert w.registry == "comfy" and w.resolved
    assert w.hub_id == "Org/Missing" and w.filename == "missing.safetensors"
    assert w.fingerprint == "weight_missing:comfy:comfy-missing"


def test_comfy_wants_id_lock_assets_are_workflow_requires(tmp_path):
    root = str(tmp_path / "bare")           # nothing on disk at all
    out = comfy_wants(rows={}, root=root)
    # Every downloadable §5b asset is wanted, pinned to its explicit source;
    # the node pack (code, not a weight) is never a want.
    assert {w.name for w in out} == {"comfy-ipadapter-sd15",
                                     "comfy-ipadapter-sdxl",
                                     "comfy-clip_vision-vit-h"}
    for w in out:
        assert w.reason == "workflow-requires"
        assert w.hub_id == "h94/IP-Adapter" and w.filename and w.resolved


def test_comfy_requirements_manifest_is_explicit():
    # Every workflow the runner can build is mapped; every non-checkpoint
    # requirement resolves to a manifest entry with cited provenance.
    assert set(COMFY_WORKFLOW_REQUIREMENTS) == {"t2i", "i2i", "id_lock"}
    for wf, reqs in COMFY_WORKFLOW_REQUIREMENTS.items():
        assert "checkpoint" in reqs
        for req in reqs:
            if req == "checkpoint":
                continue
            asset = COMFY_SHARED_ASSETS[req]
            assert asset.provenance
    # The dest-name sync with hugpy_media's comfy builder (_IPADAPTER_FILES)
    # is a cross-package check: see the server integration test
    # test_ops_media_comfy_assets.py (media is not an ops dependency).


# --------------------------------------------------------------------------
# k97b: ComfyUI-Manager catalog-first resolution


def _entry(**kw):
    base = dict(name="Missing Model", type="checkpoint", base="SD1.x",
                save_path="checkpoints/SD1.5", filename="missing.safetensors",
                size="2.13GB", installed="False",
                url="https://huggingface.co/CatalogOrg/CatalogRepo/resolve/"
                    "main/sub/missing.safetensors")
    base.update(kw)
    return base


def test_comfy_wants_catalog_hit_wins_over_hf(tmp_path):
    root = _comfy_root(tmp_path)
    catalog = [
        # cross-kind homonym listed FIRST — must lose to the checkpoint entry
        _entry(name="Homonym Lora", type="lora", save_path="loras",
               url="https://huggingface.co/Wrong/Repo/resolve/main/"
                   "missing.safetensors"),
        _entry(),
    ]
    folders = {"checkpoints": ["/somewhere/else",
                               str(tmp_path / "store" / "checkpoints")]}
    out = comfy_wants(rows=COMFY_ROWS, root=root, catalog=catalog,
                      folders=folders)
    by_name = {w.name: w for w in out}
    w = by_name["comfy-missing"]
    # The Manager entry WINS over the registry row's Org/Missing…
    assert w.hub_id == "CatalogOrg/CatalogRepo"
    assert w.filename == "sub/missing.safetensors"
    assert w.include == ["sub/missing.safetensors"]     # plane pulls ONE file
    # …its save_path lands in the under-root dir comfy actually scans…
    assert w.dest == str(tmp_path / "store" / "checkpoints" / "SD1.5"
                         / "missing.safetensors")
    assert w.est_bytes == int(2.13e9)
    assert "ComfyUI-Manager" in w.note
    assert w.resolved and w.registry == "comfy"
    # …and a row the catalog does NOT know keeps the k97 derivation.
    assert by_name["comfy-zero"].hub_id == "Org/Zero"
    assert "ComfyUI-Manager" not in by_name["comfy-zero"].note


def test_comfy_wants_comfy_down_falls_back_to_hf(tmp_path):
    root = _comfy_root(tmp_path)
    out = comfy_wants(rows=COMFY_ROWS, root=root, catalog=None, folders=None)
    by_name = {w.name: w for w in out}
    assert set(by_name) == {"comfy-zero", "comfy-missing"}
    w = by_name["comfy-missing"]
    assert w.hub_id == "Org/Missing" and w.filename == "missing.safetensors"
    assert w.dest == os.path.join(root, "checkpoints", "missing.safetensors")


def test_comfy_wants_non_hf_url_unresolved_with_url_noted(tmp_path):
    root = _comfy_root(tmp_path)
    rows = {
        "comfy-civitai": {"framework": "comfy",
                          "filename": "styleLora.safetensors"},
        "comfy-missing": COMFY_ROWS["comfy-missing"],
    }
    catalog = [
        _entry(name="Style Lora", type="lora", save_path="loras",
               filename="styleLora.safetensors",
               url="https://civitai.com/api/download/models/1234"),
        _entry(url="https://civitai.com/api/download/models/999"),
    ]
    out = comfy_wants(rows=rows, root=root, catalog=catalog, folders=None)
    by_name = {w.name: w for w in out}
    # No hub id anywhere -> UNRESOLVED, the Manager url surfaced in the note.
    w = by_name["comfy-civitai"]
    assert not w.resolved
    assert "civitai.com/api/download/models/1234" in w.note
    # The row's own hub id still resolves (route (b)), url noted alongside.
    w = by_name["comfy-missing"]
    assert w.hub_id == "Org/Missing" and w.resolved
    assert "civitai.com/api/download/models/999" in w.note


def test_comfy_wants_id_lock_assets_resolve_via_catalog(tmp_path):
    root = str(tmp_path / "bare")           # nothing on disk at all
    # The LIVE catalog's clip_vision row: desired filename differs from the
    # hub file — the §5b rename — and must carry a rename note.
    catalog = [_entry(
        name="CLIP-ViT-H-14", type="clip_vision", save_path="clip_vision",
        filename="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors", size="2.5GB",
        url="https://huggingface.co/h94/IP-Adapter/resolve/main/models/"
            "image_encoder/model.safetensors")]
    out = comfy_wants(rows={}, root=root, catalog=catalog, folders=None)
    by_name = {w.name: w for w in out}
    w = by_name["comfy-clip_vision-vit-h"]
    assert w.reason == "workflow-requires"
    assert w.hub_id == "h94/IP-Adapter"
    assert w.filename == "models/image_encoder/model.safetensors"
    assert w.est_bytes == int(2.5e9)
    assert "rename to CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" in w.note
    # Assets the catalog lacks keep their pinned §5b sources.
    assert by_name["comfy-ipadapter-sd15"].hub_id == "h94/IP-Adapter"
    assert "§5b" in by_name["comfy-ipadapter-sd15"].note


def test_comfy_wants_satisfied_at_catalog_mapped_dest(tmp_path):
    """A file already placed at the CATALOG-mapped dest (a prior Manager-route
    placement, e.g. checkpoints/SD1.5/) is present where comfy finds it —
    no want, no forever re-want against the k97 path."""
    root = _comfy_root(tmp_path)
    _touch(str(tmp_path / "store" / "checkpoints" / "SD1.5"
               / "missing.safetensors"), size=8)
    out = comfy_wants(rows=COMFY_ROWS, root=root, catalog=[_entry()],
                      folders=None)
    assert "comfy-missing" not in {w.name for w in out}
    assert "comfy-zero" in {w.name for w in out}       # still starved


def test_hub_from_url_variants():
    assert provisioner.hub_from_url(
        "https://huggingface.co/Org/Repo/resolve/main/a/b.safetensors"
    ) == ("Org/Repo", "a/b.safetensors")
    assert provisioner.hub_from_url(
        "https://huggingface.co/ShilongLiu/GroundingDINO/raw/main/x.cfg.py"
    ) == ("ShilongLiu/GroundingDINO", "x.cfg.py")
    assert provisioner.hub_from_url(
        "https://civitai.com/api/download/models/1") is None
    assert provisioner.hub_from_url(
        "https://github.com/x/y/raw/main/z.pth") is None
    assert provisioner.hub_from_url("https://huggingface.co/Org/Repo") is None
    assert provisioner.hub_from_url("") is None


def test_parse_size():
    assert provisioner._parse_size("4.71MB") == 4710000
    assert provisioner._parse_size("6.94GB") == int(6.94e9)
    assert provisioner._parse_size("143.5KB") == 143500
    assert provisioner._parse_size("") is None
    assert provisioner._parse_size(None) is None
    assert provisioner._parse_size("unknown") is None


def test_catalog_cli_lists_presence(tmp_path, monkeypatch, capsys):
    root = tmp_path / "store"
    _touch(str(root / "checkpoints" / "have.safetensors"), size=8)
    payloads = {
        "externalmodel/getlist": {"models": [
            _entry(name="Have", filename="have.safetensors",
                   save_path="checkpoints",
                   url="https://huggingface.co/O/R/resolve/main/"
                       "have.safetensors"),
            _entry(name="Nope", type="lora", base="SDXL", size="144MB",
                   filename="nope.safetensors", save_path="loras",
                   url="https://civitai.com/api/download/models/5"),
            # installed flag is a STRING on the wire — must count as present
            # even with nothing at the mapped dest.
            _entry(name="Flagged", type="VAE", base="FLUX.1", size="335MB",
                   filename="ae.safetensors", save_path="vae/FLUX1",
                   installed="True",
                   url="https://huggingface.co/O/F/resolve/main/"
                       "ae.safetensors"),
        ]},
        "experiment/models": [
            {"name": "checkpoints", "folders": [str(root / "checkpoints")]},
            {"name": "loras",
             "folders": [str(root / "comfy-kinds" / "loras")]},
        ],
    }

    def fake_fetch(url, timeout=10.0):
        for frag, payload in payloads.items():
            if frag in url:
                return payload
        raise AssertionError("unexpected fetch: %s" % url)

    monkeypatch.setattr(provisioner, "_fetch_json", fake_fetch)
    provisioner.reset_catalog_caches()
    assert provisioner.main(["catalog", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "[present] Have | checkpoint | SD1.x | 2.13GB | huggingface.co" in out
    assert "[absent ] Nope | lora | SDXL | 144MB | civitai.com" in out
    assert "[present] Flagged | VAE | FLUX.1 | 335MB | huggingface.co" in out
    assert "3 entries — 2 present, 1 absent" in out
    # --missing lists only the absent entry, counts unchanged.
    provisioner.reset_catalog_caches()
    assert provisioner.main(["catalog", "--root", str(root),
                             "--missing"]) == 0
    out = capsys.readouterr().out
    assert "Nope" in out and "[present]" not in out
    assert "3 entries — 2 present, 1 absent" in out
    # --type filters before counting.
    provisioner.reset_catalog_caches()
    assert provisioner.main(["catalog", "--root", str(root),
                             "--type", "lora"]) == 0
    out = capsys.readouterr().out
    assert "1 entry (type=lora) — 0 present, 1 absent" in out


def test_catalog_cli_degrades_when_comfy_down(tmp_path, monkeypatch, capsys):
    def boom(url, timeout=10.0):
        raise OSError("connection refused")

    monkeypatch.setattr(provisioner, "_fetch_json", boom)
    provisioner.reset_catalog_caches()
    assert provisioner.main(["catalog", "--root", str(tmp_path)]) == 0
    assert "unavailable" in capsys.readouterr().out


def _studio_cfg(model_id, weight_uri, source_url="https://huggingface.co/x",
                synthetic=False):
    return SimpleNamespace(model_id=model_id, weight_uri=weight_uri,
                           source_url=source_url, synthetic=synthetic)


def test_studio_wants_absent_zero_byte_and_unresolved(tmp_path):
    weights_root = tmp_path / "weights"
    os.makedirs(weights_root / "Org" / "Zero")          # dir with no bytes
    _touch(str(weights_root / "Org" / "Present" / "model.safetensors"))
    models = {
        "present": _studio_cfg("present", "Org/Present"),
        "zero": _studio_cfg("zero", "Org/Zero"),
        "gone": _studio_cfg("gone", "Org/Gone"),
        "github-only": _studio_cfg("github-only", "someone/Repo",
                                   source_url="https://github.com/someone/Repo"),
        "synthetic": _studio_cfg("synthetic", "synthetic://procedural",
                                 synthetic=True),
        "ffmpeg": _studio_cfg("ffmpeg", "ffmpeg://minterpolate"),
    }
    out = studio_wants(models=models, weights_root=str(weights_root))
    by_name = {w.name: w for w in out}
    assert set(by_name) == {"zero", "gone", "github-only"}
    assert by_name["zero"].reason == "0-byte"
    assert by_name["gone"].reason == "absent" and by_name["gone"].resolved
    # A weight_uri whose source_url is not huggingface is NEVER treated as a
    # hub id — surfaced UNRESOLVED, not guessed.
    assert not by_name["github-only"].resolved
    assert "not derivable" in by_name["github-only"].note


def test_tasks_wants_uses_read_through_resolver(tmp_path):
    rows = {
        "here": {"framework": "transformers", "hub_id": "Org/Here"},
        "gone": {"framework": "transformers", "hub_id": "Org/Gone"},
        "comfy-row": {"framework": "comfy", "hub_id": "Org/C",
                      "filename": "c.safetensors"},
        "no-hub": {"framework": "transformers"},
    }

    def resolver(row, root, require_complete=True):
        if row.get("hub_id") == "Org/Here":
            return str(tmp_path / "models" / "Org" / "Here")
        return None if require_complete else None

    out = tasks_wants(rows=rows, root=str(tmp_path), resolver=resolver)
    # comfy rows belong to the comfy scan; hubless rows have no source; a
    # resolver-complete row is present wherever it lives (legacy layouts too).
    assert [w.name for w in out] == ["gone"]
    assert out[0].registry == "tasks" and out[0].reason == "absent"


# --------------------------------------------------------------------------
# enqueue: rides the existing queue; refusals are explicit


def _want(**kw):
    base = dict(registry="comfy", name="comfy-missing", reason="absent",
                dest="/store/checkpoints/missing.safetensors",
                hub_id="Org/Missing", filename="missing.safetensors",
                framework="comfy")
    base.update(kw)
    return Want(**base)


def test_enqueue_rides_the_existing_download_queue():
    calls = []

    def fake_enqueue(model_key, model, total_bytes=None, transport="web"):
        calls.append((model_key, model, total_bytes, transport))
        return SimpleNamespace(id="job-1")

    res = enqueue(_want(), existing_jobs=[], free_bytes=10**13,
                  enqueue_fn=fake_enqueue)
    assert res == {"enqueued": True, "job_id": "job-1",
                   "want": "comfy-missing"}
    (model_key, model, total_bytes, transport) = calls[0]
    assert model_key == "comfy-missing" and transport == "provisioner"
    assert model["hub_id"] == "Org/Missing"
    assert model["filename"] == "missing.safetensors"


def test_enqueue_dedupes_against_live_jobs():
    boom = lambda *a, **k: pytest.fail("must not enqueue a duplicate")
    # same model_key, live
    res = enqueue(_want(), existing_jobs=[{"model_key": "comfy-missing",
                                           "status": "pending"}],
                  free_bytes=10**13, enqueue_fn=boom)
    assert res["enqueued"] is False and res["reason"] == "duplicate"
    # same hub_id+filename under a different key, live
    res = enqueue(_want(), existing_jobs=[
        {"model_key": "other", "status": "processing",
         "payload": {"model": {"hub_id": "Org/Missing",
                               "filename": "missing.safetensors"}}}],
        free_bytes=10**13, enqueue_fn=boom)
    assert res["enqueued"] is False and res["reason"] == "duplicate"


def test_enqueue_refuses_unresolved_source():
    res = enqueue(_want(hub_id=None), existing_jobs=[], free_bytes=10**13,
                  enqueue_fn=lambda *a, **k: pytest.fail("must not guess"))
    assert res == {"enqueued": False, "reason": "unresolved-source",
                   "want": "comfy-missing"}


def test_enqueue_refuses_past_disk_floor():
    floor = provisioner.floor_bytes()
    res = enqueue(_want(), existing_jobs=[], free_bytes=floor - 1,
                  enqueue_fn=lambda *a, **k: pytest.fail("floor breached"))
    assert res["enqueued"] is False and res["reason"] == "disk-floor"
    # est_bytes eats into the margin: free above floor but not by est_bytes.
    res = enqueue(_want(est_bytes=10**9), existing_jobs=[],
                  free_bytes=floor + 10**8,
                  enqueue_fn=lambda *a, **k: pytest.fail("floor breached"))
    assert res["reason"] == "disk-floor"


# --------------------------------------------------------------------------
# sentinel: the weight_missing check + downloads gate semantics


def _evidence(**kw):
    return {**_want(**kw).to_evidence()}


def test_check_missing_weights_one_anomaly_per_fingerprint():
    out = checks.check_missing_weights([_evidence(),
                                        _evidence(name="comfy-zero",
                                                  reason="0-byte")])
    assert [a.kind for a in out] == ["weight_missing", "weight_missing"]
    assert all(a.severity == "info" for a in out)
    assert [a.fingerprint for a in out] == [
        "weight_missing:comfy:comfy-missing",
        "weight_missing:comfy:comfy-zero"]
    assert out[0].evidence["hub_id"] == "Org/Missing"


def test_downloads_gate_defaults_on_and_off_respected():
    assert SentinelSettings().downloads_enabled is True
    assert load_settings(environ={}).downloads_enabled is True
    assert load_settings(environ={"HUGPY_SENTINEL_DOWNLOADS": "0"}
                         ).downloads_enabled is False
    assert load_settings(environ={"HUGPY_SENTINEL_DOWNLOADS": "1"}
                         ).downloads_enabled is True
    # downloads ride their OWN gate: remedies stay default OFF alongside.
    s = load_settings(environ={})
    assert s.remedies_enabled is False and s.downloads_enabled is True


def _wm_settings(tmp_path, **kw):
    s = SentinelSettings(state_dir=str(tmp_path / "state"),
                         central="http://central:7000")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_enqueue_download_remedy_posts_repos_download(tmp_path):
    s = _wm_settings(tmp_path)                    # downloads default ON
    remedy = next(r for r in remedies.WHITELIST
                  if r.name == "enqueue_download")
    anomaly = Anomaly("weight_missing:comfy:comfy-missing", "weight_missing",
                      "info", _evidence())
    assert remedy in remedies.eligible(anomaly)
    posts = []
    out = remedies.execute(remedy, {"central": s.central,
                                    "hub_id": "Org/Missing",
                                    "filename": "missing.safetensors",
                                    "register": False}, s,
                           http_post=lambda url, body:
                               posts.append((url, body)) or {"id": "job-7"})
    assert out == {"id": "job-7"}
    assert posts == [("http://central:7000/llm/repos/download",
                      {"hub_id": "Org/Missing",
                       "filename": "missing.safetensors",
                       "register": False})]
    # remedies gate OFF does not block downloads (separate gates)…
    assert s.remedies_enabled is False
    # …but the downloads gate OFF does.
    s.downloads_enabled = False
    with pytest.raises(remedies.DownloadsDisabled):
        remedies.execute(remedy, {"central": s.central,
                                  "hub_id": "Org/Missing"}, s,
                         http_post=lambda u, b: pytest.fail(
                             "must not POST while downloads are off"))


def test_unresolved_want_has_no_eligible_download_remedy():
    anomaly = Anomaly("weight_missing:studio:codeformer", "weight_missing",
                      "info", _evidence(hub_id=None, name="codeformer",
                                        registry="studio"))
    assert [r.name for r in remedies.eligible(anomaly)] == []


def test_runner_fast_path_enqueues_then_documents_no_agent(tmp_path):
    s = _wm_settings(tmp_path)
    store = CaseStore(s.db_path)
    posts = []
    get = lambda url, timeout=20.0: (
        {"jobs": [], "counts": {}} if "jobs" in url
        else [] if "workers" in url
        else {"ok": True, "count": 0, "capabilities": []})
    summary = runner.run_once(
        s, store=store, http_get=get,
        run=lambda *a, **k: pytest.fail("weight_missing must not spawn"),
        http_post=lambda url, body: posts.append((url, body)) or {"id": "j9"},
        wants_fn=lambda: [_evidence()])
    assert len(summary["opened"]) == 1
    case = store.get(summary["opened"][0])
    assert case.state == "remedied"
    assert "j9" in (case.note or "")
    assert posts and posts[0][0].endswith("/llm/repos/download")
    assert posts[0][1]["hub_id"] == "Org/Missing"
    assert posts[0][1]["register"] is False
    assert case.report_path and os.path.exists(case.report_path)
    # Re-detection touches, never re-enqueues.
    summary2 = runner.run_once(
        s, store=store, http_get=get,
        run=lambda *a, **k: pytest.fail("no spawn"),
        http_post=lambda url, body: pytest.fail("must not re-POST"),
        wants_fn=lambda: [_evidence()])
    assert summary2["opened"] == []
    store.close()


def test_runner_fast_path_manager_resolved_want(tmp_path):
    """A want resolved via the ComfyUI-Manager catalog rides the SAME
    weight_missing fast path and downloads gate as an HF-derived one — the
    catalog only changes WHICH hub id/filename the evidence carries."""
    s = _wm_settings(tmp_path)                    # downloads default ON
    store = CaseStore(s.db_path)
    posts = []
    get = lambda url, timeout=20.0: (
        {"jobs": [], "counts": {}} if "jobs" in url
        else [] if "workers" in url
        else {"ok": True, "count": 0, "capabilities": []})
    evidence = _evidence(hub_id="CatalogOrg/CatalogRepo",
                         filename="sub/missing.safetensors",
                         include=["sub/missing.safetensors"],
                         est_bytes=int(2.13e9),
                         note="ComfyUI-Manager catalog: Missing Model "
                              "(type checkpoint, save_path checkpoints/SD1.5)")
    summary = runner.run_once(
        s, store=store, http_get=get,
        run=lambda *a, **k: pytest.fail("weight_missing must not spawn"),
        http_post=lambda url, body: posts.append((url, body)) or {"id": "j3"},
        wants_fn=lambda: [evidence])
    case = store.get(summary["opened"][0])
    assert case.state == "remedied"
    assert posts and posts[0][0].endswith("/llm/repos/download")
    assert posts[0][1]["hub_id"] == "CatalogOrg/CatalogRepo"
    assert posts[0][1]["filename"] == "sub/missing.safetensors"
    store.close()
    # gate OFF blocks the Manager route exactly like the HF one.
    s2 = _wm_settings(tmp_path / "off", downloads_enabled=False)
    store2 = CaseStore(s2.db_path)
    summary2 = runner.run_once(
        s2, store=store2, http_get=get,
        run=lambda *a, **k: pytest.fail("no spawn"),
        http_post=lambda url, body: pytest.fail("gate off — must not POST"),
        wants_fn=lambda: [evidence])
    case2 = store2.get(summary2["opened"][0])
    assert case2.state == "documented"
    store2.close()


def test_runner_fast_path_documents_only_when_gate_off(tmp_path):
    s = _wm_settings(tmp_path, downloads_enabled=False)
    store = CaseStore(s.db_path)
    get = lambda url, timeout=20.0: (
        {"jobs": [], "counts": {}} if "jobs" in url
        else [] if "workers" in url
        else {"ok": True, "count": 0, "capabilities": []})
    summary = runner.run_once(
        s, store=store, http_get=get,
        run=lambda *a, **k: pytest.fail("no spawn"),
        http_post=lambda url, body: pytest.fail("gate off — must not POST"),
        wants_fn=lambda: [_evidence()])
    case = store.get(summary["opened"][0])
    assert case.state == "documented"
    assert "downloads gate is OFF" in (case.note or "")
    store.close()


def test_runner_fast_path_unresolved_documents_only(tmp_path):
    s = _wm_settings(tmp_path)
    store = CaseStore(s.db_path)
    get = lambda url, timeout=20.0: (
        {"jobs": [], "counts": {}} if "jobs" in url
        else [] if "workers" in url
        else {"ok": True, "count": 0, "capabilities": []})
    summary = runner.run_once(
        s, store=store, http_get=get,
        run=lambda *a, **k: pytest.fail("no spawn"),
        http_post=lambda url, body: pytest.fail("unresolved — must not POST"),
        wants_fn=lambda: [_evidence(hub_id=None, name="codeformer",
                                    registry="studio")])
    case = store.get(summary["opened"][0])
    assert case.state == "documented"
    assert "UNRESOLVED" in (case.note or "")
    store.close()

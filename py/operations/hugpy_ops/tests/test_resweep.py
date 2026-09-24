"""hugpy-model-resweep: the dry-run plan for the 12-model sweep shape, the
archive move (one rename) + MANIFEST row, refusal when a worker has the
model loaded, and the language-quant choice (never a projector, fits the
fleet). No network: catalog/workers/metadata are local fixtures."""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from hugpy_ops import resweep as R

GIB = 2 ** 30
WORKERS = [{"name": "aeb", "status": "online", "gpu_total_bytes_known": 24 * GIB,
            "ram_bar_total": 125 * GIB},
           {"name": "computron", "status": "online", "gpu_total_bytes_known": 8 * GIB,
            "ram_bar_total": 15 * GIB},
           {"name": "op", "status": "offline", "gpu_total_bytes_known": None}]

YMQ = [{"rfilename": f"Q-{n}.gguf", "size": s} for n, s in
       (("XL", 19_711_625_184), ("L", 17_225_762_784), ("XXS", 9_979_221_984))] + \
      [{"rfilename": "mmproj/Q-vision-Q8_0.gguf", "size": 629_247_488}]
QUEEN = [{"rfilename": f"Queen.{q}.gguf", "size": s} for q, s in
         (("Q6_K", 22_431_000_704), ("Q8_0", 29_047_085_184), ("Q4_K_M", 16_810_715_264))] + \
        [{"rfilename": "Queen.mmproj-f16.gguf", "size": 927_607_392}]
FLASH = [{"rfilename": f"Flash.{q}.gguf", "size": s} for q, s in
         (("IQ4_XS", 98_421_788_672), ("Q2_K", 80_447_450_112))] + \
        [{"rfilename": "Flash.mmproj-f16.gguf", "size": 904_004_352}]


def test_quant_choice_picks_a_language_quant_that_fits():
    p = R.choose_quant(YMQ, WORKERS)
    assert p["files"] == ["Q-XL.gguf"] and "mmproj" not in p["quant"]
    p = R.choose_quant(QUEEN, WORKERS)
    assert p["files"] == ["Queen.Q6_K.gguf"]                 # Q8_0 exceeds the 24 GiB card
    p = R.choose_quant(FLASH, WORKERS)
    assert p["files"] == ["Flash.IQ4_XS.gguf"] and "offload" in p["rule"]
    assert R.choose_quant([{"rfilename": "mmproj/x.gguf", "size": 1}], WORKERS)["error"]
    # a pinned language quant that fits wins; a pinned projector never does
    assert R.choose_quant(QUEEN, WORKERS, pinned="Queen.Q4_K_M.gguf")["files"] == ["Queen.Q4_K_M.gguf"]
    assert R.choose_quant(YMQ, WORKERS, pinned="mmproj/Q-vision-Q8_0.gguf")["files"] == ["Q-XL.gguf"]


def test_sharded_quant_sums_shards():
    sib = [{"rfilename": "Big-Q4-00001-of-00002.gguf", "size": 10 * GIB},
           {"rfilename": "Big-Q4-00002-of-00002.gguf", "size": 10 * GIB}]
    g = R.gguf_groups(sib)
    assert len(g) == 1 and g[0]["bytes"] == 20 * GIB and len(g[0]["files"]) == 2


def _metadata_db(path, repos, civitai):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE repo_info (hub_id TEXT PRIMARY KEY, payload TEXT, fetched_at REAL)")
    conn.execute("CREATE TABLE civitai_meta (stem TEXT PRIMARY KEY, payload TEXT, fetched_at REAL)")
    for h, sib in repos.items():
        conn.execute("INSERT INTO repo_info VALUES (?,?,0)", (h, json.dumps({"siblings": sib})))
    for stem, meta in civitai.items():
        conn.execute("INSERT INTO civitai_meta VALUES (?,?,0)", (stem, json.dumps(meta)))
    conn.commit()
    conn.close()


@pytest.fixture
def world(tmp_path):
    models = tmp_path / "models"
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    entries, catalog = [], {}
    comfy = ["dreamshaper-8"] + [f"comfy-c{i}" for i in range(6)]
    for i, key in enumerate(comfy):
        name = key.replace("comfy-", "")
        fn = f"{name}.safetensors"
        d = models / "misc" / "comfy" / name
        d.mkdir(parents=True)
        (d / fn).write_bytes(b"\0" * 16)
        (ckpt / fn).write_bytes(b"x" * 16)
        if i % 2 == 0:                                # sidecar for half, metadata store for the rest
            (ckpt / (fn + ".civitai.json")).write_text(json.dumps({"civitai_id": 100 + i, "version_id": 200 + i}))
        entries.append({"model_key": key, "framework": "comfy", "hub_id": f"comfy/{name}",
                        "why": "broken_download", "destination": str(d)})
        catalog[key] = {"model_key": key, "framework": "comfy", "filename": fn,
                        "destination": str(d), "workers": []}
    gg = {"AEON": ("o/AEON", QUEEN, "Queen.Q4_K_M.gguf"), "Echo-Mini": ("J/Echo-Mini", YMQ, "Q-XXS.gguf"),
          "YMQ": ("z/YMQ", YMQ, "mmproj/Q-vision-Q8_0.gguf"), "Flash": ("m/Flash", FLASH, None),
          "Queen": ("m/Queen", QUEEN, None)}
    repos = {}
    for key, (hub, sib, pin) in gg.items():
        d = models / "gguf" / hub
        d.mkdir(parents=True)
        (d / "x.gguf").write_bytes(b"x")
        repos[hub] = sib
        entries.append({"model_key": key, "framework": "gguf", "hub_id": hub, "why": "w", "destination": str(d)})
        catalog[key] = {"model_key": key, "framework": "gguf", "hub_id": hub, "filename": pin,
                        "primary_task": "text-generation", "destination": str(d),
                        "workers": [{"worker": "aeb", "loaded": key == "Echo-Mini"}]}
    db = str(tmp_path / "meta.db")
    _metadata_db(db, repos, {f"c{i}": {"civitai_id": 100 + i, "version_id": 200 + i} for i in range(6)})
    return {"entries": entries, "catalog": catalog, "ckpt": str(ckpt), "db": db,
            "archive": str(tmp_path / "ARCHIVE" / "MODELS_RESWEEP-2026-09-23")}


def _plans(world, force=False):
    return {e["model_key"]: R.plan_one(e, world["catalog"].get(e["model_key"]), WORKERS,
                                       archive_base=world["archive"], checkpoints_dir=world["ckpt"],
                                       db_path=world["db"], force=force)
            for e in world["entries"]}


def test_dry_run_plan_for_the_twelve(world):
    plans = _plans(world)
    assert len(plans) == 12
    comfy = [p for p in plans.values() if p["framework"] == "comfy"]
    assert len(comfy) == 7 and all(p["source"] == "civitai" and not p.get("error") for p in comfy)
    for p in comfy:
        body = p["request"]["body"]
        assert p["request"]["path"] == "/civitai/download"
        assert body["download_url"] == f"https://civitai.com/api/download/models/{body['version_id']}"
        assert len(p["moves"]) >= 2 and p["moves"][0][1].endswith(os.path.join("comfy", "comfy", p["original"].split("/")[-1]))
    assert plans["YMQ"]["request"]["body"]["filename"] == "Q-XL.gguf"      # projector pin replaced
    assert plans["Queen"]["request"]["body"]["filename"] == "Queen.Q6_K.gguf"
    assert plans["Flash"]["request"]["body"]["filename"] == "Flash.IQ4_XS.gguf"
    assert plans["AEON"]["request"]["body"]["filename"] == "Queen.Q4_K_M.gguf"  # fitting pin kept
    assert plans["AEON"]["request"]["path"] == "/llm/repos/download"
    assert plans["AEON"]["request"]["body"]["register"] is False
    assert plans["Echo-Mini"]["refused"].startswith("loaded/serving on aeb")
    # dry-run touched nothing
    assert all(os.path.exists(p["original"]) for p in plans.values())
    assert not os.path.exists(world["archive"])


def test_force_overrides_the_loaded_refusal(world):
    assert not _plans(world, force=True)["Echo-Mini"].get("refused")


def test_archive_is_a_move_with_a_manifest_row(world):
    p = _plans(world)["comfy-c1"]
    for src, dst in p["moves"]:
        R.move(src, dst)
    path = R.append_manifest(world["archive"], R.manifest_row(
        p["model_key"], p["moves"][0][0], p["moves"][0][1], p["why"], p["moves"][1:]))
    assert not os.path.exists(p["original"]) and os.path.isdir(p["archived"])
    assert os.listdir(p["archived"]) == ["c1.safetensors"]
    assert os.path.exists(os.path.join(p["archived"] + ".checkpoint", "c1.safetensors"))
    text = open(path).read()
    assert text.startswith("# MODELS_RESWEEP") and "| `comfy-c1` |" in text
    assert f'mv "{p["archived"]}" "{p["original"]}"' in text
    with pytest.raises(FileExistsError):                 # never overwrite an archive
        os.makedirs(p["original"])
        R.move(p["original"], p["archived"])


def test_no_listing_cached_is_an_error_not_a_hub_call(world, tmp_path):
    empty = str(tmp_path / "empty.db")
    _metadata_db(empty, {}, {})
    e = next(x for x in world["entries"] if x["model_key"] == "Queen")
    p = R.plan_one(e, world["catalog"]["Queen"], WORKERS, archive_base=world["archive"],
                   checkpoints_dir=world["ckpt"], db_path=empty)
    assert "no cached file listing" in p["error"]


def test_apply_refused_before_the_gate_is_live(monkeypatch):
    class C:
        base = "http://central.invalid"

        def get(self, path):
            return 200, "<html>spa</html>"
    ok, why = R.admission_live(C())
    assert not ok and "not the admission route" in why

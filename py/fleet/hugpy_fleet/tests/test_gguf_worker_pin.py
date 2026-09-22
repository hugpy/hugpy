"""Per-worker quant pin ("most amicable gguf") — resolver precedence + the
fit-aware auto-pick.

PRECEDENCE (mirrors gguf_election rule 0; an operator pin is never silently
upgraded):
    per-worker gguf_file_by_worker → model-wide gguf_file → cfg.filename →
    fit-aware auto-pick → plain election.

Runs both ways:
    venv/bin/python tests/test_gguf_worker_pin.py
    venv/bin/python -m pytest tests/test_gguf_worker_pin.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PROJECTS_HOME",
                      tempfile.mkdtemp(prefix="hugpy-worker-pin-test-"))
os.environ["SERVE_OVERRIDES_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="hugpy-worker-pin-ov-"), "serve_overrides.json")

from hugpy_fleet.worker import agent as A
from hugpy_engine.serve import overrides as OV


def _mkgguf(sizes):
    d = tempfile.mkdtemp()
    for name, sz in sizes.items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"\0" * sz)
    return d


SIZES = {
    "m.i1-Q2_K.gguf": 2000,
    "m.i1-Q4_K_M.gguf": 4000,   # plain-election winner
    "m.i1-Q6_K.gguf": 6000,
    "m.i1-Q8_0.gguf": 8000,
}


# --------------------------------------------------------------------------- #
# the pure per-worker resolver
# --------------------------------------------------------------------------- #
def test_resolver_matches_id_or_name_case_insensitive():
    by = {"AE": "q8_0", "w-123": "m.i1-Q6_K.gguf"}
    assert OV.resolve_gguf_for_worker(by, ["ae"]) == "q8_0"
    assert OV.resolve_gguf_for_worker(by, ["nope", "W-123"]) == "m.i1-Q6_K.gguf"
    assert OV.resolve_gguf_for_worker(by, ["computron"]) is None
    assert OV.resolve_gguf_for_worker({}, ["ae"]) is None
    assert OV.resolve_gguf_for_worker(None, ["ae"]) is None


def test_field_coercion_map_and_curl_string():
    OV.set_override("pin-coerce", {"gguf_file_by_worker": "ae=q8_0, computron=m.i1-Q2_K.gguf"})
    got = OV.get_override("pin-coerce").get("gguf_file_by_worker")
    assert got == {"ae": "q8_0", "computron": "m.i1-Q2_K.gguf"}, got
    # empty map clears the key
    OV.set_override("pin-coerce", {"gguf_file_by_worker": {}})
    assert "gguf_file_by_worker" not in OV.get_override("pin-coerce")


# --------------------------------------------------------------------------- #
# resolve_override_gguf precedence: per-worker beats model-wide
# --------------------------------------------------------------------------- #
def test_per_worker_pin_beats_model_wide():
    d = _mkgguf(SIZES)
    OV.set_override("pin-a", {"gguf_file": "m.i1-Q4_K_M.gguf",
                              "gguf_file_by_worker": {"ae": "q8_0"}})
    try:
        got_ae = OV.resolve_override_gguf("pin-a", d, worker=["ae"])
        got_other = OV.resolve_override_gguf("pin-a", d, worker=["computron"])
    finally:
        OV.set_override("pin-a", {"gguf_file": "", "gguf_file_by_worker": {}})
    assert got_ae and got_ae.endswith("m.i1-Q8_0.gguf"), got_ae      # token match
    assert got_other and got_other.endswith("m.i1-Q4_K_M.gguf"), got_other


def test_quant_token_substring_resolution():
    d = _mkgguf(SIZES)
    OV.set_override("pin-b", {"gguf_file_by_worker": {"box": "Q6_K"}})
    try:
        got = OV.resolve_override_gguf("pin-b", d, worker=["box"])
    finally:
        OV.set_override("pin-b", {"gguf_file_by_worker": {}})
    assert got and got.endswith("m.i1-Q6_K.gguf"), got


# --------------------------------------------------------------------------- #
# fit-aware auto-pick: largest fitting; never overrides any designation
# --------------------------------------------------------------------------- #
def test_select_fitting_gguf_picks_largest_that_fits():
    d = _mkgguf(SIZES)
    # budget admits q6_k (6000*1.15=6900) but not q8_0 (9200)
    pick = OV.select_fitting_gguf("fit-a", d, {"framework": "gguf"},
                                  budget_bytes=8000)
    assert pick == "m.i1-Q6_K.gguf", pick
    # everything fits -> the largest wins
    pick = OV.select_fitting_gguf("fit-a", d, {"framework": "gguf"},
                                  budget_bytes=10**6)
    assert pick == "m.i1-Q8_0.gguf", pick
    # nothing fits -> None (caller falls back to the plain election)
    assert OV.select_fitting_gguf("fit-a", d, {"framework": "gguf"},
                                  budget_bytes=100) is None
    assert OV.select_fitting_gguf("fit-a", d, {"framework": "gguf"},
                                  budget_bytes=None) is None


def test_select_fitting_gguf_skips_incomplete_shard_sets():
    d = _mkgguf({"m.i1-Q4_K_M.gguf": 4000,
                 "big-Q8_0-00001-of-00003.gguf": 3000})   # 1 of 3 shards
    pick = OV.select_fitting_gguf("fit-b", d, {"framework": "gguf"},
                                  budget_bytes=10**6)
    assert pick == "m.i1-Q4_K_M.gguf", pick


def test_autofit_never_overrides_a_designation():
    d = _mkgguf(SIZES)
    OV.set_gguf_autofit_hook(lambda mk, md, cfg: "m.i1-Q8_0.gguf")
    try:
        # no designation -> the hook's pick flows through
        assert OV.autofit_gguf_prefer("auto-a", d, {"framework": "gguf"}) \
            == "m.i1-Q8_0.gguf"
        # model-wide pin -> None (pin resolves via its own path)
        OV.set_override("auto-a", {"gguf_file": "m.i1-Q2_K.gguf"})
        assert OV.autofit_gguf_prefer("auto-a", d, {"framework": "gguf"}) is None
        OV.set_override("auto-a", {"gguf_file": ""})
        # cfg.filename designation -> None
        assert OV.autofit_gguf_prefer(
            "auto-a", d, {"framework": "gguf",
                          "filename": "m.i1-Q2_K.gguf"}) is None
        # per-worker pin for THIS box's identity -> None
        forms = OV._local_worker_forms()
        if forms:
            OV.set_override("auto-a",
                            {"gguf_file_by_worker": {forms[0]: "q2_k"}})
            assert OV.autofit_gguf_prefer("auto-a", d,
                                          {"framework": "gguf"}) is None
            OV.set_override("auto-a", {"gguf_file_by_worker": {}})
    finally:
        OV.set_gguf_autofit_hook(None)
        OV.set_override("auto-a", {"gguf_file": "", "gguf_file_by_worker": {}})


def test_autofit_unregistered_hook_means_plain_election():
    d = _mkgguf(SIZES)
    OV.set_gguf_autofit_hook(None)
    assert OV.autofit_gguf_prefer("auto-b", d, {"framework": "gguf"}) is None


def test_worker_selector_disabled_by_env():
    orig = os.environ.get("HUGPY_QUANT_AUTOFIT")
    try:
        os.environ["HUGPY_QUANT_AUTOFIT"] = "0"
        assert A._autofit_gguf_pick("any", "/nonexistent") is None
    finally:
        if orig is None:
            os.environ.pop("HUGPY_QUANT_AUTOFIT", None)
        else:
            os.environ["HUGPY_QUANT_AUTOFIT"] = orig


def test_worker_selector_uses_free_vram_budget():
    d = _mkgguf(SIZES)
    orig_fv = A._free_vram_bytes
    try:
        A._free_vram_bytes = lambda: 8000        # admits q6_k, not q8_0
        pick = A._autofit_gguf_pick("fit-c", d, {"framework": "gguf"})
        assert pick == "m.i1-Q6_K.gguf", pick
        A._free_vram_bytes = lambda: None        # unmeasurable -> no pick
        assert A._autofit_gguf_pick("fit-c", d, {"framework": "gguf"}) is None
    finally:
        A._free_vram_bytes = orig_fv


# --------------------------------------------------------------------------- #
# plain-script runner (pytest not required)
# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            ok += 1
            print(f"[ok]   {t.__name__}")
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(_main())

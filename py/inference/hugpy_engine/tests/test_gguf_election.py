"""Effective-GGUF election: shard completeness, quant preference, determinism.

THE INCIDENT these guard (2026-07-28). A Qwen2.5-7B-Instruct-GGUF directory held
a complete fp16 4-shard split, a complete q4_k_m 2-shard split, several complete
single-file quants, and litter (a lone ``q4_0-00002-of-00002`` with no shard 1).
With no manifest ``filename`` pin the elector chose THE FP16 SPLIT — 15.2 GB of
full precision over a 4.7 GB q4_k_m, on a fleet whose smallest card is 8 GB.

Two independent defects produced that:
  * the old rule tried "first shard of a split gguf" BEFORE the quant rank and
    broke ties lexically, and ``f`` sorts before ``q``;
  * nothing anywhere checked whether a shard set was COMPLETE, so one file of
    three was as electable as a whole model.

Operator doctrine: "defaults are promises — every default must be a SUCCESS PATH
on the real fleet". These tests are that promise, written down.
"""
import os

import pytest

from hugpy_engine import gguf_election as ge


GB = 1000 ** 3


def v(files):
    """[(relpath, bytes), …] -> variants, keyed by filename for easy asserts."""
    return {x["filename"]: x for x in ge.group_variants(files)}


# --------------------------------------------------------------------------- #
# the incident directory, reproduced exactly
# --------------------------------------------------------------------------- #

INCIDENT = [
    # complete fp16 4-shard split — 15.2 GB, the wrong answer
    ("qwen2.5-7b-instruct-fp16-00001-of-00004.gguf", 3951521376),
    ("qwen2.5-7b-instruct-fp16-00002-of-00004.gguf", 3864909312),
    ("qwen2.5-7b-instruct-fp16-00003-of-00004.gguf", 3864894976),
    ("qwen2.5-7b-instruct-fp16-00004-of-00004.gguf", 3556527872),
    # complete q4_k_m 2-shard split — 4.7 GB, the right answer
    ("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", 3993201344),
    ("qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf", 689872288),
    # LITTER: shard 2 of 2 with no shard 1 — never electable
    ("qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf", 448162496),
    # complete singles + other complete splits
    ("qwen2.5-7b-instruct-q2_k.gguf", 3015940000),
    ("qwen2.5-7b-instruct-q3_k_m.gguf", 3808391072),
    ("qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf", 3989841792),
    ("qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf", 1454989568),
    ("qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf", 3950642464),
    ("qwen2.5-7b-instruct-q6_k-00002-of-00002.gguf", 2303556416),
]


def test_the_incident_elects_q4_k_m_not_fp16():
    """THE regression test. A quantized variant must beat full precision."""
    winner = ge.elect(ge.group_variants(INCIDENT))
    assert winner["filename"] == "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    assert winner["quant"] == "q4_k_m"
    assert winner["full_precision"] is False


def test_the_incident_marks_the_lone_shard_incomplete():
    got = v(INCIDENT)
    lone = got["qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf"]
    assert lone["complete"] is False
    assert lone["missing_shards"] == [1]
    assert "1 of 2 shards present" in lone["incomplete_reason"]
    # …and everything else in that directory IS complete.
    assert all(x["complete"] for k, x in got.items() if "q4_0" not in k)


def test_shards_fold_into_one_variant_summing_bytes():
    got = v(INCIDENT)
    q4 = got["qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"]
    assert q4["bytes"] == 3993201344 + 689872288
    assert q4["shards"] == 2 and q4["shard_total"] == 2
    assert len(q4["members"]) == 2
    # The ENTRYPOINT is shard 1 — what llama.cpp is actually pointed at.
    assert q4["filename"].endswith("-00001-of-00002.gguf")


# --------------------------------------------------------------------------- #
# completeness
# --------------------------------------------------------------------------- #

def test_incomplete_shard_set_is_never_elected():
    files = [("m-q8_0-00001-of-00003.gguf", 10),      # 1 of 3 — litter
             ("m-q6_k.gguf", 5)]                      # complete, worse quant
    assert ge.elect(ge.group_variants(files))["filename"] == "m-q6_k.gguf"


def test_incomplete_reports_every_missing_index():
    got = v([("m-q8_0-00002-of-00004.gguf", 1)])
    x = got["m-q8_0-00002-of-00004.gguf"]
    assert x["complete"] is False
    assert x["missing_shards"] == [1, 3, 4]


def test_a_complete_set_is_complete_regardless_of_arrival_order():
    files = [("m-q4_k_m-00003-of-00003.gguf", 1),
             ("m-q4_k_m-00001-of-00003.gguf", 1),
             ("m-q4_k_m-00002-of-00003.gguf", 1)]
    x = ge.group_variants(files)[0]
    assert x["complete"] is True and x["missing_shards"] == []
    assert x["filename"] == "m-q4_k_m-00001-of-00003.gguf"   # entry = lowest idx


def test_a_single_unsharded_file_is_complete():
    x = ge.group_variants([("m-q5_k_m.gguf", 7)])[0]
    assert x["complete"] is True
    assert x["shard_total"] is None and x["shards"] == 1


def test_two_sharded_families_in_subdirs_do_not_merge():
    files = [("q4/m-00001-of-00002.gguf", 1), ("q4/m-00002-of-00002.gguf", 1),
             ("q8/m-00001-of-00002.gguf", 1), ("q8/m-00002-of-00002.gguf", 1)]
    got = ge.group_variants(files)
    assert len(got) == 2
    assert all(x["complete"] for x in got)


def test_all_incomplete_still_elects_rather_than_returning_none():
    """Rule 4: a dir of pure litter must NOT read as "no gguf at all".

    Returning None flips model_looks_downloaded to False for the whole
    directory, which reads as ABSENT and provokes a re-download storm. A
    half-present model should fail loudly at load, not restage the fleet."""
    files = [("m-q4_k_m-00002-of-00002.gguf", 1),
             ("m-q8_0-00001-of-00003.gguf", 1)]
    got = ge.elect(ge.group_variants(files))
    assert got is not None
    assert got["complete"] is False
    assert got["quant"] == "q4_k_m"          # still the best of a bad lot


def test_elect_on_nothing_is_none():
    assert ge.elect([]) is None
    assert ge.elect(ge.group_variants([])) is None


# --------------------------------------------------------------------------- #
# quant preference
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fp", ["f16", "fp16", "bf16", "f32", "fp32"])
def test_any_quant_beats_any_full_precision(fp):
    files = [(f"m-{fp}.gguf", 100), ("m-q2_k.gguf", 1)]
    assert ge.elect(ge.group_variants(files))["filename"] == "m-q2_k.gguf"


def test_full_precision_wins_only_when_nothing_else_is_complete():
    files = [("m-fp16.gguf", 100),
             ("m-q4_k_m-00001-of-00002.gguf", 1)]      # incomplete: no shard 2
    assert ge.elect(ge.group_variants(files))["filename"] == "m-fp16.gguf"


def test_q4_k_m_is_the_fleet_default_among_complete_quants():
    """Head of QUANT_ORDER. Deliberately NOT "largest complete quant": q6_k
    would win that and does not fit computron's 8 GB card with usable context.
    Fidelity is what a designation is for; the DEFAULT has to fit."""
    files = [("m-q8_0.gguf", 8), ("m-q6_k.gguf", 6), ("m-q5_k_m.gguf", 5),
             ("m-q4_k_m.gguf", 4), ("m-q2_k.gguf", 2)]
    assert ge.elect(ge.group_variants(files))["filename"] == "m-q4_k_m.gguf"


def test_quant_token_is_delimited_not_substring_matched():
    # "q4_0" is a substring of "q4_0_4_8" and "f16" of "bf16" — the old
    # substring rank table mis-classified both.
    assert ge.quant_token("m-q4_0_4_8.gguf") == "q4_0_4_8"
    assert ge.quant_token("m-bf16.gguf") == "bf16"
    assert ge.quant_token("m-q4_k_m-00001-of-00002.gguf") == "q4_k_m"
    assert ge.quant_token("model.gguf") == ""


def test_unknown_quant_outranks_full_precision_but_not_known_quants():
    files = [("m-q9_z_future.gguf", 1), ("m-fp16.gguf", 1), ("m-q4_k_m.gguf", 1)]
    ranked = sorted(ge.group_variants(files), key=ge.election_key)
    assert [x["quant"] for x in ranked] == ["q4_k_m", "q9_z_future", "fp16"]


def test_election_is_deterministic_regardless_of_input_order():
    import random
    files = list(INCIDENT)
    winners = set()
    for _ in range(12):
        random.shuffle(files)
        winners.add(ge.elect(ge.group_variants(files))["filename"])
    assert len(winners) == 1


# --------------------------------------------------------------------------- #
# elect_path — the form get_gguf_file calls
# --------------------------------------------------------------------------- #

def test_elect_path_returns_the_entrypoint_absolute_path(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    names = ["m-fp16-00001-of-00002.gguf", "m-fp16-00002-of-00002.gguf",
             "m-q4_k_m.gguf"]
    paths = []
    for n in names:
        p = d / n
        p.write_bytes(b"x")
        paths.append(str(p))
    got = ge.elect_path(paths)
    assert os.path.basename(got) == "m-q4_k_m.gguf"
    assert os.path.isabs(got)


def test_elect_path_on_a_single_file_returns_it(tmp_path):
    p = tmp_path / "only-fp16.gguf"
    p.write_bytes(b"x")
    assert ge.elect_path([str(p)]) == str(p)


def test_elect_path_empty_is_none():
    assert ge.elect_path([]) is None


# --------------------------------------------------------------------------- #
# get_gguf_file — the real entry point, designation still outranks election
# --------------------------------------------------------------------------- #

class _Cfg:
    def __init__(self, filename=None):
        self.filename = filename


def _mkdir_with(tmp_path, names):
    d = tmp_path / "model"
    d.mkdir()
    for n in names:
        p = d / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 16)
    return str(d)


def test_get_gguf_file_elects_the_quant_over_fp16(tmp_path):
    from hugpy_engine.config.main import get_gguf_file
    d = _mkdir_with(tmp_path, [n for n, _ in INCIDENT])
    got = get_gguf_file(d, _Cfg())
    assert os.path.basename(got) == \
        "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"


def test_get_gguf_file_designation_still_wins(tmp_path):
    """A pin must outrank the election — including a pin AT full precision and
    a pin at an INCOMPLETE variant. Election is only what happens when nobody
    said; the operator saying so is not a default."""
    from hugpy_engine.config.main import get_gguf_file
    d = _mkdir_with(tmp_path, [n for n, _ in INCIDENT])
    got = get_gguf_file(d, _Cfg("qwen2.5-7b-instruct-fp16-00001-of-00004.gguf"))
    assert "fp16" in os.path.basename(got)
    # substring/quant-tag designation, the documented shorthand
    got = get_gguf_file(d, _Cfg(), prefer="q6_k")
    assert "q6_k" in os.path.basename(got)


def test_get_gguf_file_prefer_beats_cfg_filename(tmp_path):
    from hugpy_engine.config.main import get_gguf_file
    d = _mkdir_with(tmp_path, [n for n, _ in INCIDENT])
    got = get_gguf_file(d, _Cfg("q2_k"), prefer="q6_k")
    assert "q6_k" in os.path.basename(got)


def test_get_gguf_file_on_empty_dir_is_none(tmp_path):
    from hugpy_engine.config.main import get_gguf_file
    d = tmp_path / "empty"
    d.mkdir()
    assert get_gguf_file(str(d), _Cfg()) is None


# --------------------------------------------------------------------------- #
# the wire shape overrides.py exports
# --------------------------------------------------------------------------- #

def test_variants_detail_marks_incomplete_and_elects_the_quant(tmp_path):
    from hugpy_engine.serve.overrides import gguf_variants_detail
    d = _mkdir_with(tmp_path, [n for n, _ in INCIDENT])
    detail = gguf_variants_detail("Qwen2.5-7B-Instruct-GGUF", d, _Cfg())
    assert detail["effective_gguf"] == \
        "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    by_name = {x["filename"]: x for x in detail["variants"]}
    lone = by_name["qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf"]
    assert lone["complete"] is False
    assert lone["is_effective"] is False
    assert lone["missing_shards"] == [1]
    # COMPLETE variants stay lean on the wire — `complete` is omit-when-true so
    # the common row does not grow four keys for nothing.
    assert "complete" not in by_name["qwen2.5-7b-instruct-q2_k.gguf"]
    # Exactly one effective variant, and the litter is still LISTED (the
    # operator needs to see it to reclaim the space) — just not electable.
    assert sum(1 for x in detail["variants"] if x["is_effective"]) == 1
    # 13 files -> 7 variants: fp16(4 shards), q4_k_m(2), q4_0(1 of 2, litter),
    # q5_k_m(2), q6_k(2), q2_k(1), q3_k_m(1).
    assert len(detail["variants"]) == 7

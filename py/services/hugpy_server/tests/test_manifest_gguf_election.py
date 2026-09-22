"""The /manifest transfer set collapses a multi-quant GGUF repo to ONE quant.

Regression for the 2026-08-31 "20x waste" incident: an imatrix repo like
``mradermacher~DAN-L3-R1-8B-i1-GGUF`` holds a dozen-plus quant files (~87 GB),
but a call serves exactly ONE quant (~4.6 GB). ``format_select.select_files``
passes GGUF through untouched, so the /manifest route used to offer every quant
— the worker pulled the whole repo and the budget need was the whole-repo sum.
``_elect_gguf_transfer_set`` collapses it to the elected (or pinned) quant +
projector + sidecars, mirroring the serve loader's own election.
"""
from hugpy_server.app.routes.worker_routes import _elect_gguf_transfer_set as E

GB = 2 ** 30


def _dan_repo():
    return [
        ("DAN-L3-R1-8B.i1-IQ2_XXS.gguf", int(2.4 * GB)),
        ("DAN-L3-R1-8B.i1-IQ3_M.gguf", int(3.6 * GB)),
        ("DAN-L3-R1-8B.i1-Q4_K_M.gguf", int(4.6 * GB)),
        ("DAN-L3-R1-8B.i1-Q4_K_S.gguf", int(4.3 * GB)),
        ("DAN-L3-R1-8B.i1-Q6_K.gguf", int(6.6 * GB)),
        ("DAN-L3-R1-8B.i1-Q8_0.gguf", int(8.5 * GB)),
        ("DAN-L3-R1-8B.i1-f16.gguf", int(16.0 * GB)),
        ("mmproj-DAN-f16.gguf", int(0.6 * GB)),
        ("config.json", 900),
        ("README.md", 4000),
    ]


def _names(entries):
    return [r for (r, _s) in entries]


def test_no_pin_elects_single_quant_plus_projector_and_sidecars():
    out = E(_dan_repo(), {})
    names = _names(out)
    # q4_k_m is QUANT_ORDER's first preference — the elected quant.
    assert names.count("DAN-L3-R1-8B.i1-Q4_K_M.gguf") == 1
    # every OTHER quant is dropped from the transfer set (full filenames — the
    # projector mmproj-DAN-f16.gguf legitimately carries "f16" and must stay).
    for dropped in ("DAN-L3-R1-8B.i1-IQ2_XXS.gguf", "DAN-L3-R1-8B.i1-IQ3_M.gguf",
                    "DAN-L3-R1-8B.i1-Q4_K_S.gguf", "DAN-L3-R1-8B.i1-Q6_K.gguf",
                    "DAN-L3-R1-8B.i1-Q8_0.gguf", "DAN-L3-R1-8B.i1-f16.gguf"):
        assert dropped not in names, dropped
    # projector + non-gguf sidecars always ride along.
    assert "mmproj-DAN-f16.gguf" in names
    assert "config.json" in names and "README.md" in names
    # the served size is one quant + projector, not the whole repo.
    assert sum(s for (_r, s) in out) == int(4.6 * GB) + int(0.6 * GB) + 900 + 4000


def test_filename_pin_wins_over_election():
    out = E(_dan_repo(), {"filename": "DAN-L3-R1-8B.i1-Q6_K.gguf"})
    names = _names(out)
    assert "DAN-L3-R1-8B.i1-Q6_K.gguf" in names
    assert "DAN-L3-R1-8B.i1-Q4_K_M.gguf" not in names
    assert "mmproj-DAN-f16.gguf" in names


def test_include_glob_selects_matching_quant():
    out = E(_dan_repo(), {"include": ["*IQ3_M*"]})
    names = _names(out)
    assert any("IQ3_M" in n for n in names)
    assert not any("Q4_K_M" in n for n in names)


def test_sharded_quant_transfers_all_member_shards():
    # A split GGUF is N shards that are ONE model — the elected variant must
    # bring its whole shard set, never just the entrypoint shard.
    repo = [
        ("big.Q4_K_M-00001-of-00003.gguf", int(15 * GB)),
        ("big.Q4_K_M-00002-of-00003.gguf", int(15 * GB)),
        ("big.Q4_K_M-00003-of-00003.gguf", int(15 * GB)),
        ("big.Q2_K.gguf", int(10 * GB)),
        ("config.json", 500),
    ]
    names = _names(E(repo, {}))
    shards = [n for n in names if "Q4_K_M-000" in n]
    assert len(shards) == 3, shards          # all three shards ride
    assert "big.Q2_K.gguf" not in names      # the loser quant is dropped


def test_non_gguf_listing_is_untouched():
    tf = [("model.safetensors", int(5 * GB)), ("config.json", 900)]
    assert E(tf, {}) == tf


def test_projector_only_repo_is_left_unchanged():
    # No quant to elect (only a projector) -> never under-offer; pass through.
    only = [("mmproj-x-f16.gguf", int(0.6 * GB)), ("config.json", 200)]
    assert E(only, {}) == only

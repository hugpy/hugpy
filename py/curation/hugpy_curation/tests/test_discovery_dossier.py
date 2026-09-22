"""k120 — the dossier package, offline.

Nothing here touches the network, the GPU, the shared store or the hub. Every
external read is either mocked or hard-disabled (``DOSSIER_OFFLINE``), which is
the point: the machinery that decides whether the fleet adopts a model has to be
provable on a laptop.

The rules under test, in the order they matter:

  * a verdict may not exist without evidence, and a blocked trial must NAME its
    cause (``test_verdict_*``, ``test_blocked_trial_*``);
  * every k120 card knob defaults to the pre-k120 behaviour, so an existing
    criteria file behaves identically (``test_card_knob_*``);
  * a card README is read as CLAIMS, never as measurements, and a card with no
    benchmark table produces no benchmark claims (``test_card_*``);
  * a VRAM estimate that cannot include the KV cache SAYS SO (``test_quant_*``);
  * an unreachable source is a recorded row, not a missing one
    (``test_*_unavailable``).

Run:
  cd py/curation/hugpy_curation
  ./venv/bin/python -m pytest tests/test_discovery_dossier.py -q
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import pytest

logging.disable(logging.INFO)

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ.setdefault("DOSSIER_OFFLINE", "1")       # belt: no accidental fetch
# Pin the summarizer/judge model so nothing in this file resolves it from the
# live catalog or the routing matrix. Every test that exercises a model-written
# section supplies its own `dispatch` callable; the pin only decides what name
# the output is STAMPED with, and stamping it "test-judge" is exactly right.
os.environ.setdefault("DOSSIER_LLM_MODEL", "test-judge")

from hugpy_curation.dossier import cards
from hugpy_curation.dossier import community
from hugpy_curation.dossier import radar
from hugpy_curation.dossier import research
from hugpy_curation.dossier import screening
from hugpy_curation.dossier import store
from hugpy_curation.dossier import trial
from hugpy_curation.dossier import verdicts
from hugpy_curation.dossier import weights
from hugpy_curation.dossier.build import build_identity, build_trust
from hugpy_curation.dossier.dossier import (
    Community,
    IncumbentComparison,
    Mention,
    ModelDossier,
    SampleScore,
    Source,
    TRIAL_DEPTHS,
    TrialEvidence,
    TrustSignals,
    WeightsFacts,
)


# ---------------------------------------------------------------------------
# Fixtures — two model cards, one with a benchmark table and one without
# ---------------------------------------------------------------------------

CARD_WITH_TABLE = """---
license: apache-2.0
language:
  - en
  - zh
base_model: Qwen/Qwen3-8B
---

# Nightwatch-8B-Coder

Nightwatch-8B is fine-tuned for code generation and repository-scale reasoning.
It was trained on permissively licensed source and instruction data.

## Evaluation

| Benchmark  | Score |
|------------|-------|
| HumanEval  | 78.4  |
| MMLU       | 66.1  |
| GSM8K      | 71.0  |

## Training data

Trained on 40B tokens of permissively licensed source code plus a 2M-row
instruction mixture. No proprietary corpora were used.

## Limitations

The model refuses very little and should not be deployed without a filter. It
has not been evaluated for languages other than English and Chinese.

Paper: https://arxiv.org/abs/2401.12345
"""

CARD_NO_TABLE = """
# Someone's Q4 quant of Nightwatch

GGUF quants of Nightwatch-8B-Coder. Made with llama.cpp b4321.
Use the Q4_K_M unless you have the VRAM for Q6_K.
"""

CARD_COLUMN_TABLE = """
## Results

| Model        | MMLU | GSM8K |
|--------------|------|-------|
| Nightwatch-8B| 66.1 | 71.0  |
| Baseline-8B  | 61.0 | 63.2  |
"""

QWEN_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 36, "hidden_size": 4096,
    "num_attention_heads": 32, "num_key_value_heads": 8,
    "max_position_embeddings": 32768, "vocab_size": 151936,
    "torch_dtype": "bfloat16", "tokenizer_class": "Qwen2Tokenizer",
}

SCREEN_ROW = {
    "hub_id": "someone/Nightwatch-8B-Coder-GGUF",
    "passed": True, "score": 3.2, "reasons": [], "notes": [],
    "params": 8_000_000_000, "context_length": 32768,
    "architecture": "Qwen3ForCausalLM", "total_bytes": 5_000_000_000,
    "quants": [
        {"quant": "Q4_K_M", "bytes": 4_900_000_000, "files": ["a-Q4_K_M.gguf"]},
        {"quant": "Q6_K", "bytes": 6_600_000_000, "files": ["a-Q6_K.gguf"]},
        {"quant": "Q8_0", "bytes": 8_500_000_000, "files": ["a-Q8_0.gguf"]},
    ],
    "best_quant": "Q6_K", "task": "text-generation",
    "tags": ["gguf", "code", "text-generation"],
    "downloads": 4200, "likes": 61, "trust_tier": 1,
    "base_model": "Qwen/Qwen3-8B", "license": "apache-2.0", "gated": False,
    "last_modified": "2026-08-01T00:00:00Z", "age_days": 20,
}

PAYLOAD = {
    "pipeline_tag": "text-generation",
    "tags": ["gguf", "code", "base_model:quantized:Qwen/Qwen3-8B"],
    "downloads": 4200, "likes": 61, "license": "apache-2.0",
    "card_data": {"language": ["en", "zh"]},
}


# ---------------------------------------------------------------------------
# 1-6  the model card
# ---------------------------------------------------------------------------


def test_card_with_table_yields_benchmark_claims():
    digest = cards.parse_card(CARD_WITH_TABLE)
    names = {c.benchmark for c in digest.benchmark_claims}
    assert {"HumanEval", "MMLU", "GSM8K"} <= names
    assert digest.benchmark_tables == 1
    values = {c.benchmark: c.value for c in digest.benchmark_claims}
    assert values["HumanEval"] == "78.4"


def test_card_without_table_yields_no_claims():
    digest = cards.parse_card(CARD_NO_TABLE)
    assert digest is not None
    assert digest.benchmark_claims == ()
    assert digest.benchmark_tables == 0


def test_card_column_oriented_table_is_read():
    digest = cards.parse_card(CARD_COLUMN_TABLE)
    names = {c.benchmark for c in digest.benchmark_claims}
    assert "MMLU" in names and "GSM8K" in names
    # the row subject rides along as the comparator, so a table comparing two
    # models is not silently attributed to the candidate
    assert any(c.comparator == "Baseline-8B" for c in digest.benchmark_claims)


def test_card_sections_and_limitations():
    digest = cards.parse_card(CARD_WITH_TABLE)
    assert "refuses very little" in digest.limitations
    assert "permissively licensed source code" in digest.training_data
    assert "Nightwatch-8B is fine-tuned" in digest.summary


def test_no_card_is_none_not_an_empty_digest():
    assert cards.parse_card(None) is None
    assert cards.parse_card("   ") is None


def test_arxiv_ids_from_card():
    assert cards.arxiv_ids(CARD_WITH_TABLE) == ("2401.12345",)
    assert cards.arxiv_ids("no papers here") == ()


# ---------------------------------------------------------------------------
# 7-10  specialization + the emphasis weights
# ---------------------------------------------------------------------------


def test_specialization_reads_domain_and_languages():
    spec = cards.build_specialization(
        "someone/Nightwatch-8B-Coder-GGUF", PAYLOAD,
        cards.parse_card(CARD_WITH_TABLE), card_text=CARD_WITH_TABLE)
    assert "code" in spec.domains
    assert set(spec.languages) == {"en", "zh"}
    assert spec.modality == "text"
    assert spec.headline


def test_emphasis_weights_carry_their_evidence():
    rows = cards.build_emphasis(
        "someone/Nightwatch-8B-Coder-GGUF", ["code"], CARD_WITH_TABLE,
        "text-generation")
    code = next(r for r in rows if r.domain == "code")
    assert 0.0 < code.weight <= 1.0
    assert code.evidence, "a weight with no evidence is a number without a source"
    # name + tag + card all agree, so it outranks a domain with one weak signal
    assert code.weight >= 0.5


def test_specialization_says_so_when_there_is_no_card():
    spec = cards.build_specialization("org/thing-7B", {"tags": []}, None,
                                      card_text="")
    assert any("no model card text" in n for n in spec.notes)
    assert spec.languages == ()          # never guessed from silence


def test_focus_sentences_are_verbatim():
    quotes = cards.focus_sentences(CARD_WITH_TABLE)
    assert quotes
    assert quotes[0] in " ".join(CARD_WITH_TABLE.split())


# ---------------------------------------------------------------------------
# 11-15  weights + quant VRAM
# ---------------------------------------------------------------------------


def test_quant_vram_includes_kv_and_overhead():
    facts = weights.quant_facts(SCREEN_ROW["quants"], QWEN_CONFIG,
                                context=16384,
                                vram_budget_bytes=23 * 1024 ** 3)
    q4 = next(q for q in facts if q.quant == "Q4_K_M")
    assert q4.est_kv_bytes and q4.est_kv_bytes > 0
    assert q4.est_vram_bytes > q4.bytes          # weights alone is not the answer
    assert q4.est_vram_bytes == q4.bytes + q4.est_kv_bytes + 1_200 * 1024 ** 2


def test_quant_ladder_orders_by_size_and_flags_fit():
    facts = weights.quant_facts(SCREEN_ROW["quants"], QWEN_CONFIG,
                                context=16384,
                                vram_budget_bytes=8 * 1024 ** 3)
    assert [q.quant for q in facts] == ["Q4_K_M", "Q6_K", "Q8_0"]
    assert facts[0].fits_vram is True
    assert facts[-1].fits_vram is False          # 8.5 GB of weights in 8 GiB


def test_quant_without_geometry_says_the_estimate_reads_low():
    facts = weights.quant_facts(SCREEN_ROW["quants"], {}, context=16384,
                                vram_budget_bytes=23 * 1024 ** 3)
    assert facts[0].est_kv_bytes is None
    assert "EXCLUDES it" in facts[0].note


def test_bits_per_weight_marks_the_guessed_ones():
    exact, was_exact = weights.bits_per_weight("Q4_K_M")
    assert exact == 4.85 and was_exact is True
    guessed, was_exact = weights.bits_per_weight("Q7_WEIRD")
    assert guessed == 7.5 and was_exact is False
    assert weights.bits_per_weight(None) == (None, False)


def test_params_from_name_handles_moe_active_counts():
    total, active = weights.params_from_name("org/Qwen3.6-35B-A3B-Something")
    assert total == 35_000_000_000
    assert active == 3_000_000_000


def test_build_weights_records_where_the_param_count_came_from():
    facts = weights.build_weights(
        "someone/Nightwatch-8B-Coder-GGUF", SCREEN_ROW, QWEN_CONFIG,
        vram_budget_bytes=23 * 1024 ** 3, target_context=16384)
    assert facts.params == 8_000_000_000
    assert facts.params_source == "safetensors"
    assert facts.architecture_family == "qwen3"
    assert facts.tokenizer == "Qwen2Tokenizer"
    assert len(facts.quants) == 3


# ---------------------------------------------------------------------------
# 16-17  lineage
# ---------------------------------------------------------------------------


def test_lineage_extraction_names_the_relation():
    ident = build_identity("someone/Nightwatch-8B-Coder-GGUF", PAYLOAD,
                           SCREEN_ROW)
    assert ident.org == "someone"
    assert ident.base_model == "Qwen/Qwen3-8B"
    assert ident.lineage == ("Qwen/Qwen3-8B",)
    assert ident.relation == "quantization"
    assert ident.is_derivative is True


def test_lineage_of_an_original_repo_is_empty_not_guessed():
    ident = build_identity("Qwen/Qwen3-8B", {"tags": []},
                           {"hub_id": "Qwen/Qwen3-8B"})
    assert ident.base_model is None
    assert ident.lineage == ()
    assert ident.relation is None


# ---------------------------------------------------------------------------
# 18-21  card-config knobs
# ---------------------------------------------------------------------------


def test_card_knob_defaults_preserve_pre_k120_behaviour():
    from hugpy_curation.review.criteria import ReviewCriteria
    crit = ReviewCriteria(name="legacy")
    assert crit.trial_depth == "load-test"
    assert crit.sample_count == 2
    assert crit.compare_against == []
    assert crit.required_specializations == []
    assert crit.licenses_allowed == []
    assert crit.external_research is True
    assert crit.community is True
    assert crit.radar is False


def test_card_knob_vocabulary_matches_the_dossier_package():
    """The duplication in criteria.py is deliberate (it must not import the
    dossier package); this test is what stops it drifting."""
    from hugpy_curation.review.criteria import TRIAL_DEPTHS as CARD_DEPTHS
    assert tuple(CARD_DEPTHS) == tuple(TRIAL_DEPTHS)


def test_card_knob_bad_trial_depth_fails_at_load():
    from hugpy_curation.review.criteria import ReviewCriteria
    with pytest.raises(ValueError) as exc:
        ReviewCriteria(name="typo", trial_depth="full_samples")
    assert "full_samples" in str(exc.value)


def test_card_knob_old_criteria_file_loads_unchanged():
    """A July criteria file has none of these fields. It must load, and it must
    read as the defaults — that is the whole additivity promise."""
    from hugpy_curation.review.criteria import ReviewCriteria
    legacy = {"name": "nightly", "query": "qwen3", "task": "text-generation",
              "min_downloads": 500, "max_age_days": "120"}
    crit = ReviewCriteria.from_dict(legacy)
    assert crit.max_age_days == 120           # the old string coercion still works
    assert crit.trial_depth == "load-test"
    assert crit.dossier is True


# ---------------------------------------------------------------------------
# 22-24  the screen knobs (no network)
# ---------------------------------------------------------------------------


def test_screen_knob_licences_empty_allows_everything():
    ok, why = screening.licence_allowed("cc-by-nc-4.0", ())
    assert ok and why == ""


def test_screen_knob_licence_rejects_with_a_reason():
    ok, why = screening.licence_allowed("cc-by-nc-4.0", ["apache", "mit"])
    assert not ok
    assert "cc-by-nc-4.0" in why and "licenses_allowed" in why
    ok, _ = screening.licence_allowed("apache-2.0", ["apache"])
    assert ok


def test_screen_knob_required_specialization_names_what_it_found():
    reasons = screening.extra_reasons(
        "org/generic-7B", {"required_specializations": ["code"]},
        "apache-2.0", ["text-generation"], "text-generation")
    assert len(reasons) == 1
    assert "code" in reasons[0]
    assert screening.extra_reasons(
        "org/deepseek-coder-7B", {"required_specializations": ["code"]},
        "apache-2.0", ["code"], "text-generation") == []


# ---------------------------------------------------------------------------
# 25-29  the verdict rule
# ---------------------------------------------------------------------------


def _dossier_with(trial_evidence, license_id="apache-2.0"):
    return ModelDossier(
        hub_id="someone/Nightwatch-8B-Coder-GGUF", criteria="test",
        trust=TrustSignals(license=license_id, downloads=4200, trust_tier=1),
        weights=WeightsFacts(params=8_000_000_000,
                             vram_budget_bytes=23 * 1024 ** 3,
                             target_context=16384),
        trial=trial_evidence)


def test_verdict_requires_evidence():
    """No trial data -> the ONLY permitted verdict is screened-only, and it must
    name the cause."""
    blocked = trial.blocked("full-samples",
                            "download failed: HfHubHTTPError: 429")
    verdict = verdicts.rule_verdict(_dossier_with(blocked))
    assert verdict.verdict == "screened-only"
    assert verdict.confidence == "screened-only"
    assert "trial blocked" in verdict.reasons[0]
    assert "429" in verdict.reasons[0]
    assert verdict.evidence_refs == ("trial.blocked",)


def test_verdict_with_a_win_is_evidence_backed_and_cites_it():
    evidence = TrialEvidence(
        depth="full-samples", backend="dispatch",
        scores=(SampleScore(operation="plot.construct", ok=True,
                            deterministic=80.0, judge=76.0, quality=78.0),),
        comparisons=(IncumbentComparison(
            operation="plot.construct", incumbent="Qwen3.8_4B_Distilled_GGUF",
            incumbent_quality=61.0, candidate_quality=78.0, margin=17.0,
            beats_incumbent="yes", basis="routing-matrix scale"),))
    verdict = verdicts.rule_verdict(_dossier_with(evidence))
    assert verdict.verdict == "adopt"
    assert verdict.confidence == "evidence-backed"
    assert any("trial.comparisons[0]" == r for r in verdict.evidence_refs)
    assert any("Qwen3.8_4B_Distilled_GGUF" in r for r in verdict.reasons)


def test_verdict_rejects_when_every_sample_failed():
    evidence = TrialEvidence(
        depth="full-samples", backend="dispatch",
        scores=(SampleScore(operation="plot.construct", ok=False,
                            failure="timeout"),))
    verdict = verdicts.rule_verdict(_dossier_with(evidence))
    assert verdict.verdict == "reject"
    assert any("did not produce a valid artifact" in r for r in verdict.reasons)


def test_verdict_restrictive_licence_downgrades_to_trial():
    evidence = TrialEvidence(
        depth="full-samples", backend="dispatch",
        scores=(SampleScore(operation="plot.construct", ok=True, quality=70.0),))
    verdict = verdicts.rule_verdict(
        _dossier_with(evidence, license_id="cc-by-nc-4.0"))
    assert verdict.verdict == "trial"
    assert "trust.license" in verdict.evidence_refs


def test_judge_that_cites_nothing_cannot_set_the_verdict():
    """A judge with an opinion and no citation is recorded as a REASON on the
    rule verdict; it never becomes the verdict itself."""
    evidence = TrialEvidence(
        depth="full-samples", backend="dispatch",
        scores=(SampleScore(operation="plot.construct", ok=True, quality=70.0),),
        comparisons=(IncumbentComparison(
            operation="plot.construct", incumbent="incumbent-a",
            incumbent_quality=80.0, candidate_quality=70.0, margin=-10.0,
            beats_incumbent="no"),))
    reply = json.dumps({"verdict": "adopt", "reasons": ["it feels strong"],
                        "evidence": ["vibes"], "summary": "great model"})
    verdict = verdicts.judge_verdict(
        _dossier_with(evidence),
        dispatch=lambda model, prompt, max_tokens: reply)
    assert verdict.verdict != "adopt"
    assert any("cited no dossier fact" in r for r in verdict.reasons)


def test_judge_adopt_is_downgraded_without_a_measured_win():
    evidence = TrialEvidence(
        depth="full-samples", backend="dispatch",
        scores=(SampleScore(operation="plot.construct", ok=True, quality=70.0),),
        comparisons=(IncumbentComparison(
            operation="plot.construct", incumbent="incumbent-a",
            incumbent_quality=80.0, candidate_quality=70.0, margin=-10.0,
            beats_incumbent="no"),))
    reply = json.dumps({
        "verdict": "adopt",
        "reasons": ["it scored 70 on plot.construct"],
        "evidence": ["trial.scores[0]"], "summary": "solid"})
    verdict = verdicts.judge_verdict(
        _dossier_with(evidence),
        dispatch=lambda model, prompt, max_tokens: reply)
    assert verdict.verdict == "trial"
    assert any("downgraded to trial" in r for r in verdict.reasons)


def test_evidence_facts_only_expose_what_can_be_cited():
    evidence = TrialEvidence(
        depth="full-samples", backend="dispatch",
        scores=(SampleScore(operation="plot.construct", ok=True, quality=70.0),))
    facts = verdicts.evidence_facts(_dossier_with(evidence))
    assert "trial.scores[0]" in facts
    assert "trust.license" in facts
    assert "identity.hub_id" not in facts or facts["identity.hub_id"]


# ---------------------------------------------------------------------------
# 30-32  the trial: depths, blocking and the incumbent comparison
# ---------------------------------------------------------------------------


def test_blocked_trial_screen_only_says_which_depth():
    evidence = trial.run_trial("org/x", modality="text", depth="screen-only")
    assert evidence.has_evidence is False
    assert "screen-only" in evidence.blocked
    assert evidence.backend == "none"


def test_blocked_trial_gated_repo_is_named():
    evidence = trial.run_trial("org/x", modality="text", depth="full-samples",
                               gated=True)
    assert "gated repo" in evidence.blocked
    assert evidence.has_evidence is False


def test_load_test_depth_keeps_the_load_and_says_no_samples_ran():
    evidence = trial.run_trial(
        "org/x", modality="text", depth="load-test",
        load={"ok": True, "gen_tokens_per_sec": 22.5})
    assert evidence.has_evidence is True         # a successful load IS evidence
    assert "no sample battery" in evidence.blocked
    assert evidence.load["gen_tokens_per_sec"] == 22.5


def test_incumbent_comparison_untested_without_a_matrix():
    evidence = TrialEvidence(
        depth="full-samples", backend="local-gguf",
        scores=(SampleScore(operation="plot.construct", ok=True, quality=64.0),))

    class _NoMatrix:
        entries = ()

        def entry(self, _operation):
            return None

    rows = trial.compare_to_incumbent(evidence, matrix=_NoMatrix())
    assert len(rows) == 1
    assert rows[0].beats_incumbent == "untested"
    assert "no measured primary" in rows[0].basis


def test_incumbent_comparison_computes_the_margin():
    from hugpy_oracle.routing_matrix import Candidate, RouteEntry, RoutingMatrix
    incumbent = Candidate(model="incumbent-a", ok_rate=1.0, quality=61.0,
                          attempts=2)
    matrix = RoutingMatrix(entries=(RouteEntry(
        operation="plot.construct", primary="incumbent-a",
        candidates=(incumbent,)),), registry_version="sha256:test")
    evidence = TrialEvidence(
        depth="full-samples", backend="local-gguf",
        scores=(SampleScore(operation="plot.construct", ok=True, quality=78.0),))
    rows = trial.compare_to_incumbent(evidence, matrix=matrix)
    assert rows[0].incumbent == "incumbent-a"
    assert rows[0].margin == 17.0
    assert rows[0].beats_incumbent == "yes"
    assert "no fleet route" in rows[0].basis     # local-gguf basis is disclosed


def test_candidate_quality_uses_the_matrix_scale_when_cells_exist():
    cells = [{"operation": "plot.construct", "model": "cand", "ok": True,
              "deterministic": {"score": 80.0},
              "judge": {"score": 70.0, "available": True},
              "perf": {"latency_s": 3.0}}]
    value = trial.candidate_quality("plot.construct", (), cells)
    assert value == 75.0                          # (80 + 70) / 2, quality_of's rule


# ---------------------------------------------------------------------------
# 33-38  community intelligence
# ---------------------------------------------------------------------------


def _reddit_payload(title, body, score=42, age_days=2.0):
    return {"data": {"children": [{"data": {
        "permalink": "/r/LocalLLaMA/comments/abc/def/",
        "title": title, "selftext": body, "author": "someone",
        "created_utc": time.time() - age_days * 86400, "score": score}}]}}


def _hn_payload(title, text, points=12, age_days=1.0):
    return {"hits": [{"objectID": "999", "title": title,
                      "comment_text": text, "author": "hn_user",
                      "created_at_i": int(time.time() - age_days * 86400),
                      "points": points}]}


def _fake_fetch(mapping):
    class _R:
        def __init__(self, url, payload):
            self.url, self.ok, self.status = url, payload is not None, 200
            self.text = json.dumps(payload) if payload is not None else ""
            self.detail = "" if payload is not None else "HTTP 403 Forbidden"
            self.from_cache = False

        def json(self):
            return json.loads(self.text) if self.text else None

    def fetch(url, **_kw):
        for needle, payload in mapping.items():
            if needle in url:
                return _R(url, payload)
        return _R(url, None)
    return fetch


def test_reddit_source_yields_typed_mentions(monkeypatch):
    payload = _reddit_payload("Nightwatch-8B-Coder is quietly excellent",
                              "Been running the Q4_K_M all week.")
    monkeypatch.setattr(community, "fetch",
                        _fake_fetch({"reddit.com": payload}))
    reading = community.reddit_source(community.MentionQuery(
        hub_id="someone/Nightwatch-8B-Coder-GGUF",
        aliases=("Nightwatch-8B-Coder",), subreddits=("LocalLLaMA",)))
    assert len(reading.mentions) == 1
    m = reading.mentions[0]
    assert m.source == "reddit:LocalLLaMA"
    assert m.url.startswith("https://www.reddit.com/r/LocalLLaMA/")
    assert m.score == 42
    assert reading.sources[0].ok is True


def test_reddit_source_unavailable_is_a_recorded_row(monkeypatch):
    monkeypatch.setattr(community, "fetch", _fake_fetch({}))   # everything 403s
    reading = community.reddit_source(community.MentionQuery(
        hub_id="org/x", aliases=("x-9B",), subreddits=("LocalLLaMA",)))
    assert reading.mentions == ()
    assert len(reading.sources) == 1
    assert reading.sources[0].ok is False
    assert "403" in reading.sources[0].detail


def test_hackernews_source_strips_html(monkeypatch):
    monkeypatch.setattr(community, "fetch", _fake_fetch({
        "hn.algolia.com": _hn_payload(
            "Show HN: local models",
            "<p>Nightwatch-8B-Coder needs a <b>setuptools</b> pin.</p>")}))
    reading = community.hackernews_source(community.MentionQuery(
        hub_id="someone/Nightwatch-8B-Coder-GGUF",
        aliases=("Nightwatch-8B-Coder",)))
    assert "<p>" not in reading.mentions[0].snippet
    assert "setuptools" in reading.mentions[0].snippet


def test_youtube_source_is_honestly_unavailable():
    reading = community.youtube_source(community.MentionQuery(hub_id="org/x"))
    assert reading.mentions == ()
    assert reading.sources[0].ok is False
    assert "youtube-transcript-api" in reading.sources[0].detail


def test_aliases_strip_packaging_suffixes():
    aliases = community.aliases_for("bartowski/Nightwatch-8B-Coder-GGUF")
    assert aliases[0] == "Nightwatch-8B-Coder-GGUF"
    assert "Nightwatch-8B-Coder" in aliases


def test_heat_is_recency_weighted():
    fresh = [Mention(source="reddit", ts=time.time(), score=10)]
    stale = [Mention(source="reddit", ts=time.time() - 120 * 86400, score=10)]
    assert community.heat_of(fresh) > community.heat_of(stale) * 3
    assert community.heat_of([]) == 0.0


def test_irrelevant_hits_are_dropped():
    aliases = ("Nightwatch-8B-Coder",)
    hit = Mention(source="reddit", title="Qwen3 is great", snippet="")
    miss = Mention(source="reddit", title="Nightwatch-8B-Coder review",
                   snippet="")
    assert community.relevant(hit, aliases) is False
    assert community.relevant(miss, aliases) is True


def test_claims_without_a_supporting_quote_are_dropped():
    mentions = [Mention(source="reddit", url="https://r/1",
                        title="Nightwatch review",
                        snippet="It needs a setuptools pin below 81.")]
    reply = json.dumps({"claims": [
        {"kind": "quirk", "text": "needs a setuptools pin",
         "quote": "It needs a setuptools pin below 81.",
         "url": "https://r/1"},
        {"kind": "benchmark", "text": "scores 92 on HumanEval",
         "quote": "it scores 92 on HumanEval", "url": "https://r/1"},
    ]})
    claims, model, _detail = community.extract_claims(
        "Nightwatch", mentions,
        dispatch=lambda m, p, t: reply)
    assert len(claims) == 1                      # the invented one is gone
    assert claims[0].kind == "quirk"
    assert claims[0].url == "https://r/1"


def test_gather_folds_sources_and_records_the_unavailable(monkeypatch):
    monkeypatch.setattr(community, "fetch", _fake_fetch({
        "reddit.com": _reddit_payload("Nightwatch-8B-Coder is good", "x"),
    }))                                          # HN and discussions will fail
    found, urls = community.gather(
        "someone/Nightwatch-8B-Coder-GGUF",
        sources=("reddit", "hackernews", "youtube"),
        subreddits=("LocalLLaMA",), want_claims=False)
    assert found.heat > 0
    assert len(found.mentions) == 1
    assert any("hackernews" in u for u in found.unavailable)
    assert any("youtube" in u for u in found.unavailable)
    assert urls


def test_gather_names_an_unknown_source_rather_than_ignoring_it():
    found, _urls = community.gather("org/x", sources=("tiktok",),
                                    want_claims=False)
    assert any("unknown community source" in u for u in found.unavailable)


# ---------------------------------------------------------------------------
# 39-42  the gem radar
# ---------------------------------------------------------------------------


def test_radar_harvests_repo_ids_and_bare_names():
    mention = Mention(
        source="reddit:LocalLLaMA",
        title="Try huggingface.co/tinylab/Kestrel-9B-Instruct",
        snippet="Also Foundry-13B is punching above its weight. "
                "See /usr/share/docs for nothing.")
    names = radar.harvest_names(mention)
    assert "tinylab/Kestrel-9B-Instruct" in names
    assert "Foundry-13B" in names
    assert not any(n.startswith("usr/") for n in names)


def test_radar_dedupes_against_known_candidates():
    mentions = [
        Mention(source="reddit", url="https://r/1",
                title="huggingface.co/tinylab/Kestrel-9B-Instruct is great",
                ts=time.time(), score=30),
        Mention(source="reddit", url="https://r/2",
                title="bartowski/Nightwatch-8B-Coder-GGUF still the best",
                ts=time.time(), score=30),
    ]
    hits, provenance = radar.scan(
        known=["bartowski/Nightwatch-8B-Coder-GGUF"], mentions=mentions,
        resolve=False)
    names = {h.name for h in hits}
    assert "tinylab/Kestrel-9B-Instruct" in names
    assert not any("Nightwatch" in n for n in names)
    assert provenance.ok is True


def test_radar_drops_a_quant_of_a_known_model_too():
    """`known` covers the de-packaged alias, so someone else's GGUF of an
    incumbent is not reported as a discovery."""
    mentions = [Mention(source="reddit", url="https://r/1",
                        title="someoneelse/Nightwatch-8B-Coder-GGUF quants up",
                        ts=time.time(), score=50)]
    hits, _p = radar.scan(known=["bartowski/Nightwatch-8B-Coder"],
                          mentions=mentions, resolve=False)
    assert hits == ()


def test_radar_with_no_cached_pulls_says_so():
    hits, provenance = radar.scan(known=[], mentions=[], resolve=False)
    assert hits == ()
    assert "no cached community pulls" in provenance.detail


def test_radar_min_heat_filters_the_noise():
    old = [Mention(source="reddit", url="https://r/1",
                   title="tinylab/Kestrel-9B-Instruct existed once",
                   ts=time.time() - 300 * 86400, score=0)]
    hits, _p = radar.scan(known=[], mentions=old, resolve=False)
    assert hits == ()


# ---------------------------------------------------------------------------
# 43-46  research, sources and round-tripping
# ---------------------------------------------------------------------------


def test_research_disabled_says_it_was_turned_off():
    found, sources, readme = research.build_research(
        "org/x", {}, enabled=False)
    assert readme is None and sources == ()
    assert any("disabled for this card" in u for u in found.unavailable)


def test_arxiv_unavailable_keeps_the_id_and_the_reason(monkeypatch):
    class _R:
        ok, status, text, detail, from_cache = False, None, "", "timed out", False
    monkeypatch.setattr(research, "fetch", lambda *a, **k: _R())
    papers, sources = research.fetch_papers(["2401.12345"])
    assert papers[0].arxiv_id == "2401.12345"
    assert papers[0].title is None
    assert "timed out" in papers[0].unavailable
    assert sources[0].ok is False


def test_arxiv_feed_is_parsed(monkeypatch):
    feed = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Nightwatch: code models at 8B</title>
    <summary>We present Nightwatch, a code model.</summary>
  </entry>
</feed>"""

    class _R:
        ok, status, detail, from_cache = True, 200, "", False
        text = feed
    monkeypatch.setattr(research, "fetch", lambda *a, **k: _R())
    papers, sources = research.fetch_papers(["2401.12345"])
    assert papers[0].title == "Nightwatch: code models at 8B"
    assert "code model" in papers[0].abstract
    assert sources[0].ok is True


def test_research_notes_are_labelled_model_generated():
    card = cards.parse_card(CARD_WITH_TABLE)
    notes, model, cited, _detail = research.write_research_notes(
        "someone/Nightwatch-8B-Coder-GGUF", card, (), PAYLOAD,
        dispatch=lambda m, p, t: "It is a code fine-tune of Qwen3-8B.")
    assert notes == "It is a code fine-tune of Qwen3-8B."
    assert model is not None
    assert any("README" in c for c in cited)


def test_research_notes_none_when_no_model_answers():
    def _refuse(model, prompt, max_tokens):
        raise RuntimeError("no route")
    notes, _model, _cited, detail = research.write_research_notes(
        "org/x", None, (), {}, dispatch=_refuse)
    assert notes is None
    assert "did not answer" in detail or "no route" in detail


def test_unavailable_sources_surface_on_the_dossier():
    dossier = ModelDossier(hub_id="org/x", criteria="test")
    dossier.add_source(Source.unavailable("hf-readme", "no README.md in repo"))
    dossier.community = Community(sources=(
        Source.unavailable("reddit:LocalLLaMA", "HTTP 403 Forbidden"),))
    rows = dossier.unavailable
    assert any("hf-readme" in r for r in rows)
    assert any("403" in r for r in rows)


def test_dossier_round_trips_through_json():
    dossier = ModelDossier(
        hub_id="someone/Nightwatch-8B-Coder-GGUF", criteria="test",
        identity=build_identity("someone/Nightwatch-8B-Coder-GGUF", PAYLOAD,
                                SCREEN_ROW),
        specialization=cards.build_specialization(
            "someone/Nightwatch-8B-Coder-GGUF", PAYLOAD,
            cards.parse_card(CARD_WITH_TABLE), card_text=CARD_WITH_TABLE),
        weights=weights.build_weights(
            "someone/Nightwatch-8B-Coder-GGUF", SCREEN_ROW, QWEN_CONFIG,
            vram_budget_bytes=23 * 1024 ** 3),
        trial=trial.blocked("full-samples", "download failed"))
    dossier.verdict = verdicts.rule_verdict(dossier)
    back = ModelDossier.from_dict(json.loads(dossier.to_json()))
    assert back.hub_id == dossier.hub_id
    assert back.identity.base_model == "Qwen/Qwen3-8B"
    assert len(back.weights.quants) == 3
    assert back.specialization.domains == dossier.specialization.domains
    assert back.verdict.verdict == "screened-only"
    assert back.trial.blocked == "download failed"


def test_store_summary_is_small_and_carries_the_decision(tmp_path):
    os.environ["DOSSIER_DIR"] = str(tmp_path)
    try:
        evidence = TrialEvidence(
            depth="full-samples", backend="dispatch",
            scores=(SampleScore(operation="plot.construct", ok=True,
                                quality=78.0),),
            comparisons=(IncumbentComparison(
                operation="plot.construct", incumbent="incumbent-a",
                incumbent_quality=61.0, candidate_quality=78.0, margin=17.0,
                beats_incumbent="yes"),))
        dossier = _dossier_with(evidence)
        dossier.criteria = "test"
        dossier.verdict = verdicts.rule_verdict(dossier)
        path = store.save(dossier)
        assert path and os.path.isfile(path)
        row = store.summary(dossier, path)
        assert row["verdict"] == "adopt"
        assert row["beats_incumbent"] == "yes"
        assert row["margin"] == 17.0
        assert len(json.dumps(row)) < 4000     # it rides in every review row
        assert store.load("test", dossier.hub_id).hub_id == dossier.hub_id
    finally:
        os.environ.pop("DOSSIER_DIR", None)


def test_fetch_is_hard_offline_in_this_suite():
    """The belt-and-braces check: with DOSSIER_OFFLINE set, no test in this
    file can reach the network even by accident."""
    from hugpy_curation.dossier.fetch import fetch as raw_fetch
    result = raw_fetch("https://example.invalid/nothing", use_cache=False)
    assert result.ok is False
    assert "offline" in result.detail

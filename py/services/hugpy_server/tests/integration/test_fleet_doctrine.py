"""k118 — the environment doctrine: report, snapshot, diff, heartbeat, probe.

The four failures this slice exists for are each asserted here as a test, with
a FAKE report standing in for the box that had them:

  * ffmpeg absent            -> a blocker naming the ASR/TTS tasks it takes down
  * bitsandbytes absent      -> a blocker naming the 4-bit rows
  * diffusers absent         -> a blocker naming text-to-image
  * setuptools 81 in the
    chatterbox profile venv  -> a PIN blocker, and only in that venv

Plus the rule that keeps the whole thing honest and is easy to erode: a check
that could NOT be performed answers unknown, and unknown never clears a worker
and never blocks one.

Everything is offline: two dicts and a tmpdir. No worker, no GPU, no network,
no shared store.

"""
from __future__ import annotations

import json
import logging
import os
import sys

import pytest

logging.disable(logging.INFO)

from hugpy_fleet.doctrine import doctor, doctrine as store
from hugpy_fleet.worker import environment_report as er


# ---------------------------------------------------------------------------
# Fixtures: a reference report (a healthy box) and its doctrine.
# ---------------------------------------------------------------------------


def _report(**over):
    """A COMPLETE, healthy report — the shape /ops/environment returns."""
    base = {
        "schema": "1",
        "worker": "ref",
        "generated_at": "2026-08-21T00:00:00Z",
        "python": "3.13.12",
        "worker_root": "/home/ref/hugpy-worker",
        "pkg_version": "0.1.229",
        "report_digest": "deadbeefdeadbeef",
        "venvs": {
            "main": {
                "python": "/home/ref/hugpy-worker/venv/bin/python",
                "python_version": "3.13.12",
                "error": None,
                "packages": {
                    "torch": "2.13.0", "diffusers": "0.39.0",
                    "transformers": "5.14.1", "bitsandbytes": "0.50.0",
                    "accelerate": "1.14.0", "openai-whisper": "20250625",
                    "numba": "0.66.0", "numpy": "2.4.6",
                    "llama-cpp-python": "0.3.34", "setuptools": "83.0.0",
                    "sentence-transformers": "5.7.0", "keybert": "0.9.0",
                    "pdfplumber": "0.11.10", "beautifulsoup4": "4.15.0",
                    "safetensors": "0.8.0", "huggingface-hub": "1.24.0",
                    "some-random-lib": "1.0.0",
                },
            },
            "chatterbox-tts": {
                "python": "/home/ref/hugpy-worker/envs/chatterbox-tts/bin/python",
                "python_version": "3.13.12",
                "error": None,
                "packages": {"chatterbox-tts": "0.1.7", "setuptools": "80.10.2",
                             "torch": "2.6.0", "torchaudio": "2.6.0"},
            },
        },
        "binaries": {
            "ffmpeg": {"present": True, "path": "/usr/bin/ffmpeg", "version": "6.1.1"},
            "ffprobe": {"present": True, "path": "/usr/bin/ffprobe", "version": "6.1.1"},
            "git": {"present": True, "path": "/usr/bin/git", "version": "2.43.0"},
            "nvidia-smi": {"present": True, "path": "/usr/bin/nvidia-smi",
                           "version": "595.71.05"},
            "bsdtar": {"present": False, "path": None, "version": None},
            "python3": {"present": True, "path": "/usr/bin/python3", "version": "3.12.3"},
        },
        "nvidia": {"driver": "595.71.05", "cuda": "13.2",
                   "gpus": [{"name": "NVIDIA GeForce RTX 3090", "vram_mib": 24576}]},
        "mounts": {"/mnt/llm_storage": {"present": True, "writable": True}},
        "os": {"system": "Linux", "pretty_name": "Ubuntu 24.04.4 LTS",
               "id": "ubuntu", "version_id": "24.04", "release": "6.8.0",
               "machine": "x86_64"},
    }
    base.update(over)
    return base


@pytest.fixture()
def reference():
    return _report()


@pytest.fixture()
def doctrine(reference):
    return store.snapshot(reference, version="test.1", reference="ref")


def _drop_pkg(report, name, venv="main"):
    report["venvs"][venv]["packages"].pop(name, None)
    return report


def _find(findings, dep, venv=None):
    match = [f for f in findings if f.dep == dep
             and (venv is None or f.venv == venv)]
    assert match, f"{dep} not among {[(f.dep, f.venv) for f in findings]}"
    return match[0]


# ---------------------------------------------------------------------------
# 1. Report parsing / shape.
# ---------------------------------------------------------------------------


def test_normalize_dist_folds_the_spellings_of_one_package():
    """chatterbox-tts / Chatterbox_TTS / chatterbox.tts are ONE package; a
    doctrine keyed by raw metadata names would miss half the fleet."""
    for spelling in ("chatterbox-tts", "Chatterbox_TTS", "chatterbox.tts",
                     "  CHATTERBOX--TTS "):
        assert er.normalize_dist(spelling) == "chatterbox-tts"


def test_packages_here_reads_this_interpreter_and_finds_pytest():
    packages = er.packages_here()
    assert isinstance(packages, dict) and packages
    assert "pytest" in packages


def test_report_digest_ignores_the_timestamp(reference):
    """A digest that changed every ten minutes could not answer "did this box
    change?" — which is the only question the heartbeat rider exists to ask."""
    later = _report(generated_at="2999-01-01T00:00:00Z")
    assert er.report_digest(reference) == er.report_digest(later)


def test_report_digest_changes_when_a_package_changes(reference):
    moved = _report()
    moved["venvs"]["main"]["packages"]["torch"] = "9.9.9"
    assert er.report_digest(reference) != er.report_digest(moved)


def test_packages_in_returns_none_for_a_python_that_is_not_there(tmp_path):
    """None, not {}. An empty package list would make the diff report every dep
    as missing and hand the operator a repair plan for a venv that needs
    rebuilding, not pip-installing."""
    assert er.packages_in(str(tmp_path / "nope" / "bin" / "python")) is None
    assert er.packages_in("") is None


def test_packages_in_asks_a_real_interpreter(tmp_path):
    parsed = er.packages_in(sys.executable)
    assert parsed is not None
    assert "pytest" in parsed["packages"]
    assert parsed["python"].startswith(str(sys.version_info[0]))


def test_compact_digest_is_small_and_carries_the_binaries(reference):
    compact = er.compact_digest(reference)
    assert compact["digest"] == reference["report_digest"]
    assert compact["profiles"] == ["chatterbox-tts"]
    assert compact["binaries"]["ffmpeg"] is True
    assert compact["binaries"]["bsdtar"] is False
    assert compact["nvidia"] == {"driver": "595.71.05", "cuda": "13.2"}
    # The rider must never carry the document.
    assert "packages" not in json.dumps(compact)


def test_environment_report_is_cached_then_refreshable(monkeypatch):
    calls = {"n": 0}

    def _build(worker_name=None):
        calls["n"] += 1
        return {"schema": "1", "report_digest": f"d{calls['n']}"}

    monkeypatch.setattr(er, "build_report", _build)
    er.clear_cache()
    assert er.environment_report()["report_digest"] == "d1"
    assert er.environment_report()["report_digest"] == "d1"   # TTL hit
    assert er.environment_report(refresh=True)["report_digest"] == "d2"
    er.clear_cache()


def test_a_broken_build_never_raises_into_the_caller(monkeypatch):
    def _boom(worker_name=None):
        raise RuntimeError("the probe itself broke")

    monkeypatch.setattr(er, "build_report", _boom)
    er.clear_cache()
    out = er.environment_report()
    assert "RuntimeError" in out["error"]
    er.clear_cache()


# ---------------------------------------------------------------------------
# 2. Pin semantics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version,pin,expected", [
    ("80.10.2", "<81", True),        # the chatterbox seat, satisfied
    ("81.0.0", "<81", False),        # the silent-failure version
    ("83.0.0", "<81", False),
    ("2.4.6", "<2.5", True),         # a-brain: whisper works
    ("2.5.1", "<2.5", False),        # computron: `import whisper` dies
    ("2.5.2", "<2.5", False),        # ae: same
    ("1.14.0", ">=1.0", True),
    ("0.9.0", ">=1.0", False),
    ("2.13.0", "==2.13.0", True),
    ("2.13.0", "!=2.13.0", False),
    ("2.6.0", ">=2.0,<3.0", True),
    ("3.1.0", ">=2.0,<3.0", False),
])
def test_pin_satisfied(version, pin, expected):
    assert store.pin_satisfied(version, pin) is expected


def test_an_empty_pin_is_always_satisfied():
    assert store.pin_satisfied("1.0", None) is True
    assert store.pin_satisfied("1.0", "") is True


def test_an_unknown_version_or_unparseable_pin_is_unknown_not_false():
    """A typo in the doctrine must never manufacture a blocker on a box that is
    fine."""
    assert store.pin_satisfied(None, "<81") is None
    assert store.pin_satisfied("1.0", "roughly 2") is None
    assert store.pin_satisfied("1.0", "<") is None


# ---------------------------------------------------------------------------
# 3. Snapshot + versioning.
# ---------------------------------------------------------------------------


def test_snapshot_records_every_observed_package_in_every_venv(doctrine):
    main = doctrine.entry("pip", "torch", "main")
    assert main is not None and main.version == "2.13.0"
    profile = doctrine.entry("pip", "chatterbox-tts", "profile:chatterbox-tts")
    assert profile is not None and profile.version == "0.1.7"


def test_an_unclassified_package_defaults_to_info(doctrine):
    entry = doctrine.entry("pip", "some-random-lib", "main")
    assert entry.severity == "info"
    assert entry.required_for == ()


def test_snapshot_carries_the_classification_onto_observed_entries(doctrine):
    diffusers = doctrine.entry("pip", "diffusers", "main")
    assert diffusers.severity == "blocker"
    assert "text-to-image" in diffusers.required_for
    assert diffusers.source == "reference"


def test_a_declared_requirement_the_reference_lacks_is_still_in_the_doctrine():
    """bsdtar is absent on every box in the fleet. It still belongs in the
    doctrine — as a declared, version-less, info entry."""
    doc = store.snapshot(_report(), version="t", reference="ref")
    entry = doc.entry("binary", "bsdtar", "any")
    assert entry is not None
    assert entry.source == "declared" and entry.version is None
    assert entry.severity == "info"


def test_absent_binaries_and_mounts_are_not_snapshotted_as_present():
    report = _report()
    report["binaries"]["ffmpeg"] = {"present": False, "path": None, "version": None}
    doc = store.snapshot(report, version="t", reference="ref")
    entry = doc.entry("binary", "ffmpeg", "any")
    assert entry.source == "declared" and entry.version is None


def test_snapshot_freezes_the_driver_and_the_reference_facts(doctrine):
    assert doctrine.entry("driver", "nvidia-driver", "any").version == "595.71.05"
    assert doctrine.entry("driver", "cuda", "any").version == "13.2"
    assert doctrine.reference_facts["os"]["id"] == "ubuntu"
    assert doctrine.reference_digest == "deadbeefdeadbeef"


def test_provisional_and_pending_are_in_the_document_not_a_commit_message():
    doc = store.snapshot(_report(), version="t", reference="a-brain",
                         provisional=True, pending="ae full report")
    assert doc.provisional is True
    assert doc.to_dict()["pending"] == "ae full report"
    assert store.Doctrine.from_dict(doc.to_dict()).provisional is True


def test_save_load_list_versions_roundtrip(doctrine, tmp_path):
    directory = str(tmp_path)
    store.save(doctrine, directory)
    store.save(store.snapshot(_report(), version="test.2"), directory)
    assert store.list_versions(directory) == ["test.1", "test.2"]
    loaded = store.load("test.1", directory)
    assert loaded.version == "test.1"
    assert len(loaded.entries) == len(doctrine.entries)
    assert store.latest(directory).version == "test.2"


def test_a_missing_or_corrupt_doctrine_is_none_never_an_exception(tmp_path):
    assert store.load("nope", str(tmp_path)) is None
    assert store.latest(str(tmp_path)) is None
    assert store.list_versions(str(tmp_path / "nowhere")) == []
    bad = tmp_path / store.FILENAME.format(version="broken")
    bad.write_text("{not json", encoding="utf-8")
    assert store.load("broken", str(tmp_path)) is None


def test_classification_resolves_the_most_specific_venv_first():
    """setuptools is a pinned BLOCKER inside the chatterbox profile and plain
    drift-info in main. One name, two meanings, decided by venv."""
    profile = store.classification_for("pip", "setuptools",
                                       "profile:chatterbox-tts")
    assert profile.pin == "<81" and profile.severity == "blocker"
    assert store.classification_for("pip", "setuptools", "main") is None
    assert store.classification_for("pip", "nothing-here", "main") is None


def test_every_known_requirement_declares_a_legal_severity():
    for req in store.KNOWN_REQUIREMENTS:
        assert req.severity in store.SEVERITIES
        if req.severity == "blocker":
            assert req.required_for, f"{req.name} blocks nothing — why block?"


def test_a_requirement_with_a_bad_severity_or_kind_refuses_to_exist():
    with pytest.raises(ValueError):
        store.Requirement("x", severity="catastrophic")
    with pytest.raises(ValueError):
        store.Requirement("x", kind="vibes")


# ---------------------------------------------------------------------------
# 4. assess — the four failures, plus the classification boundaries.
# ---------------------------------------------------------------------------


def test_the_reference_assessed_against_its_own_doctrine_is_clean(reference, doctrine):
    """No blockers, no warnings — and exactly one INFO: bsdtar, the declared
    requirement the reference itself does not meet. The reference box is a box,
    not scripture, and the assessment says so without blocking anything."""
    result = doctor.assess(reference, doctrine)
    assert result.blockers == ()
    assert result.warnings == ()
    assert result.verdict == doctor.VERDICT_OK
    assert [f.dep for f in result.infos] == ["bsdtar"]
    assert result.ok_count == result.checked - 1


def test_missing_ffmpeg_blocks_asr_and_tts_with_an_apt_line(doctrine):
    """a-brain, 2026-08-17."""
    report = _report()
    report["binaries"]["ffmpeg"] = {"present": False, "path": None, "version": None}
    result = doctor.assess(report, doctrine)
    finding = _find(result.blockers, "ffmpeg")
    assert finding.status == doctor.STATUS_MISSING
    assert "automatic-speech-recognition" in finding.tasks
    assert "text-to-speech" in finding.tasks
    assert finding.repair == "sudo apt install -y ffmpeg"
    assert "automatic-speech-recognition" in result.blocked_tasks()


def test_missing_bitsandbytes_blocks_the_4bit_rows(doctrine):
    """computron, 2026-08-16."""
    result = doctor.assess(_drop_pkg(_report(), "bitsandbytes"), doctrine)
    finding = _find(result.blockers, "bitsandbytes")
    assert finding.tasks == (store.TASK_4BIT,)
    assert finding.repair.endswith("-m pip install bitsandbytes==0.50.0")
    assert "/home/ref/hugpy-worker/venv/bin/python" in finding.repair


def test_missing_diffusers_blocks_text_to_image(doctrine):
    """a-brain t2i, 2026-08-19."""
    result = doctor.assess(_drop_pkg(_report(), "diffusers"), doctrine)
    assert _find(result.blockers, "diffusers").tasks == ("text-to-image",)
    assert result.verdict == doctor.VERDICT_BLOCKED


def test_setuptools_81_in_the_chatterbox_profile_is_a_pin_blocker(doctrine):
    """The silent one: chatterbox imports pkg_resources, gone in setuptools 81,
    and synthesizes nothing rather than raising."""
    report = _report()
    report["venvs"]["chatterbox-tts"]["packages"]["setuptools"] = "81.0.0"
    result = doctor.assess(report, doctrine)
    finding = _find(result.blockers, "setuptools", "profile:chatterbox-tts")
    assert finding.status == doctor.STATUS_PIN
    assert finding.tasks == ("text-to-speech",)
    assert "setuptools<81" in finding.repair
    # aimed at the PROFILE interpreter, not the agent's
    assert "envs/chatterbox-tts/bin/python" in finding.repair


def test_setuptools_83_in_main_is_not_a_finding_at_all(reference, doctrine):
    """The pin is scoped to the profile. main has 83.0.0 and is fine."""
    result = doctor.assess(reference, doctrine)
    assert not [f for f in result.blockers + result.warnings
                if f.dep == "setuptools"]


def test_a_warn_dep_never_lands_in_blockers(doctrine):
    result = doctor.assess(_drop_pkg(_report(), "accelerate"), doctrine)
    finding = _find(result.warnings, "accelerate")
    assert finding.severity == "warn"
    assert not [f for f in result.blockers if f.dep == "accelerate"]
    assert result.verdict == doctor.VERDICT_WARN


def test_version_drift_is_info_even_for_a_blocker_dep(doctrine):
    """Otherwise the first `pip install -U` anywhere lights up the fleet."""
    report = _report()
    report["venvs"]["main"]["packages"]["diffusers"] = "0.40.0"
    result = doctor.assess(report, doctrine)
    finding = _find(result.infos, "diffusers")
    assert finding.status == doctor.STATUS_DRIFT
    assert finding.severity == "info"
    assert result.blockers == ()


def test_an_unreadable_venv_is_unknown_and_only_ever_a_warning(doctrine):
    report = _report()
    report["venvs"]["main"] = {
        "python": "/home/ref/hugpy-worker/venv/bin/python",
        "python_version": None, "packages": None,
        "error": "interpreter did not answer"}
    result = doctor.assess(report, doctrine)
    assert result.blockers == ()
    finding = _find(result.warnings, "diffusers")
    assert finding.status == doctor.STATUS_UNKNOWN
    assert finding.repair == ""


def test_an_absent_env_profile_is_availability_not_breakage(doctrine):
    """A box with no envs/chatterbox-tts does not seat TTS. That is what the
    catalog already reports; it is not a broken worker."""
    report = _report()
    report["venvs"].pop("chatterbox-tts")
    result = doctor.assess(report, doctrine)
    assert result.blockers == ()
    finding = _find(result.warnings, "chatterbox-tts")
    assert finding.status == doctor.STATUS_NO_PROFILE
    assert "not a fault" in finding.detail


def test_an_unmounted_llm_storage_warns_and_does_not_block(doctrine):
    report = _report()
    report["mounts"]["/mnt/llm_storage"] = {"present": False, "writable": False}
    result = doctor.assess(report, doctrine)
    assert result.blockers == ()
    assert _find(result.warnings, "/mnt/llm_storage").status == doctor.STATUS_MISSING


def test_a_report_that_never_described_a_binary_is_unknown_not_missing(doctrine):
    """An older agent that reports no ffmpeg key has not proven it is absent."""
    report = _report()
    report["binaries"].pop("ffmpeg")
    result = doctor.assess(report, doctrine)
    assert result.blockers == ()
    assert _find(result.warnings, "ffmpeg").status == doctor.STATUS_UNKNOWN


def test_verdict_is_derived_from_the_findings(reference, doctrine):
    assert doctor.assess(reference, doctrine).verdict == doctor.VERDICT_OK
    warned = doctor.assess(_drop_pkg(_report(), "accelerate"), doctrine)
    assert warned.verdict == doctor.VERDICT_WARN
    blocked = doctor.assess(_drop_pkg(_report(), "diffusers"), doctrine)
    assert blocked.verdict == doctor.VERDICT_BLOCKED


def test_blockers_for_task_selects_only_that_task(doctrine):
    report = _drop_pkg(_drop_pkg(_report(), "diffusers"), "bitsandbytes")
    result = doctor.assess(report, doctrine)
    assert [f.dep for f in result.blockers_for_task("text-to-image")] == ["diffusers"]
    assert set(result.blocked_tasks()) == {"text-to-image", store.TASK_4BIT}


# ---------------------------------------------------------------------------
# 5. Repair plan + rendering.
# ---------------------------------------------------------------------------


def test_repair_plan_is_shell_lines_blockers_first_and_deduped(doctrine):
    report = _drop_pkg(_drop_pkg(_drop_pkg(_report(), "diffusers"),
                                 "bitsandbytes"), "accelerate")
    plan = doctor.assess(report, doctrine).repair_plan()
    text = "\n".join(plan)
    assert plan[0].startswith("# hugpy worker doctrine test.1")
    assert text.index("--- blockers ---") < text.index("--- warnings ---")
    commands = [line for line in plan if not line.startswith("#")]
    assert len(commands) == len(set(commands))
    assert any("diffusers==0.39.0" in c for c in commands)


def test_a_clean_box_has_nothing_to_repair(reference, doctrine):
    assert "# nothing to repair" in doctor.assess(reference, doctrine).repair_plan()


def test_a_provisional_doctrine_says_so_in_its_repair_plan(reference):
    doc = store.snapshot(reference, version="p", provisional=True,
                         pending="ae full report")
    plan = doctor.assess(reference, doc).repair_plan()
    assert any("PROVISIONAL" in line for line in plan)


def test_render_uses_the_hugpy_doctor_prefixes(doctrine):
    text = doctor.render(doctor.assess(_drop_pkg(_report(), "diffusers"),
                                       doctrine))
    assert "  FAIL  diffusers" in text
    assert "        repair: " in text
    assert "tasks blocked here: text-to-image" in text
    assert doctor.render(doctor.assess(_report(), doctrine)).count("PASS") == 1


# ---------------------------------------------------------------------------
# 6. The heartbeat fold + the k101 probe.
# ---------------------------------------------------------------------------


def test_heartbeat_status_is_compact_and_carries_a_repair_per_task(doctrine):
    status = doctor.assess(_drop_pkg(_report(), "diffusers"),
                           doctrine).heartbeat_status()
    assert status["verdict"] == "blocked"
    assert status["blocked_tasks"] == ["text-to-image"]
    assert "diffusers==0.39.0" in status["repairs"]["text-to-image"]
    assert status["doctrine_version"] == "test.1"
    # compact: the findings themselves stay off the wire
    assert "detail" not in json.dumps(status)


def test_the_worker_folds_both_fields_onto_the_beat(monkeypatch, reference,
                                                    doctrine, tmp_path):
    """The two additive heartbeat fields, exercised through the agent's own
    helpers — without importing the agent (594k lines and a torch prime)."""
    monkeypatch.setattr(er, "build_report", lambda worker_name=None: reference)
    er.clear_cache()
    digest = er.compact_digest()
    assert digest["digest"] == reference["report_digest"]
    store.save(doctrine, str(tmp_path))
    monkeypatch.setenv(store.ENV_DOCTRINE_DIR, str(tmp_path))
    status = doctor.assess(er.environment_report(), store.latest()).heartbeat_status()
    assert status["verdict"] == "ok" and status["blocked_tasks"] == []
    er.clear_cache()


def _probe_module():
    from hugpy_oracle import probes
    return probes


def _seat(name, task, doctrine_status=None):
    worker = {"id": name, "name": name, "task_capabilities": {task: True}}
    if doctrine_status is not None:
        worker["doctrine_status"] = doctrine_status
    return worker


def test_probe_unknown_when_the_registry_is_unreadable():
    probes = _probe_module()
    spec = probes.ProbeSpec(capability="a.b", task="text-to-image", doctrine=True)
    check = probes._check_doctrine(spec, None)
    assert check.status is probes.ProbeStatus.UNKNOWN
    assert "registry unreadable" in check.detail


def test_probe_unknown_when_central_holds_no_doctrine(monkeypatch):
    probes = _probe_module()
    monkeypatch.setattr(probes, "_latest_doctrine", lambda: None)
    spec = probes.ProbeSpec(capability="a.b", task="text-to-image", doctrine=True)
    check = probes._check_doctrine(spec, [_seat("w1", "text-to-image")])
    assert check.status is probes.ProbeStatus.UNKNOWN
    assert "no environment doctrine" in check.detail


def test_probe_unknown_when_no_worker_has_reported_yet(monkeypatch, doctrine):
    """A worker that never answered is not a broken worker."""
    probes = _probe_module()
    monkeypatch.setattr(probes, "_latest_doctrine", lambda: doctrine)
    spec = probes.ProbeSpec(capability="a.b", task="text-to-image", doctrine=True)
    check = probes._check_doctrine(spec, [_seat("w1", "text-to-image")])
    assert check.status is probes.ProbeStatus.UNKNOWN
    assert "has reported a doctrine assessment" in check.detail


def test_probe_fails_when_every_seating_worker_is_blocked(monkeypatch, doctrine):
    probes = _probe_module()
    monkeypatch.setattr(probes, "_latest_doctrine", lambda: doctrine)
    blocked = doctor.assess(_drop_pkg(_report(), "diffusers"),
                            doctrine).heartbeat_status()
    spec = probes.ProbeSpec(capability="a.b", task="text-to-image", doctrine=True)
    check = probes._check_doctrine(spec, [_seat("w1", "text-to-image", blocked)])
    assert check.status is probes.ProbeStatus.FAIL
    assert "diffusers==0.39.0" in check.detail       # the repair is the reason


def test_one_healthy_seat_is_enough_for_ok(monkeypatch, doctrine, reference):
    probes = _probe_module()
    monkeypatch.setattr(probes, "_latest_doctrine", lambda: doctrine)
    blocked = doctor.assess(_drop_pkg(_report(), "diffusers"),
                            doctrine).heartbeat_status()
    healthy = doctor.assess(reference, doctrine).heartbeat_status()
    spec = probes.ProbeSpec(capability="a.b", task="text-to-image", doctrine=True)
    workers = [_seat("w1", "text-to-image", blocked),
               _seat("w2", "text-to-image", healthy)]
    assert probes._check_doctrine(spec, workers).status is probes.ProbeStatus.OK


def test_a_silent_worker_beside_a_blocked_one_downgrades_fail_to_unknown(
        monkeypatch, doctrine):
    """A working seat cannot be ruled out, so the capability is not condemned."""
    probes = _probe_module()
    monkeypatch.setattr(probes, "_latest_doctrine", lambda: doctrine)
    blocked = doctor.assess(_drop_pkg(_report(), "diffusers"),
                            doctrine).heartbeat_status()
    workers = [_seat("w1", "text-to-image", blocked),
               _seat("w2", "text-to-image")]
    check = probes._check_doctrine(spec=probes.ProbeSpec(
        capability="a.b", task="text-to-image", doctrine=True), workers=workers)
    assert check.status is probes.ProbeStatus.UNKNOWN
    assert "cannot be ruled out" in check.detail


def test_probe_is_opt_in_so_it_changes_no_existing_capability():
    probes = _probe_module()
    spec = probes.ProbeSpec(capability="a.b", task="text-to-image")
    assert spec.doctrine is False
    assert probes._check_doctrine(spec, []).status is probes.ProbeStatus.UNKNOWN
    result = probes.run_probe(spec, workers=[])
    assert "doctrine" not in [c.name for c in result.checks]


def test_the_two_real_specs_opt_in_and_plan_the_check():
    probes = _probe_module()
    assert probes.PROBE_SPECS["audio.tts"].doctrine is True
    assert probes.PROBE_SPECS["audio.transcribe.word_timestamps"].doctrine is True
    result = probes.run_probe(probes.PROBE_SPECS["audio.tts"],
                              rows={}, model_ids=(), workers=None)
    assert "doctrine" in [c.name for c in result.checks]
    assert result.status is not probes.ProbeStatus.FAIL


def test_an_unknown_worker_never_reads_as_ok(monkeypatch, doctrine):
    """The whole point, stated once more: a box we know nothing about is never
    cleared for work."""
    probes = _probe_module()
    monkeypatch.setattr(probes, "_latest_doctrine", lambda: doctrine)
    spec = probes.ProbeSpec(capability="a.b", task="text-to-image", doctrine=True)
    for workers in (None, [], [_seat("w1", "text-to-image")],
                    [{"id": "w2", "name": "w2", "task_capabilities": {}}]):
        assert probes._check_doctrine(spec, workers).status \
            is not probes.ProbeStatus.OK


# ---------------------------------------------------------------------------
# 7. The shipped doctrine is real.
# ---------------------------------------------------------------------------


def test_the_committed_doctrine_loads_and_names_its_reference():
    current = store.latest()
    if current is None:
        pytest.skip("no doctrine directory in this checkout")
    assert current.entries
    assert current.reference
    assert any(e.severity == "blocker" for e in current.entries)
    for entry in current.entries:
        assert entry.severity in store.SEVERITIES
        assert entry.kind in ("pip", "binary", "mount", "driver")

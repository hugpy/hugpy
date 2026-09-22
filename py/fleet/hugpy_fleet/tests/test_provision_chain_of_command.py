"""Provision chain-of-command + non-downloading probe (ae 1.2TB incident 2026-07-17).

Two operator rulings, exercised without touching the network or the model store:

  A) The worker /probe NEVER downloads. Probing an ABSENT model returns an honest
     non-downloading verdict and does NOT call ensure_model_present. A LOCAL model
     probes as before (ensure_model_present may be called — files already there —
     and the runner path runs).

  B) Central's verdict is AUTHORITATIVE. In _provision_now:
       * central gave an HTTP VERDICT (4xx refusal, or any non-unreachable
         failure) -> NO HF fallback; returns False with the central reason.
       * central was NETWORK-UNREACHABLE (CentralUnreachable) on both transports,
         or no central URL -> HF fallback attempted.
       * HUGPY_HF_FALLBACK=always -> old behavior (any failure falls to HF).

Runs like the other tests here:
    venv/bin/python tests/test_provision_chain_of_command.py
"""
import logging
logging.disable(logging.CRITICAL)

import os
import sys
import types
import urllib.error
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.pop("HUGPY_HF_FALLBACK", None)

provision = importlib.import_module("hugpy_storage.provision")

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


# ══════════════════════════════════════════════════════════════════════════
# Deliverable B — _provision_now chain of command
# ══════════════════════════════════════════════════════════════════════════
CANON = "org/Some-Model"


def _run_provision(*, central_url, parallel, archive, hf, hf_always=False):
    """Drive _provision_now with the three sources stubbed.

    parallel/archive: each is either a bool (return value) or an Exception
    instance to raise. hf: a callable recording marker or an Exception to raise.
    Returns (result, calls) where calls is a set of {"parallel","archive","hf"}.
    """
    calls = set()

    def _fake_parallel(url, canonical, progress=None):
        calls.add("parallel")
        if isinstance(parallel, BaseException):
            raise parallel
        return parallel

    def _fake_archive(url, canonical, progress=None):
        calls.add("archive")
        if isinstance(archive, BaseException):
            raise archive
        return archive

    def _fake_hf(canonical):
        calls.add("hf")
        if isinstance(hf, BaseException):
            raise hf
        return "hf-path"

    orig = (provision.fetch_from_central, provision.fetch_archive_from_central,
            provision.fetch_from_hf)
    if hf_always:
        os.environ["HUGPY_HF_FALLBACK"] = "always"
    else:
        os.environ.pop("HUGPY_HF_FALLBACK", None)
    provision.fetch_from_central = _fake_parallel
    provision.fetch_archive_from_central = _fake_archive
    provision.fetch_from_hf = _fake_hf
    try:
        result = provision._provision_now(CANON, central_url)
    finally:
        (provision.fetch_from_central, provision.fetch_archive_from_central,
         provision.fetch_from_hf) = orig
        os.environ.pop("HUGPY_HF_FALLBACK", None)
    return result, calls


def _unreachable():
    return provision.CentralUnreachable(urllib.error.URLError("connection refused"))


# --- central 4xx / refusal verdict -> NO HF ---------------------------------
# fetch_from_central returns False (a 404/409 verdict), archive also refuses.
res, calls = _run_provision(central_url="http://c", parallel=False,
                            archive=False, hf=lambda: None)
check("central refusal (False/False verdict) -> provision returns False",
      res is False)
check("central refusal -> HF NOT attempted (verdict is authoritative)",
      "hf" not in calls)
check("central refusal -> both central transports were tried",
      calls == {"parallel", "archive"})


# --- central non-unreachable EXCEPTION (a 500 propagated) -> verdict, NO HF --
http500 = urllib.error.HTTPError("http://c", 500, "boom", {}, None)
res, calls = _run_provision(central_url="http://c", parallel=http500,
                            archive=http500, hf=lambda: None)
check("central 5xx (HTTPError) is a VERDICT -> provision False", res is False)
check("central 5xx -> HF NOT attempted", "hf" not in calls)


# --- central UNREACHABLE on both transports -> HF permitted -----------------
res, calls = _run_provision(central_url="http://c", parallel=_unreachable(),
                            archive=_unreachable(), hf=lambda: None)
check("central unreachable on both -> HF attempted (survival path)",
      "hf" in calls)
check("central unreachable -> provision succeeds via HF", res is True)


# --- parallel unreachable, archive gives a VERDICT -> central alive, NO HF --
# Mixed: the parallel transport couldn't connect, but the archive transport DID
# get an HTTP answer (a refusal). Central is alive => no HF.
res, calls = _run_provision(central_url="http://c", parallel=_unreachable(),
                            archive=False, hf=lambda: None)
check("one transport unreachable but the other got a verdict -> central ALIVE",
      "hf" not in calls and res is False)


# --- no central URL -> HF is the only source --------------------------------
res, calls = _run_provision(central_url=None, parallel=False, archive=False,
                            hf=lambda: None)
check("no central URL -> HF attempted directly", "hf" in calls and res is True)
check("no central URL -> central transports NOT called",
      "parallel" not in calls and "archive" not in calls)


# --- escape hatch: HUGPY_HF_FALLBACK=always restores old behavior ------------
res, calls = _run_provision(central_url="http://c", parallel=False,
                            archive=False, hf=lambda: None, hf_always=True)
check("HUGPY_HF_FALLBACK=always -> central verdict still falls through to HF",
      "hf" in calls and res is True)


# --- central succeeds on first transport -> no archive, no HF ---------------
res, calls = _run_provision(central_url="http://c", parallel=True,
                            archive=False, hf=lambda: None)
check("central parallel success -> archive+HF not called",
      calls == {"parallel"} and res is True)


# ── exception taxonomy: _is_unreachable vs a verdict ────────────────────────
check("_is_unreachable(URLError) is True (no HTTP response)",
      provision._is_unreachable(urllib.error.URLError("refused")) is True)
check("_is_unreachable(HTTPError) is False (an HTTP verdict)",
      provision._is_unreachable(
          urllib.error.HTTPError("u", 409, "no", {}, None)) is False)
check("_is_unreachable(TimeoutError) is True",
      provision._is_unreachable(TimeoutError()) is True)
check("_is_unreachable(ConnectionError) is True",
      provision._is_unreachable(ConnectionError()) is True)
check("_is_unreachable(ValueError) is False (not a transport error)",
      provision._is_unreachable(ValueError()) is False)


# ── the fetchers RAISE CentralUnreachable on a bare URLError (not swallow) ───
def _boom_urlerror(*a, **k):
    raise urllib.error.URLError("connection refused")


_orig_get = provision._get_json
provision._get_json = _boom_urlerror
try:
    raised = None
    try:
        provision.fetch_from_central("http://c", CANON)
    except provision.CentralUnreachable as exc:
        raised = exc
    check("fetch_from_central RAISES CentralUnreachable on URLError (was False)",
          raised is not None)
    raised = None
    try:
        provision.fetch_archive_from_central("http://c", CANON)
    except provision.CentralUnreachable as exc:
        raised = exc
    check("fetch_archive_from_central RAISES CentralUnreachable on URLError",
          raised is not None)
finally:
    provision._get_json = _orig_get


# ── the fetchers still return False (a verdict) on a 404/409 HTTPError ───────
def _boom_404(*a, **k):
    raise urllib.error.HTTPError("http://c", 404, "nope", {}, None)


provision._get_json = _boom_404
try:
    check("fetch_from_central returns False on 404 (a verdict, not unreachable)",
          provision.fetch_from_central("http://c", CANON) is False)
    check("fetch_archive_from_central returns False on 409/404 verdict",
          provision.fetch_archive_from_central("http://c", CANON) is False)
finally:
    provision._get_json = _orig_get


# ══════════════════════════════════════════════════════════════════════════
# Deliverable A — probe does NOT download
# ══════════════════════════════════════════════════════════════════════════
agent = importlib.import_module("hugpy_fleet.worker.agent")


class _FakeState:
    central_url = "http://c"


def _probe_with(*, is_local, ensure_should_raise=False):
    """Drive _probe_model with locality + ensure_model_present stubbed, recording
    whether ensure_model_present (the download trigger) was called."""
    calls = {"ensure": False}

    def _fake_registered(mk, url):
        return mk

    def _fake_local(mk):
        return is_local

    def _fake_ensure(*a, **k):
        calls["ensure"] = True
        if ensure_should_raise:
            raise AssertionError("probe must not download an absent model")
        return True

    # Patch the provision names the probe imports locally.
    orig = (provision.ensure_model_registered, provision.model_is_local,
            provision.ensure_model_present)
    provision.ensure_model_registered = _fake_registered
    provision.model_is_local = _fake_local
    provision.ensure_model_present = _fake_ensure
    # Neutralize the GPU + runner machinery so a LOCAL probe doesn't need a card.
    orig_vram = agent._free_vram_bytes
    orig_runner = agent.runner_for
    agent._free_vram_bytes = lambda: 8 * 1024 * 1024 * 1024
    agent.runner_for = lambda model_key=None: types.SimpleNamespace(
        base_url="http://slot", runner=None)
    try:
        result = agent._probe_model("org/Some-Model", _FakeState())
    finally:
        (provision.ensure_model_registered, provision.model_is_local,
         provision.ensure_model_present) = orig
        agent._free_vram_bytes = orig_vram
        agent.runner_for = orig_runner
    return result, calls


# --- absent model: NO download, honest verdict ------------------------------
res, calls = _probe_with(is_local=False, ensure_should_raise=True)
check("probe of an ABSENT model does NOT call ensure_model_present",
      calls["ensure"] is False)
check("probe of an absent model -> ok False, fit False",
      res.get("ok") is False and res.get("fit") is False)
check("probe of an absent model -> local:False in the honest shape",
      res.get("local") is False)
check("probe of an absent model names the lazy-doctrine reason",
      "not local" in (res.get("error") or "")
      and "probe does not download" in (res.get("error") or ""))


# --- local model: probe proceeds as before ----------------------------------
res, calls = _probe_with(is_local=True)
check("probe of a LOCAL model proceeds to the runner path (ok True)",
      res.get("ok") is True)
check("probe of a local model reports local:True", res.get("local") is True)


print(f"\nall {ok} checks passed")

# Re-enable logging for the rest of the pytest session (the disable above is import-time only).
logging.disable(logging.NOTSET)

"""Harness attribution for /v1 (2026-09-24).

A /v1 call is tagged with the harness/client identity so the per-call metrics
row credits the harness (hermes, aider, codex, ...) instead of the bare "api"
default. The identity comes from RECORDED facts only, in a fixed precedence, and
is None when nothing identifies the caller (no guess). These pin the pure
derivation helper in v1_helpers (stdlib-only, offline-testable).
"""
from hugpy_server.app.routes.v1_helpers import (
    _caller_from_user_agent,
    _derive_caller,
)


def test_x_hugpy_client_header_wins():
    h = {"X-Hugpy-Client": "hermes", "X-Title": "other", "User-Agent": "aider/1"}
    assert _derive_caller(h, {"user": "u"}, key_name="k") == "hermes"


def test_x_title_when_no_native_header():
    h = {"X-Title": "opencode", "User-Agent": "curl/8"}
    assert _derive_caller(h, {}, key_name=None) == "opencode"


def test_openai_user_field_before_key_and_ua():
    h = {"User-Agent": "aider/1.2"}
    assert _derive_caller(h, {"user": "batch-job"}, key_name="opkey") == "batch-job"


def test_api_key_name_before_user_agent():
    h = {"User-Agent": "OpenAI/Python 1.2"}
    assert _derive_caller(h, {}, key_name="hermes-key") == "hermes-key"


def test_user_agent_product_is_the_last_resort():
    h = {"User-Agent": "aider/0.42.0"}
    assert _derive_caller(h, {}, key_name=None) == "aider"


def test_none_when_nothing_identifies_the_caller():
    # Absence is explicit: no header, no user, no key, no UA -> None, so the
    # metrics row keeps its honest "api" default rather than a guessed identity.
    assert _derive_caller({}, {}, key_name=None) is None


def test_identity_is_bounded():
    long = "x" * 200
    assert len(_derive_caller({"X-Hugpy-Client": long}, {}, None)) == 64


def test_user_agent_product_token():
    assert _caller_from_user_agent("aider/1.2 (linux)") == "aider"
    assert _caller_from_user_agent("OpenAI/Python 1.99") == "OpenAI"
    assert _caller_from_user_agent("") is None
    assert _caller_from_user_agent("   ") is None

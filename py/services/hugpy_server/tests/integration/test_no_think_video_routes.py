"""no_think seam as consumed by the server's video routes (moved from
hugpy_engine/tests/test_no_think.py: these import hugpy_server)."""
from hugpy_engine.utils.no_think import strip_think, with_no_think  # noqa: F401


def test_video_routes_uses_the_shared_seam():
    import importlib
    vr = importlib.import_module(
        "hugpy_server.app.routes.video_routes")
    assert vr.no_think is strip_think
    assert vr._with_no_think is with_no_think


def _snt():
    import importlib
    return importlib.import_module(
        "hugpy_server.app.routes.video_routes")._studio_no_think


def test_studio_salvages_reasoning_only_reply_as_the_prompt():
    text, reasoning, from_reasoning = _snt()(
        "<think>a sleek red sports car, rain-slicked street, neon reflections</think>")
    assert text == "a sleek red sports car, rain-slicked street, neon reflections"
    assert reasoning == text
    assert from_reasoning is True


def test_studio_salvages_an_unclosed_reasoning_ramble():
    # budget ran out mid-thought — no closing tag; still salvageable, not a block
    text, _r, from_reasoning = _snt()("<think>a red car at night, wet asphalt")
    assert text == "a red car at night, wet asphalt"
    assert from_reasoning is True


def test_studio_prefers_prose_over_reasoning_when_both_present():
    text, reasoning, from_reasoning = _snt()(
        "<think>deliberating</think>a red car on a wet street")
    assert text == "a red car on a wet street"
    assert reasoning == "deliberating"
    assert from_reasoning is False


def test_studio_plain_prose_is_unchanged():
    assert _snt()("a red car") == ("a red car", "", False)


def test_studio_genuinely_empty_stays_empty():
    # neither prose nor reasoning -> the caller still raises an honest error
    assert _snt()("") == ("", "", False)

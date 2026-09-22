"""The bot-side no-think helpers (client copy of the engine seam)."""
from hugpy_discord.no_think import NO_THINK_DIRECTIVE, strip_think, with_no_think


def test_strip_closed_block():
    assert strip_think("<think>hmm</think>\nA red car.") == ("A red car.", "hmm")


def test_strip_unclosed_block_is_all_reasoning():
    assert strip_think("<think>Okay, the user wants") == ("", "Okay, the user wants")


def test_strip_multiple_and_case_insensitive():
    prose, reasoning = strip_think("<THINK>a</THINK>x <think>b</think> y")
    assert prose == "x  y" and reasoning == "a\nb"


def test_strip_empty_and_plain():
    assert strip_think("") == ("", "")
    assert strip_think("plain") == ("plain", "")


def test_with_no_think_appends_once():
    assert with_no_think("") == NO_THINK_DIRECTIVE
    out = with_no_think("describe it")
    assert out == f"describe it\n\n{NO_THINK_DIRECTIVE}"
    assert with_no_think(out) == out

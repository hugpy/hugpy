"""k96 no_makeroom: the /v1 seam forwards the key (truthy only).

Moved from hugpy_engine/tests/test_no_makeroom.py: it exercises the server's
v1_helpers, which the engine may not import.
"""
# ---------------------------------------------------------------------------
# /v1 seam (file-path loaded, same as test_v1_seam)
# ---------------------------------------------------------------------------

from hugpy_server.app.routes import v1_helpers


def test_v1_forwards_no_makeroom_only_when_truthy():
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert v1_helpers._completion_kwargs(dict(base)).get("no_makeroom") is None
    assert v1_helpers._completion_kwargs(
        {**base, "no_makeroom": False}).get("no_makeroom") is None
    assert v1_helpers._completion_kwargs(
        {**base, "no_makeroom": True})["no_makeroom"] is True



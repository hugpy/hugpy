"""Identity namespacing so chatshare ids never collide with Discord ids.

Discord prefs/settings/history are keyed by the raw numeric Discord id. Chatshare
users and rooms are namespaced with a ``cs:`` prefix so their keys live in a
disjoint space from the Discord integers — existing Discord prefs keep working
untouched, and a chatshare user id can never shadow a Discord one.
"""
from __future__ import annotations

CS_PREFIX = "cs:"


def cs_user_key(user_id) -> str:
    """Prefs/settings key for a chatshare user."""
    return f"{CS_PREFIX}{user_id}"


def cs_conv_key(room_id) -> str:
    """Conversation/history + model-resolution key for a chatshare room."""
    return f"{CS_PREFIX}{room_id}"

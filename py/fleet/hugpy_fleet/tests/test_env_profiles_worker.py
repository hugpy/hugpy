"""The worker adopts DB-declared environments from the heartbeat reply
(2026-10-02): merged over its settings-file profiles, kicks a build on change."""
from hugpy_fleet.worker import agent as A


def test_adopt_merges_and_kicks_build_once(monkeypatch):
    for k in ("profiles", "model_profiles", "profiles_db", "model_profiles_db"):
        monkeypatch.setitem(A._RUNTIME_SETTINGS, k, {})
    A._RUNTIME_SETTINGS["profiles"] = {"file-p": {"packages": ["a"]}}
    kicks = []
    from hugpy_engine.serve import profiles as P
    monkeypatch.setattr(P, "materialize_all", lambda specs, register=None: kicks.append(dict(specs)))
    reply = {"env_profiles": {"ct": {"packages": ["compressed-tensors>=0.15.0"], "base": "worker"}},
             "model_profiles": {"M": "ct"}}
    A._adopt_env_profiles(reply)
    A._adopt_env_profiles(reply)                    # unchanged: no second build
    assert len(kicks) == 1 and "ct" in kicks[0]
    assert set(A._declared_profiles()) == {"file-p", "ct"}
    assert A._model_profile_map() == {"M": "ct"}
    A._adopt_env_profiles({"other": 1})             # older central: no change
    assert A._model_profile_map() == {"M": "ct"}


def test_resolver_uses_db_attribution_and_base(monkeypatch):
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "profiles_db", {"ct": {"packages": [], "base": "worker"}})
    monkeypatch.setitem(A._RUNTIME_SETTINGS, "model_profiles_db", {"M": "ct"})
    from hugpy_engine.serve import profiles as P
    seen = {}
    monkeypatch.setattr(P, "state_for", lambda n, pk, base="isolated": seen.update(base=base) or "materializing")
    out = A._resolve_model_profile("M")
    assert out["name"] == "ct" and out["state"] == "materializing" and seen["base"] == "worker"
    assert A._resolve_model_profile("other") is None

"""hugpy_platform.constants: env-driven roots, lazy hfApi, no huggingface_hub at import."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def test_roots_follow_default_root_and_dirs_are_created(tmp_path):
    root = tmp_path / "store"
    code = (
        "import sys, json, os; sys.modules['abstract_hugpy_dev'] = None; "
        "from hugpy_platform import constants as c; "
        "print(json.dumps({'models': c.MODELS_HOME, 'projects': c.PROJECTS_HOME, "
        "'placement': c.PROJECTS_PLACEMENT_PATH, "
        "'placement_exists': os.path.isfile(c.PROJECTS_PLACEMENT_PATH), "
        "'hf_hub_loaded': 'huggingface_hub' in sys.modules, "
        "'hfapi_eager': 'hfApi' in vars(c), "
        "'hf_home_env': os.environ.get('HF_HOME')}))"
    )
    env = dict(os.environ, DEFAULT_ROOT=str(root))
    # Clear EVERY sibling storage-root override so the assertion measures pure
    # DEFAULT_ROOT derivation (the test_isolation conftest now FORCES these to a
    # session temp dir, and an env override always wins over the derived default).
    for k in ("MODELS_HOME", "UPLOADS_HOME", "PROJECTS_HOME", "IDENTITIES_HOME",
              "DATASETS_HOME", "HUGPY_VIDEO_STATE_DIR",
              "HF_HOME", "HF_CACHE", "HF_HUB_CACHE"):
        env.pop(k, None)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout)
    assert out["models"] == str(root / "models")
    assert out["projects"] == str(root / "projects")
    assert out["placement_exists"] is True
    assert out["hf_hub_loaded"] is False, "constants must not import huggingface_hub"
    assert out["hfapi_eager"] is False
    assert out["hf_home_env"].startswith(str(root))
    assert (root / "models").is_dir() and (root / "uploads").is_dir()


def test_env_bool_coerces_strings(monkeypatch):
    from hugpy_platform import constants as c
    monkeypatch.setenv("HUGPY_TEST_FLAG", "false")
    assert c.env_bool("HUGPY_TEST_FLAG", True) is False
    monkeypatch.setenv("HUGPY_TEST_FLAG", "YES")
    assert c.env_bool("HUGPY_TEST_FLAG", False) is True
    monkeypatch.delenv("HUGPY_TEST_FLAG")
    assert c.env_bool("HUGPY_TEST_FLAG", True) is True


def test_unwritable_env_root_falls_back_to_default(tmp_path):
    from hugpy_platform import constants as c
    default = str(tmp_path / "fallback")
    unwritable = str(tmp_path / "ro")
    os.makedirs(unwritable)
    os.chmod(unwritable, 0o500)
    try:
        if os.access(unwritable, os.W_OK):  # running as root: cannot make it unwritable
            pytest.skip("cannot create an unwritable dir as this user")
        os.environ["HUGPY_TEST_ROOT_X"] = unwritable
        assert c._env_root_or_default("HUGPY_TEST_ROOT_X", default) == default
    finally:
        os.environ.pop("HUGPY_TEST_ROOT_X", None)
        os.chmod(unwritable, 0o700)


def test_hfapi_is_lazy_and_rebindable(monkeypatch):
    from hugpy_platform import constants as c
    # Rebinding (what hugpy_storage.hf_token does after a token change) must work
    # even before the lazy attribute was ever built.
    sentinel = object()
    monkeypatch.setattr(c, "hfApi", sentinel, raising=False)
    assert c.hfApi is sentinel
    monkeypatch.delattr(c, "hfApi")
    pytest.importorskip("huggingface_hub")
    api = c.hfApi
    assert type(api).__name__ == "HfApi"
    assert c.hfApi is api, "built once, then cached in the module namespace"


def test_unknown_attribute_still_raises():
    from hugpy_platform import constants as c
    with pytest.raises(AttributeError):
        c.no_such_constant_xyz

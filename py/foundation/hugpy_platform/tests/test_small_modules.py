"""central / module_imports / compat_pydantic / trust / except_utils smoke tests."""
from __future__ import annotations

import sys

import pytest

from hugpy_platform import central, compat_pydantic, except_utils, module_imports, trust


# -- central ------------------------------------------------------------------
def test_central_base_url_default_and_aliases(monkeypatch):
    for k in central.CENTRAL_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    assert central.central_base_url() == central.DEFAULT_CENTRAL
    assert central.central_base_url(default=None) is None
    monkeypatch.setenv("WORKER_CENTRAL_URL", "http://legacy:7002/")
    assert central.central_base_url() == "http://legacy:7002"
    monkeypatch.setenv("HUGPY_BASE_URL", "https://canonical/")
    assert central.central_base_url() == "https://canonical", "canonical name wins"


# -- module_imports -----------------------------------------------------------
def test_is_available_and_require():
    assert module_imports.is_available("json")
    assert not module_imports.is_available("hugpy_no_such_package_x9")
    with pytest.raises(ImportError, match="pip install hugpy_no_such_package_x9"):
        module_imports.require("hugpy_no_such_package_x9", "for the test")
    assert module_imports.require("json") is __import__("json")


def test_require_names_the_extra_for_known_heavy_packages(monkeypatch):
    monkeypatch.setattr(module_imports, "lazy_import", lambda name: module_imports.nullProxy(name))
    with pytest.raises(ImportError, match=r"hugpy\[keywords\]"):
        module_imports.require("keybert")


# -- compat_pydantic ------------------------------------------------------------
def test_shim_models_construct_default_and_dump():
    shim = compat_pydantic._build_shim()

    class Inner(shim.BaseModel):
        n: int = 1

    class Req(shim.BaseModel):
        model_config = shim.ConfigDict(extra="forbid")
        name: str
        tags: list = shim.Field(default_factory=list)
        inner: Inner = shim.Field(default_factory=Inner)
        opt: str = "d"

    r = Req(name="x", extra=3)
    assert r.name == "x" and r.tags == [] and r.opt == "d" and r.inner.n == 1
    assert r.model_dump() == {"name": "x", "tags": [], "inner": {"n": 1}, "opt": "d"}
    assert r.dict() == r.model_dump()
    assert shim.BaseModel.model_validate.__func__ is not None
    assert Req.model_validate({"name": "y"}).name == "y"
    assert Req.model_validate(r) is r
    assert "Req(" in repr(r)
    # decorators are no-ops; unknown names are lenient stubs, dunders stay honest
    assert shim.model_validator(mode="after")(len) is len
    assert shim.field_validator(len) is len
    assert shim.SomeUnknownThing(1, 2)(len) is len
    with pytest.raises(AttributeError):
        shim.__file__


def test_ensure_pydantic_is_a_noop_with_the_real_package():
    real = pytest.importorskip("pydantic")
    assert compat_pydantic.ensure_pydantic() is False
    assert sys.modules["pydantic"] is real


def test_ensure_pydantic_installs_the_shim_when_pydantic_is_missing(monkeypatch):
    saved = {k: sys.modules[k] for k in list(sys.modules) if k == "pydantic" or k.startswith("pydantic.") or k == "pydantic_core" or k.startswith("pydantic_core.")}
    for k in saved:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.setitem(sys.modules, "pydantic", None)      # make `import pydantic` fail
    try:
        assert compat_pydantic.ensure_pydantic() is True
        import pydantic
        assert getattr(pydantic, "SHIM", False) is True
        assert sys.modules["pydantic_core"].__version__ == "0-shim"
    finally:
        for k in ("pydantic", "pydantic_core"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)


# -- trust ----------------------------------------------------------------------
def test_trust_tiers():
    assert trust.trust_tier("meta-llama/Llama-3") == 2
    assert trust.trust_tier("bartowski/Qwen-GGUF") == 1
    assert trust.trust_tier("someone/thing") == 0
    assert trust.trust_tier("someone/thing", author="Qwen") == 2
    assert trust.trust_label(2) == "first-party" and trust.trust_label(0) is None


# -- except_utils --------------------------------------------------------------
def test_caught_attempt_catching():
    def boom():
        raise ValueError("x")

    assert except_utils.caught(boom) is except_utils.FAILED
    assert except_utils.caught(boom, default=None) is None
    with pytest.raises(ValueError):
        except_utils.caught(boom, reraise=True)
    ok, val, exc = except_utils.attempt(boom)
    assert (ok, val) == (False, None) and isinstance(exc, ValueError)
    assert except_utils.attempt(lambda: 3) == (True, 3, None)
    assert except_utils.catching(default=7)(boom)() == 7
    with except_utils.caught_block("blk"):
        boom()

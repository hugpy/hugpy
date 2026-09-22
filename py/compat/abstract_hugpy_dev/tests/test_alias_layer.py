"""The alias layer resolves old dotted paths to the new module objects."""

from __future__ import annotations

import importlib
import sys

import pytest

import abstract_hugpy_dev
from abstract_hugpy_dev import _relocations as R


def test_version_is_the_bridge_line():
    assert abstract_hugpy_dev.__version__.startswith("0.1.267")


def test_finder_installed_once():
    importlib.import_module("abstract_hugpy_dev")
    finders = [f for f in sys.meta_path if isinstance(f, R.RelocationFinder)]
    assert len(finders) == 1


@pytest.mark.parametrize("old, new", [
    ("abstract_hugpy_dev._compat_pydantic", "hugpy_platform.compat_pydantic"),
    ("abstract_hugpy_dev._platform.hardware", "hugpy_platform.hardware"),
    ("abstract_hugpy_dev.chaos", "hugpy_ops.chaos"),
    ("abstract_hugpy_dev.bot.config", "hugpy_discord.config"),
])
def test_relocated_module_is_the_same_object(old, new):
    assert R.relocated(old) == new
    assert importlib.import_module(old) is importlib.import_module(new)


def test_relocated_prefix_extends_to_children():
    # a mapped package prefix aliases every submodule beneath it
    assert R.relocated("abstract_hugpy_dev.bot.cogs.chat") == "hugpy_discord.cogs.chat"


def test_every_map_target_exists():
    missing = [old for old, new in R._MAP.items() if not R._target_exists(new)]
    assert not missing, f"{len(missing)} alias targets are not installed: {missing[:10]}"


def test_every_old_module_imports():
    failed = {}
    for old, new, kind in R.iter_old_modules():
        try:
            mod = importlib.import_module(old)
        except ModuleNotFoundError as exc:
            if exc.name and not exc.name.startswith(("hugpy", "abstract_hugpy_dev")):
                continue  # optional extra (discord.py, torch, ...) not installed here
            failed[old] = f"{type(exc).__name__}: {exc}"
            continue
        except Exception as exc:  # noqa: BLE001
            failed[old] = f"{type(exc).__name__}: {exc}"
            continue
        if kind == "relocated" and sys.modules.get(new) is not mod:
            failed[old] = f"not the same object as {new}"
    assert not failed, f"{len(failed)} old paths broken: {dict(list(failed.items())[:10])}"


def test_aggregator_namespace_is_lazy_and_resolves():
    # retired re-export layer: synthesised namespace whose names resolve lazily
    imports = importlib.import_module("abstract_hugpy_dev.imports")
    assert R.is_namespace("abstract_hugpy_dev.imports")
    assert imports.__path__ == []
    with pytest.raises(AttributeError):
        getattr(imports, "definitely_not_a_name_xyz")


def test_deep_namespace_prefix_importable():
    pkg = importlib.import_module("abstract_hugpy_dev.flask_app.app.routes")
    assert "chat_routes" in R.children("abstract_hugpy_dev.flask_app.app.routes")
    assert hasattr(pkg, "__path__")


def test_retired_modules_do_not_resolve():
    for old in R._RETIRED:
        assert not R.resolvable(old), old
        with pytest.raises(ImportError):
            importlib.import_module(old)


def test_top_level_surface_present():
    unavailable = set(abstract_hugpy_dev._SURFACE_UNAVAILABLE)
    missing = [n for n in abstract_hugpy_dev._SURFACE
               if not hasattr(abstract_hugpy_dev, n) and n not in unavailable]
    assert not missing, f"{len(missing)} names missing from the old surface: {missing[:20]}"
    assert set(abstract_hugpy_dev.__all__) <= set(abstract_hugpy_dev._SURFACE)
    # only stdlib names newer than this interpreter may be unavailable
    assert unavailable <= {"NoDefault", "ReadOnly", "TypeIs", "TypeAliasType", "override",
                           "Never", "Self", "ByteString", "get_protocol_members", "is_protocol"}, unavailable


def test_unknown_top_level_name_raises():
    with pytest.raises(AttributeError):
        abstract_hugpy_dev.definitely_not_a_name_xyz

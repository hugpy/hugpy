"""Behavioural smoke tests for the ``hugpy`` command.

The meta CLI owns no feature; it maps subcommands to the packages that do.
These tests pin that contract: every ``--help`` exits 0, every dispatch table
entry points at a real module when its owner is installed, and a missing owner
turns into an install hint rather than a traceback.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys

import pytest

from hugpy import cli

# Owner modules known to be absent in this workspace (none at present); the
# console-script fallback covers such a module when its script is on PATH.
KNOWN_MISSING_MODULES: set[str] = set()


def _run(*argv: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "hugpy.cli", *argv],
        capture_output=True, text=True, timeout=110, env=env,
    )


# --------------------------------------------------------------------------- #
# --help never fails
# --------------------------------------------------------------------------- #
def test_top_level_help_exits_zero():
    proc = _run("--help")
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in cli.DISPATCH:
        assert name in proc.stdout, f"{name} missing from top-level help"


@pytest.mark.parametrize("command", sorted(set(cli.DISPATCH) | {"chat", "install-deps", "version"}))
def test_every_subcommand_help_exits_zero(command):
    proc = _run(command, "--help")
    assert proc.returncode == 0, f"{command} --help:\n{proc.stderr[-3000:]}"
    assert proc.stdout.strip(), f"{command} --help printed nothing"


def test_passthrough_help_without_owner_is_friendly(monkeypatch, capsys):
    """``hugpy <owner-cmd> --help`` still exits 0 when the owner is absent."""
    monkeypatch.setitem(sys.modules, cli.DISPATCH["worker"].module, None)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert cli.main(["worker", "--help"]) == 0
    out = capsys.readouterr().out
    assert 'pip install "hugpy[worker]"' in out


# --------------------------------------------------------------------------- #
# dispatch table
# --------------------------------------------------------------------------- #
def test_dispatch_table_covers_the_documented_surface():
    expected = {
        "serve", "worker", "gguf-worker", "phone-brick", "bot", "download",
        "storage", "install-engine", "reclassify-images", "keeper", "sentinel",
        "chaos", "provision", "oracle", "curation", "video",
    }
    assert expected == set(cli.DISPATCH)
    for name in cli.PASSTHROUGH:
        assert name in cli.DISPATCH


@pytest.mark.parametrize("command", sorted(cli.DISPATCH))
def test_target_module_is_locatable_when_owner_installed(command):
    target = cli.DISPATCH[command]
    if target.module in KNOWN_MISSING_MODULES:
        pytest.xfail(f"{target.module} does not exist yet (console-script fallback)")
    assert importlib.util.find_spec(target.package) is not None, f"{target.package} not installed"
    assert importlib.util.find_spec(target.module) is not None, target.module


@pytest.mark.parametrize("command", sorted(cli.DISPATCH))
def test_resolve_target_returns_callable_or_dependency_hint(command):
    """With the owner installed, resolution yields its entry point; when only
    a third-party dependency of the owner is missing (discord.py, torch...),
    the error names the extra that would install it."""
    target = cli.DISPATCH[command]
    if target.module in KNOWN_MISSING_MODULES:
        pytest.xfail(f"{target.module} does not exist yet (console-script fallback)")
    try:
        fn = cli.resolve_target(command)
    except cli.DispatchError as exc:
        msg = str(exc)
        assert "is installed but its dependency" in msg, msg
        assert f'pip install "hugpy[{target.extra}]"' in msg
        return
    assert callable(fn)
    assert getattr(fn, "__name__", "") in target.attrs or fn.__name__.startswith("console_script:")


@pytest.mark.parametrize("command", sorted(cli.DISPATCH))
def test_blocked_owner_gives_install_hint(command, monkeypatch):
    target = cli.DISPATCH[command]
    monkeypatch.setitem(sys.modules, target.module, None)
    monkeypatch.setitem(sys.modules, target.package, None)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(cli.DispatchError) as info:
        cli.resolve_target(command)
    msg = str(info.value)
    assert target.distribution in msg
    assert f'pip install "hugpy[{target.extra}]"' in msg
    assert "Traceback" not in msg


def test_blocked_owner_via_main_exits_one_without_traceback(monkeypatch, capsys):
    target = cli.DISPATCH["reclassify-images"]
    monkeypatch.setitem(sys.modules, target.module, None)
    monkeypatch.setitem(sys.modules, target.package, None)
    rc = cli.main(["reclassify-images"])
    captured = capsys.readouterr()
    assert rc == 1
    assert 'pip install "hugpy[engine]"' in captured.err
    assert "Traceback" not in captured.err


def test_console_script_fallback_when_module_lacks_entry_point(monkeypatch, tmp_path):
    """An owner without the expected attribute still dispatches through its
    console script when one is on PATH."""
    import types

    target = cli.DISPATCH["curation"]
    monkeypatch.setitem(sys.modules, target.module, types.ModuleType(target.module))
    script = tmp_path / target.script
    script.write_text("#!/bin/sh\nexit 7\n")
    script.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(script) if name == target.script else None)
    fn = cli.resolve_target("curation")
    assert fn.__name__ == f"console_script:{script}"
    assert fn([]) == 7


def test_blocked_owner_in_subprocess_prints_hint_only():
    target = cli.DISPATCH["worker"]
    code = (
        "import sys, shutil; "
        f"sys.modules[{target.package!r}] = None; sys.modules[{target.module!r}] = None; "
        "shutil.which = lambda _n: None; "
        "from hugpy import cli; sys.exit(cli.main(['worker', '--central', 'http://x']))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=110)
    assert proc.returncode == 1
    assert 'pip install "hugpy[worker]"' in proc.stderr
    assert "Traceback" not in proc.stderr


# --------------------------------------------------------------------------- #
# stdlib-only subcommands
# --------------------------------------------------------------------------- #
def test_install_deps_dry_run_targets_the_meta_distribution(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_detect_profile", lambda: ("cpu", "no usable NVIDIA GPU detected; choosing cpu-worker"))
    assert cli.main(["install-deps", "--cpu", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "hugpy[cpu-worker]" in out
    assert "abstract_hugpy_dev" not in out


def test_install_deps_auto_profile_uses_platform_probe(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_detect_profile", lambda: ("gpu", "detected 1 GPU(s): X (index 0); choosing gpu-worker"))
    assert cli.main(["install-deps", "--profile", "auto", "--dry-run"]) == 0
    assert "hugpy[gpu-worker]" in capsys.readouterr().out


def test_version_lists_distributions(capsys):
    assert cli.main(["version"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("hugpy ")
    assert "hugpy-platform" in out


def test_chat_unreachable_central_is_a_clean_failure(capsys):
    rc = cli.main(["chat", "--central", "http://127.0.0.1:1", "hello"])
    assert rc == 1
    assert "cannot reach central" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #
def test_import_hugpy_loads_only_platform():
    code = (
        "import sys; sys.modules['abstract_hugpy_dev'] = None; "
        "import hugpy, hugpy.cli, hugpy.hpy; "
        "print(sorted(m.split('.')[0] for m in sys.modules if m.startswith('hugpy')))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=110)
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = eval(proc.stdout.strip())
    assert set(loaded) <= {"hugpy", "hugpy_platform"}, loaded


def test_version_is_installed_metadata():
    from importlib.metadata import version

    import hugpy

    assert hugpy.__version__ == version("hugpy")
    assert set(hugpy.__all__) >= {"__version__", "cli", "hpy"}

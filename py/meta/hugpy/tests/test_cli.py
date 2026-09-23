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


@pytest.mark.parametrize("command", sorted(set(cli.DISPATCH) | {"chat", "install-deps", "version", "build"}))
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
        "chaos", "provision", "drift", "oracle", "curation", "video",
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


def test_version_prints_identity_line_first_when_buildinfo_present(monkeypatch, capsys):
    import types

    fake = types.ModuleType("hugpy_platform.buildinfo")
    fake.identity_line = lambda: "hugpy-fleet 1.2.3.dev4+gabc1234 (editable /ws)"
    monkeypatch.setattr(cli, "_buildinfo", lambda: fake)
    assert cli.main(["version"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "hugpy-fleet 1.2.3.dev4+gabc1234 (editable /ws)"
    assert lines[1].startswith("hugpy ")


# --------------------------------------------------------------------------- #
# drift / build
# --------------------------------------------------------------------------- #
def test_drift_is_a_passthrough_to_hugpy_ops(monkeypatch):
    """``hugpy drift ...`` hands every argument to hugpy_ops.drift:main."""
    import types

    target = cli.DISPATCH["drift"]
    assert target.module == "hugpy_ops.drift" and target.extra == "ops"
    assert target.script == "hugpy-drift-check"
    assert "drift" in cli.PASSTHROUGH
    seen: list[list[str]] = []
    fake = types.ModuleType(target.module)
    fake.main = lambda argv=None: seen.append(list(argv or [])) or 3
    monkeypatch.setitem(sys.modules, target.module, fake)
    rc = cli.main(["drift", "--sections", "A,B", "--allow-dirty", "--json"])
    assert rc == 3
    assert seen == [["--sections", "A,B", "--allow-dirty", "--json"]]


def test_drift_without_ops_gives_install_hint(monkeypatch, capsys):
    target = cli.DISPATCH["drift"]
    monkeypatch.setitem(sys.modules, target.module, None)
    monkeypatch.setitem(sys.modules, target.package, None)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert cli.main(["drift", "--sections", "A"]) == 1
    err = capsys.readouterr().err
    assert 'pip install "hugpy[ops]"' in err and "Traceback" not in err


def test_build_prints_identity_line_and_json(monkeypatch, capsys):
    import json
    import types

    fake = types.ModuleType("hugpy_platform.buildinfo")
    fake.identity_line = lambda: "hugpy-server 2.0.0 (g0123abc)"
    fake.build_info = lambda: {"identity": {"version": "2.0.0", "sha": "0123abc"},
                               "lockstep": {"ok": True, "versions": {}}}
    monkeypatch.setattr(cli, "_buildinfo", lambda: fake)
    assert cli.main(["build"]) == 0
    assert capsys.readouterr().out.strip() == "hugpy-server 2.0.0 (g0123abc)"
    assert cli.main(["build", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["identity"]["sha"] == "0123abc" and doc["lockstep"]["ok"] is True


def test_build_without_buildinfo_exits_one(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_buildinfo", lambda: None)
    assert cli.main(["build"]) == 1
    err = capsys.readouterr().err
    assert "buildinfo" in err and "Traceback" not in err


def test_build_lazy_import_is_blocked_cleanly(monkeypatch, capsys):
    """The real lazy import path: a blocked module resolves to None, exit 1."""
    monkeypatch.setitem(sys.modules, "hugpy_platform.buildinfo", None)
    assert cli.main(["build"]) == 1
    assert "buildinfo" in capsys.readouterr().err


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

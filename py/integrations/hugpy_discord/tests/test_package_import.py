"""``import hugpy_discord`` must work without discord.py / python-dotenv (they are
the ``bot`` extra), and the bot entry point must be declared and runnable."""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


def test_import_without_discord_or_dotenv():
    proc = _run(
        "import sys; sys.modules['abstract_hugpy_dev'] = None; "
        "sys.modules['discord'] = None; sys.modules['dotenv'] = None; "
        "import hugpy_discord, hugpy_discord.no_think, hugpy_discord.config as cfg; "
        "print(hugpy_discord.__version__, cfg.HUGPY_BASE_URL.startswith('http'))"
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.split() == ["0.1.0", "True"]


def test_bot_names_are_lazy_and_need_the_extra():
    proc = _run(
        "import sys\n"
        "sys.modules['discord'] = None\n"
        "import hugpy_discord\n"
        "assert 'hugpy_discord.bot' not in sys.modules\n"
        "try:\n"
        "    hugpy_discord.main\n"
        "except ImportError:\n"
        "    print('lazy-ok')\n"
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "lazy-ok"


def test_entry_point_is_declared():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["scripts"]["hugpy-bot"] == "hugpy_discord.bot:main"
    assert set(data["project"]["optional-dependencies"]["bot"]) == {"discord.py", "python-dotenv"}


def test_main_help_and_missing_token(monkeypatch, tmp_path):
    pytest.importorskip("discord")
    from hugpy_discord.bot import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.setenv("HUGPY_BOT_LEGACY_ENV", str(tmp_path / "absent.env"))
    monkeypatch.setenv("HUGPY_BOT_ENV", str(tmp_path / "absent2.env"))
    assert main(["--env", str(tmp_path / "absent2.env")]) == 1

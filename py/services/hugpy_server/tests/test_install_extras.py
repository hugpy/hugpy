"""The enroll installer ships the media-intelligence deps + the numpy pin.

The canonical [engine] venv omits sentence-transformers / openai-whisper /
keybert, and numpy>=2.5 breaks numba so `import whisper` dies — the three
2026-07-11 request-time failures. bootstrap.sh (served by the
GET /llm/workers/install.sh route) must install all three plus `numpy<2.5`, and
the agent's `pip install -U --no-deps` self-update must never strip them.

We assert against the ACTUAL rendered route output (workers_install_sh), and,
as a second check, that the packaged resource the route serves carries the block.
"""
from importlib import resources

import pytest
from flask import Flask

from hugpy_server.app.routes import worker_routes as wr

EXTRAS = ("sentence-transformers", "openai-whisper", "keybert")


@pytest.fixture(scope="module")
def body():
    """The rendered installer body for THIS central."""
    app = Flask(__name__)
    with app.test_request_context("/api/llm/workers/install.sh",
                                  base_url="https://dev.hugpy.ai"):
        resp = wr.workers_install_sh()
        return resp.get_data(as_text=True)


# --- the rendered route output ---------------------------------------------
def test_route_renders_shell_script_for_this_central(body):
    assert body.lstrip().startswith("#!")
    assert "https://dev.hugpy.ai" in body


@pytest.mark.parametrize("pkg", EXTRAS)
def test_rendered_installer_ships_extra(body, pkg):
    assert pkg in body


def test_rendered_installer_pins_numpy(body):
    """numba/whisper landmine."""
    assert "numpy<2.5" in body


def test_extras_and_pin_are_one_pip_line(body):
    # All three deps + the pin belong to a SINGLE clear `install --upgrade …` line
    # (the human-readable `say` echo above it doesn't carry `--upgrade`).
    pip_lines = [ln for ln in body.splitlines()
                 if "install" in ln and "--upgrade" in ln and "sentence-transformers" in ln]
    assert len(pip_lines) == 1
    assert all(p in pip_lines[0] for p in EXTRAS + ("numpy<2.5",))


def test_comment_names_incident_class(body):
    assert "2026-07-11" in body and "request time" in body.lower()


# --- the packaged resource the route serves --------------------------------
def test_packaged_bootstrap_carries_extras_block():
    raw = (resources.files("hugpy_fleet.worker")
           .joinpath("bootstrap.sh").read_text(encoding="utf-8"))
    assert all(p in raw for p in EXTRAS) and "numpy<2.5" in raw
    # Self-update persistence: the block sits AFTER the main [engine] install and the
    # agent's converge uses --no-deps, so the extras survive every version bump.
    assert "--no-deps" in raw


def test_agent_self_update_uses_no_deps():
    agent_src = (resources.files("hugpy_fleet.worker")
                 .joinpath("agent.py").read_text(encoding="utf-8"))
    assert '"install", "-U", "--no-deps"' in agent_src

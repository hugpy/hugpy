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
    # Self-update persistence: the block sits AFTER the main profile install and
    # the agent's converge only pins hugpy-* (constraints + only-if-needed; the
    # no-constraints fallback is --no-deps), so the extras survive every bump.
    assert "only-if-needed" in raw and "--no-deps" in raw


def test_bootstrap_installs_profile_under_lockstep_constraints():
    """WP5: the pinned install passes central's constraints.txt as -c."""
    raw = (resources.files("hugpy_fleet.worker")
           .joinpath("bootstrap.sh").read_text(encoding="utf-8"))
    assert "/llm/workers/constraints.txt" in raw
    assert 'PIP_CONSTRAINT="-c ${CONSTRAINTS_FILE}"' in raw
    assert 'SPEC="hugpy[${PROFILE}]==${VERSION}"' in raw
    assert 'install --upgrade $PIP_CONSTRAINT $PIP_EXTRA_INDEX "$SPEC"' in raw
    assert "abstract_hugpy_dev" not in raw


def test_bootstrap_adds_central_index_as_an_extra_index_when_advertised():
    """A release published on central's own index (py/build_wheels.py
    --publish) reaches a bare box too: required-version names ``pkg_index_url``
    and the bootstrap passes it as ``--extra-index-url`` — PyPI stays for the
    third-party deps, and an operator can pre-set WORKER_PKG_INDEX_URL."""
    raw = (resources.files("hugpy_fleet.worker")
           .joinpath("bootstrap.sh").read_text(encoding="utf-8"))
    assert '"pkg_index_url"' in raw
    assert 'PIP_EXTRA_INDEX="--extra-index-url ${PKG_INDEX_URL}"' in raw
    assert 'PKG_INDEX_URL="${WORKER_PKG_INDEX_URL:-}"' in raw
    assert "--index-url" not in raw.replace("--extra-index-url", "")   # never replaces PyPI


def test_agent_self_update_converges_under_constraints_with_no_deps_fallback():
    agent_src = (resources.files("hugpy_fleet.worker")
                 .joinpath("agent.py").read_text(encoding="utf-8"))
    # ONE helper builds the pip line for both the heartbeat and /ops/update.
    assert agent_src.count("def _pip_converge_command(") == 1
    # the two call sites (heartbeat self-update, /ops/update), not the def
    assert agent_src.count("= _prepare_converge(args, ") == 2
    assert '"--upgrade-strategy", "only-if-needed", "-c"' in agent_src
    assert '"--no-deps"' in agent_src   # the fetch-failed fallback survives


def test_route_renders_forwarded_https_origin_behind_proxy():
    """Behind nginx the backend sees plain http; the script must default
    --central to the PUBLIC https origin (port 80 does not answer)."""
    app = Flask(__name__)
    with app.test_request_context("/llm/workers/install.sh", base_url="http://dev.hugpy.ai",
                                  headers={"X-Forwarded-Proto": "https",
                                           "X-Forwarded-Host": "dev.hugpy.ai"}):
        out = wr.workers_install_sh().get_data(as_text=True)
    assert 'CENTRAL="https://dev.hugpy.ai"' in out


def test_bootstrap_queries_central_through_the_api_mount():
    """--central is the bare origin; the lookups must go through /api or they
    hit the console SPA and fall back to an unpinned, unconstrained install."""
    raw = (resources.files("hugpy_fleet.worker")
           .joinpath("bootstrap.sh").read_text(encoding="utf-8"))
    assert '"${CENTRAL}/api$1"' in raw and '"${CENTRAL}$1"' not in raw
    assert 'CENTRAL="${CENTRAL%/api}"' in raw

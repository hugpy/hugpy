"""item 3 — a 401/403 is terminal ONLY when it is CENTRAL's own refusal.

The a-brain incident (2026-09-24): the dev.hugpy.ai WEB GATE returned a 403 HTML
page while central's own record was 'approved' and central logged nothing; the
agent read it as a block and os._exit(0)'d permanently (Restart=on-failure never
respawns exit 0). The fix: only central's own marker body is terminal; any other
401/403 is an intermediary (proxy/gate) and must be retried.

Run: `PYTHONPATH=$(ls -d py/*/*/src|tr '\n' :) pytest tests/test_worker_reject_marker.py -q`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import agent as a


def test_central_block_body_is_terminal():
    assert a._is_central_terminal_refusal(403, "<p>Worker is blocked by the operator.</p>")


def test_central_enroll_body_is_terminal():
    assert a._is_central_terminal_refusal(401, "Worker enrollment token invalid or required.")


def test_web_gate_403_html_is_NOT_terminal():
    # The exact class of body a-brain died on: a reverse-proxy / login page.
    assert not a._is_central_terminal_refusal(
        403, "<html><head><title>403 Forbidden</title></head><body>nginx</body></html>")


def test_empty_403_body_is_NOT_terminal():
    assert not a._is_central_terminal_refusal(403, "")


def test_non_auth_code_never_terminal():
    assert not a._is_central_terminal_refusal(500, "Worker is blocked by the operator.")
    assert not a._is_central_terminal_refusal(410, "Worker is blocked by the operator.")


def test_marker_match_is_whitespace_tolerant_and_case_insensitive():
    assert a._is_central_terminal_refusal(
        403, "   worker   IS   BLOCKED   by the operator.  ")

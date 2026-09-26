"""Turnkey worker setup — the by-hand steps, now converged + self-checked in code.

Covers the PURE logic of hugpy_fleet.worker.setup (no box touched) plus the
installer wiring that consumes it:
  * item 1/10 advertise URL derivation (explicit wins; derived; none)
  * item 2 slot-port collision plan (the a-brain 9101 child==worker collision)
  * item 7 unit drift report + the fleet serve-mode default (swap)
  * item 8 central-host resolution for the firewall rule
  * item 11 comfy port that avoids the worker + slot ports
  * the Check/render/overall_rc contract (a required FAIL => non-zero install)

Run: `PYTHONPATH=$(ls -d py/*/*/src|tr '\n' :) pytest tests/test_worker_setup.py -q`
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import setup as S
from hugpy_fleet.worker import install as wi


# --------------------------------------------------------------------------- #
# item 2 — slot port plan                                                     #
# --------------------------------------------------------------------------- #
def test_slot_port_plan_flags_abrain_child_collision():
    # a-brain: worker --port 9101, default base 8101, 2 slots -> slot 1 child 9101.
    plan = S.slot_port_plan(9101, 2, 8101)
    assert any("child port 9101 == worker port 9101" in c for c in plan["collisions"])
    assert plan["children"] == [9101, 9102]


def test_slot_port_plan_no_collision_default_port():
    plan = S.slot_port_plan(9100, 2, 8101)
    assert plan["collisions"] == []


def test_check_slot_ports_control_collision_is_required_fail():
    # A control-port collision can't be silently relocated (agent finds slots at
    # slot_urls()) -> required FAIL. A child-only collision is a WARN.
    child = S.check_slot_ports(9101, 2, 8101)
    assert child.status == S.WARN and child.required is False
    control = S.check_slot_ports(8101, 2, 8101)
    assert control.status == S.FAIL and control.required is True


# --------------------------------------------------------------------------- #
# item 1 / 10 — advertise URL                                                 #
# --------------------------------------------------------------------------- #
def test_advertise_explicit_wins():
    url, src = S.derive_advertise_url("http://10.99.0.3:7002", 9101,
                                      "http://10.99.0.1:9101")
    assert src == "explicit" and url == "http://10.99.0.1:9101"


def test_advertise_none_when_unroutable(monkeypatch):
    monkeypatch.setattr(S, "_local_ip_toward", lambda _c: None)
    c = S.check_advertise("http://10.99.0.3:7002", 9101, None)
    assert c.status == S.FAIL and "--advertise" in (c.fix or "")


def test_advertise_derived_reports_source(monkeypatch):
    monkeypatch.setattr(S, "_local_ip_toward", lambda _c: "10.168.168.238")
    c = S.check_advertise("http://10.99.0.3:7002", 9101, None)
    assert c.status == S.OK and "DERIVED" in c.detail and "10.168.168.238" in c.detail


# --------------------------------------------------------------------------- #
# item 7 — unit drift + fleet serve-mode default                             #
# --------------------------------------------------------------------------- #
def test_unit_drift_names_serve_mode_and_central_and_exec():
    old = ('Environment="DEFAULT_SERVE_MODE=off"\n'
           'Environment="WORKER_CENTRAL_URL=https://dev.hugpy.ai"\n'
           'ExecStart=/old\n')
    new = ('Environment="DEFAULT_SERVE_MODE=swap"\n'
           'Environment="WORKER_CENTRAL_URL=http://10.99.0.3:7002"\n'
           'Environment="WORKER_URL=http://10.99.0.1:9101"\n'
           'ExecStart=/new\n')
    d = S.unit_drift(old, new)
    assert "~ DEFAULT_SERVE_MODE: off -> swap" in d
    assert any("WORKER_CENTRAL_URL" in x and "dev.hugpy.ai" in x for x in d)
    assert any(x.startswith("+ WORKER_URL=") for x in d)
    assert "~ ExecStart changed" in d


def test_unit_drift_empty_when_identical():
    text = 'Environment="A=1"\nExecStart=/x\n'
    assert S.unit_drift(text, text) == []


def test_installer_serve_mode_default_is_fleet_swap():
    assert wi._fleet_default_serve_mode() == "swap"


# --------------------------------------------------------------------------- #
# item 8 — central host resolution for the ufw rule                          #
# --------------------------------------------------------------------------- #
def test_central_host_passes_ip_through():
    assert S._central_host("http://10.99.0.3:7002") == "10.99.0.3"


# --------------------------------------------------------------------------- #
# item 11 — comfy port avoids worker + slot ports                            #
# --------------------------------------------------------------------------- #
def test_comfy_port_default_when_free(monkeypatch):
    monkeypatch.delenv("HUGPY_COMFY_PORT", raising=False)
    assert S._comfy_port(9100, 8101, 2) == 8188


def test_comfy_port_relocates_off_worker_port(monkeypatch):
    monkeypatch.delenv("HUGPY_COMFY_PORT", raising=False)
    assert S._comfy_port(8188, 8101, 2) != 8188


# --------------------------------------------------------------------------- #
# Check / render / overall_rc contract                                        #
# --------------------------------------------------------------------------- #
def test_overall_rc_fails_only_on_required_fail():
    assert S.overall_rc([S.Check("a", S.OK, "ok"),
                         S.Check("b", S.WARN, "meh", required=True),
                         S.Check("c", S.FAIL, "bad", required=False)]) == 0
    assert S.overall_rc([S.Check("d", S.FAIL, "bad", required=True)]) == 1


def test_render_lists_actions_and_result():
    out = S.render([S.Check("x", S.OK, "did it", actions=["ran: foo"]),
                    S.Check("y", S.FAIL, "broke", fix="do z", required=True)])
    assert "· ran: foo" in out and "fix: do z" in out
    assert "required check(s) FAILED" in out


# --------------------------------------------------------------------------- #
# item 9 — recorded as central-side (not changed from the worker installer)   #
# --------------------------------------------------------------------------- #
def test_stale_record_is_skip_not_fail():
    c = S.note_stale_record()
    assert c.status == S.SKIP and c.required is False

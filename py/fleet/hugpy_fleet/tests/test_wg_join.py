"""Unit tests for the one-step WireGuard join (hub side).

Covers: IP allocation, strict input validation (incl. injection attempts against
the privileged helper), the pure-python X25519 keypair (RFC 7748 vector), helper
idempotency + collision safety against hand-written peers, and the JOIN BUNDLE
shape — including that no private key or token plaintext appears in the
log/response-safe redacted view.

The privileged helper (py/tooling/hugpy_wg_peer.py) is exercised by subprocess
exactly as it runs in production, with the ``wg`` binary stubbed to ``/bin/true``
and the conf pointed at a temp file — so these tests need no real wg0 device.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hugpy_fleet.central import wg_join as wj

REPO = Path(__file__).resolve().parents[4]
HELPER = str(REPO / "py/tooling/hugpy_wg_peer.py")


def _b64key() -> str:
    return base64.b64encode(os.urandom(32)).decode()


# --------------------------------------------------------------------------- #
# IP allocation                                                               #
# --------------------------------------------------------------------------- #
def test_allocate_lowest_free():
    assert wj.allocate_ip([]) == "10.66.0.2"
    assert wj.allocate_ip(["10.66.0.2", "10.66.0.3"]) == "10.66.0.4"
    assert wj.allocate_ip(["10.66.0.2", "10.66.0.4"]) == "10.66.0.3"


def test_allocate_skips_hub_and_ignores_junk():
    assert wj.allocate_ip(["10.66.0.1", "not-an-ip", ""]) == "10.66.0.2"
    # explicit .1 in the used set must never be handed out
    assert wj.allocate_ip(["10.66.0.2"]) == "10.66.0.3"


def test_allocate_exhaustion_raises():
    used = ["10.66.0.%d" % i for i in range(2, 255)]
    with pytest.raises(wj.JoinError):
        wj.allocate_ip(used)


# --------------------------------------------------------------------------- #
# input validation                                                            #
# --------------------------------------------------------------------------- #
def test_validate_name_rejects_bad_and_injection():
    for bad in ["a;rm -rf /", "a b", "../x", "UPPER", "", "-lead",
                "$(whoami)", "`id`", "x" * 40, "na/me"]:
        with pytest.raises(wj.JoinError):
            wj.validate_name(bad)
    assert wj.validate_name("a-brain") == "a-brain"
    assert wj.validate_name("gpu_box3") == "gpu_box3"


def _helper(env, *args):
    cp = subprocess.run([sys.executable, HELPER, *args],
                        capture_output=True, text=True, env=env)
    return json.loads(cp.stdout)


@pytest.fixture
def helper_env(tmp_path):
    conf = tmp_path / "wg0.conf"
    conf.write_text("[Interface]\nAddress = 10.66.0.1/24\nListenPort = 51820\n")
    env = dict(os.environ)
    env["HUGPY_WG_CONF"] = str(conf)
    env["HUGPY_WG_BIN"] = "/bin/true"        # stub: `wg set` succeeds, `wg show` empty
    return env, conf


def test_helper_rejects_injection(helper_env):
    env, _ = helper_env
    pk = _b64key()
    for bad_name in ["a;rm -rf /", "a b", "$(whoami)", "`id`", "../../x"]:
        out = _helper(env, "add", bad_name, pk, "10.66.0.5")
        assert out["ok"] is False, bad_name
    # malformed public key
    assert _helper(env, "add", "good", "not-a-key", "10.66.0.5")["ok"] is False
    assert _helper(env, "add", "good", "AAAA", "10.66.0.5")["ok"] is False
    # ip outside subnet, hub ip, and shell-metachar ip
    for bad_ip in ["10.66.1.5", "10.66.0.1", "10.66.0.5; ls", "999.9.9.9", "10.66.0.0"]:
        assert _helper(env, "add", "good", pk, bad_ip)["ok"] is False, bad_ip


def test_helper_idempotent_and_collision_safe(tmp_path):
    conf = tmp_path / "wg0.conf"
    handwritten = base64.b64encode(bytes(range(32))).decode()
    conf.write_text(
        "[Interface]\nAddress = 10.66.0.1/24\n\n"
        "[Peer]\nPublicKey = %s\nAllowedIPs = 10.66.0.2/32\n" % handwritten)
    env = dict(os.environ, HUGPY_WG_CONF=str(conf), HUGPY_WG_BIN="/bin/true")

    pk = _b64key()
    assert _helper(env, "add", "abrain", pk, "10.66.0.3")["ok"]
    # re-add the same triple: idempotent success, still ONE block for the name
    assert _helper(env, "add", "abrain", pk, "10.66.0.3")["ok"]
    peers = _helper(env, "list")["peers"]
    named = [p for p in peers if p["name"] == "abrain"]
    assert len(named) == 1 and named[0]["ip"] == "10.66.0.3"

    pk2 = _b64key()
    # collide with the HAND-WRITTEN peer (.2) and the named peer (.3) -> refused
    assert _helper(env, "add", "other", pk2, "10.66.0.2")["ok"] is False
    assert _helper(env, "add", "other", pk2, "10.66.0.3")["ok"] is False
    # a free address is accepted
    assert _helper(env, "add", "other", pk2, "10.66.0.4")["ok"]

    used = set(_helper(env, "used-ips")["used"])
    assert {"10.66.0.1", "10.66.0.2", "10.66.0.3", "10.66.0.4"} <= used

    # remove by name drops the block; used-ips no longer counts it via a NAMED block
    assert _helper(env, "remove", "abrain")["ok"]
    peers = _helper(env, "list")["peers"]
    assert not [p for p in peers if p["name"] == "abrain"]


# --------------------------------------------------------------------------- #
# keypair                                                                      #
# --------------------------------------------------------------------------- #
def test_x25519_rfc7748_vector():
    a = bytes.fromhex(
        "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
    assert wj._x25519(a, 9).hex() == (
        "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")


def test_gen_keypair_python_shape_and_consistency():
    priv, pub = wj._gen_keypair_python()
    pb = base64.b64decode(priv)
    kb = base64.b64decode(pub)
    assert len(pb) == 32 and len(kb) == 32
    # the public key IS x25519(private, basepoint)
    assert wj._x25519(pb, 9) == kb


# --------------------------------------------------------------------------- #
# the bundle                                                                   #
# --------------------------------------------------------------------------- #
def test_build_bundle_shape_and_no_secret_leak(tmp_path, monkeypatch):
    conf = tmp_path / "wg0.conf"
    conf.write_text("[Interface]\nAddress = 10.66.0.1/24\nListenPort = 51820\n")
    monkeypatch.setenv("HUGPY_WG_CONF", str(conf))
    monkeypatch.setenv("HUGPY_WG_BIN", "/bin/true")
    monkeypatch.setenv("HUGPY_WG_PEER_HELPER", "%s %s" % (sys.executable, HELPER))
    monkeypatch.setenv("HUGPY_FLEET_STATE_DIR", str(tmp_path / "state"))
    # force the pure-python keygen path (no dependency on a wg binary being present)
    monkeypatch.setattr(wj, "_wg_binary", lambda: None)

    b = wj.build_bundle("abrain")

    assert b["ip"] == "10.66.0.2"
    assert b["central_url"] == "http://10.66.0.1:7002"
    assert b["advertise_url"] == "http://10.66.0.2:9100"
    assert b["token"].startswith("hpw_")
    wg = b["wg"]
    assert wg["address"] == "10.66.0.2/32"
    assert wg["allowed_ips"] == "10.66.0.1/32"       # HUB ONLY — not a full tunnel
    assert wg["persistent_keepalive"] == 25
    assert wg["endpoint"]                             # an endpoint is set
    assert wg["iface"]

    # the private key is present in the bundle + the client conf (served once)
    assert wg["private_key"] and wg["private_key"] in b["client_conf"]
    assert "AllowedIPs = 10.66.0.1/32" in b["client_conf"]

    # the self-contained join script carries the conf as BASE64 — the raw private
    # key never appears in cleartext in the script body.
    assert wg["private_key"] not in b["join_script"]
    assert base64.b64encode(b["client_conf"].encode()).decode() in b["join_script"]

    # the log/response-safe view drops every secret and the rendered artifacts,
    # while keeping the non-secret facts.
    red = wj.redacted_bundle(b)
    dumped = json.dumps(red)
    assert wg["private_key"] not in dumped
    assert b["token"] not in dumped
    assert "client_conf" not in red and "join_script" not in red
    assert red["ip"] == "10.66.0.2"
    assert red["wg"]["public_key"] == wg["public_key"]


def test_build_bundle_rejects_bad_name(monkeypatch):
    with pytest.raises(wj.JoinError):
        wj.build_bundle("bad name!")

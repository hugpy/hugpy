"""Central-side call-time attribution for relay-dispatched foreign GPU services
(identity-render) — the PURE correlation core.

``attribute_foreign_relay_procs`` is a pure function with zero package-level
deps (comms/media_bus imports are all lazy, inside the impure glue this test
does not touch), so it needs no flask app factory / comms / media_bus.
"""
from hugpy_fleet.central import pid_attribution as PA


# ── identity-render pid + one active mesh job -> stamped ─────────────────────
def test_identity_render_attribution():
    pr = {"models": [], "unattributed": [
        {"pid": 5001, "name": "/srv/identity-render/venv/bin/python3", "mib": 5400},
        {"pid": 6001, "name": "/usr/bin/xmrig", "mib": 8000},  # genuine squatter
    ]}
    active = [{"kind": "identity_mesh_build", "id": "job-abc",
               "model": "identity_mesh_build", "slug": "luigi"}]
    out = PA.attribute_foreign_relay_procs(pr, active)
    by_pid = {u["pid"]: u for u in out["unattributed"]}
    # hy3dgen pid stamped host_mode / service / job_id / slug / model_key
    assert by_pid[5001].get("host_mode") == "identity-render"
    assert by_pid[5001].get("service") == "identity-render"
    assert by_pid[5001].get("job_id") == "job-abc"
    assert by_pid[5001].get("slug") == "luigi"
    assert by_pid[5001].get("model_key") == "luigi"
    assert by_pid[5001].get("attribution") == "relay-job"
    # pid/name/mib preserved on the stamped row
    assert by_pid[5001].get("pid") == 5001 and by_pid[5001].get("mib") == 5400
    # genuine squatter left untouched (no host_mode)
    assert by_pid[6001].get("host_mode") is None


# ── hy3dgen marker also matches (process name variety) ───────────────────────
def test_hy3dgen_marker_and_slug_from_media_field():
    pr = {"unattributed": [{"pid": 5002,
                            "name": "/opt/hy3dgen/venv/bin/python", "mib": 5000}]}
    # No explicit slug on the job row -> falls back to model_key then model.
    active = [{"kind": "identity_mesh_build", "id": "j9", "model_key": "gio"}]
    out = PA.attribute_foreign_relay_procs(pr, active)
    e = out["unattributed"][0]
    assert e.get("host_mode") == "identity-render", "marker matches identity-render service"
    assert e.get("model_key") == "gio", "model_key falls back to job.model_key when no slug"


# ── recognized service, NO active job -> recognized-idle (ours, just idle) ───
def test_recognized_idle_no_job():
    pr = {"unattributed": [{"pid": 5001,
                            "name": "/srv/identity-render/venv/bin/python", "mib": 5400}]}
    out = PA.attribute_foreign_relay_procs(pr, [])
    e = out["unattributed"][0]
    assert e.get("host_mode") == "identity-render"
    assert e.get("attribution") == "recognized-idle"
    assert e.get("model_key") is None


# ── multiple active mesh jobs -> honest ambiguity ───────────────────────────
def test_ambiguous_multiple_jobs():
    pr = {"unattributed": [{"pid": 5001,
                            "name": "/opt/hy3dgen/venv/bin/python", "mib": 5400}]}
    active = [{"kind": "identity_mesh_build", "id": "j1", "slug": "a"},
              {"kind": "identity_mesh_build", "id": "j2", "slug": "b"}]
    out = PA.attribute_foreign_relay_procs(pr, active)
    e = out["unattributed"][0]
    assert e.get("attribution") == "relay-job-ambiguous"
    assert isinstance(e.get("job_id"), list) and set(e.get("job_id")) == {"j1", "j2"}
    assert e.get("model_key") is None, "ambiguous: model_key not asserted"


# ── a job of an UNRELATED kind never matches identity-render ─────────────────
def test_unrelated_job_kind_ignored():
    pr = {"unattributed": [{"pid": 5001,
                            "name": "/srv/identity-render/venv/bin/python", "mib": 5400}]}
    active = [{"kind": "crop", "id": "c1"}, {"kind": "studio_i2v", "id": "s1"}]
    out = PA.attribute_foreign_relay_procs(pr, active)
    e = out["unattributed"][0]
    assert e.get("attribution") == "recognized-idle", \
        "recognized service but recognized-idle (no matching kind)"


# ── degrade-safe: bad / empty inputs pass through unchanged ──────────────────
def test_degrade_safe():
    assert PA.attribute_foreign_relay_procs(None, []) is None
    empty = {"unattributed": []}
    assert PA.attribute_foreign_relay_procs(
        empty, [{"kind": "identity_mesh_build", "id": "j"}]) is empty
    no_key = {"models": []}
    assert PA.attribute_foreign_relay_procs(no_key, []) is no_key
    only_squatters = {"unattributed": [{"pid": 9, "name": "/usr/bin/xmrig", "mib": 1}]}
    out = PA.attribute_foreign_relay_procs(
        only_squatters, [{"kind": "identity_mesh_build", "id": "j", "slug": "x"}])
    assert out is only_squatters, "no recognized proc -> input returned unchanged"

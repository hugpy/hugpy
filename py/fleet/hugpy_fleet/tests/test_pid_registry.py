"""Precision model->PID registry — record/verify/forget, the recycled-PID guard,
reconcile attribution (subprocess vs in-process vs comfy + foreign squatter),
and degrade-to-empty on no-GPU inputs.

No real GPU or subprocess: a FAKE /proc probe injects process identity
(start-time + cmdline), so every path is exercised deterministically.

Runs like the other tests here:
    venv/bin/python tests/test_pid_registry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hugpy_fleet.worker import pid_registry as PR

_MIB = 1024 * 1024

ok = 0
fail = 0


def check(name, cond):
    """Pytest-native: a failed check fails the test function that made it."""
    assert cond, name


class FakeProc:
    """A settable process table: {pid: {"starttime", "cmdline", "name"}}.

    Deleting a pid = the process exited; re-adding the same pid with a DIFFERENT
    starttime = the OS recycled the number for a stranger.
    """

    def __init__(self):
        self.table = {}

    def probe(self, pid):
        return self.table.get(pid)


def new_registry(fake):
    r = PR.PidRegistry(proc_info=fake.probe)
    return r


# ── record / verify / forget ────────────────────────────────────────────────
def test_record_verify_forget():
    fake = FakeProc()
    fake.table[4242] = {"starttime": 100, "cmdline": "llama-server -m foo.gguf", "name": "llama-server"}
    r = new_registry(fake)

    rec = r.record_launch("foo/model", 4242, "subprocess")
    check("record returns the pid", rec["pid"] == 4242)
    check("record captured start_tick anchor", rec["start_tick"] == 100)
    check("verify returns live pid", r.verify("foo/model") == 4242)

    check("forget drops the record", r.forget("foo/model") is True)
    check("verify after forget -> None", r.verify("foo/model") is None)
    check("forget unknown -> False", r.forget("nope") is False)
    check("verify unknown -> None", r.verify("nope") is None)


# ── recycled-PID guard: the core "precision" ────────────────────────────────
def test_recycled_pid_guard():
    fake = FakeProc()
    fake.table[5001] = {"starttime": 555, "cmdline": "llama-server -m mA.gguf", "name": "llama-server"}
    r = new_registry(fake)
    r.record_launch("model/A", 5001, "subprocess")
    check("guard: fresh pid verifies", r.verify("model/A") == 5001)

    # Process exits -> pid vanishes from /proc.
    del fake.table[5001]
    check("guard: gone pid -> verify None", r.verify("model/A") is None)

    # OS reuses the SAME number 5001 for an unrelated process (different
    # starttime + different cmdline). The number is alive again, but it is NOT
    # our model's process.
    fake.table[5001] = {"starttime": 999, "cmdline": "python some_other_thing.py", "name": "python"}
    check("guard: recycled pid (new starttime) -> verify None", r.verify("model/A") is None)

    # sweep_dead removes the poisoned record.
    dropped = r.sweep_dead()
    check("guard: sweep_dead forgets the recycled record", "model/A" in dropped)
    check("guard: record gone after sweep", r.verify("model/A") is None)


def test_cmdline_fallback_when_no_starttime():
    # If starttime is unreadable at record time, identity falls back to cmdline.
    fake = FakeProc()
    fake.table[6001] = {"starttime": None, "cmdline": "llama-server -m secret.gguf", "name": "llama-server"}
    r = new_registry(fake)
    r.record_launch("model/B", 6001, "subprocess")
    check("cmdline-fallback: matching cmdline verifies", r.verify("model/B") == 6001)

    # Same pid, no starttime, but a DIFFERENT cmdline -> can't corroborate -> None.
    fake.table[6001] = {"starttime": None, "cmdline": "python stranger.py", "name": "python"}
    check("cmdline-fallback: changed cmdline -> verify None", r.verify("model/B") is None)


# ── reconcile: attribution across host modes + foreign squatter ─────────────
def test_reconcile_attribution():
    fake = FakeProc()
    worker_pid = 1000          # the worker python holding in-process torch models
    slot_pid = 2000            # a slot child (llama-server)
    comfy_pid = 3000           # external ComfyUI
    foreign_pid = 4000         # a ROGUE VRAM squatter the registry can't explain
    fake.table[worker_pid] = {"starttime": 10, "cmdline": "python -m ...agent", "name": "python"}
    fake.table[slot_pid] = {"starttime": 20, "cmdline": "llama-server -m g.gguf", "name": "llama-server"}
    fake.table[comfy_pid] = {"starttime": 30, "cmdline": "python main.py", "name": "ComfyUI"}
    fake.table[foreign_pid] = {"starttime": 40, "cmdline": "./miner", "name": "xmrig"}
    r = new_registry(fake)

    r.record_launch("slot/gguf", slot_pid, "subprocess")
    r.record_launch("inproc/vision", worker_pid, "in_process")
    r.record_launch("comfy/sdxl", comfy_pid, "comfy")

    # nvidia-smi ground truth (mib per pid).
    gpu_procs = {
        worker_pid: {"name": "python", "mib": 3600},   # in-process lump (torch splits it)
        slot_pid: {"name": "llama-server", "mib": 5120},
        comfy_pid: {"name": "ComfyUI", "mib": 8000},
        foreign_pid: {"name": "xmrig", "mib": 12000},  # squatter
    }
    inprocess_bytes = {"inproc/vision": {"vram_bytes": 2_000_000_000, "device": "cuda"}}
    comfy_bytes = 8000 * _MIB

    res = r.reconcile(gpu_procs, inprocess_bytes, comfy_bytes)
    att = res["attributed"]
    check("reconcile: subprocess model gets its pid's mib",
          att["slot/gguf"] == 5120 * _MIB)
    # MEASURED-TRUTH ruling (E/M 2026-07-31): the in-process model's SIZE is its
    # PID's measured nvidia-smi mib (3600 MiB — the whole lump, since it is the
    # sole in-process model on the worker PID), NOT the 2.0 GB torch estimate.
    # The estimate rides along as the PLANNED figure for the panel to show beside.
    check("reconcile: in-process model gets MEASURED pid mib (not the estimate)",
          att["inproc/vision"] == 3600 * _MIB)
    check("reconcile: comfy model gets comfy_bytes",
          att["comfy/sdxl"] == 8000 * _MIB)

    unatt_pids = {u["pid"] for u in res["unattributed"]}
    check("reconcile: foreign squatter surfaced as unattributed",
          foreign_pid in unatt_pids)
    check("reconcile: worker-python lump NOT unattributed (explained by in-proc)",
          worker_pid not in unatt_pids)
    check("reconcile: slot pid NOT unattributed", slot_pid not in unatt_pids)
    check("reconcile: comfy pid NOT unattributed (name-matched)",
          comfy_pid not in unatt_pids)
    check("reconcile: exactly one unattributed", len(res["unattributed"]) == 1)

    # snapshot reflects the attribution + live guarded aliveness.
    snap = r.snapshot_for_heartbeat()
    by_key = {m["model_key"]: m for m in snap["models"]}
    check("snapshot: three model rows", len(snap["models"]) == 3)
    check("snapshot: slot row carries reconciled vram",
          by_key["slot/gguf"]["vram_bytes"] == 5120 * _MIB)
    check("snapshot: slot row host_mode", by_key["slot/gguf"]["host_mode"] == "subprocess")
    check("snapshot: in-process row carries MEASURED vram (the lump)",
          by_key["inproc/vision"]["vram_bytes"] == 3600 * _MIB)
    check("snapshot: in-process row carries PLANNED estimate beside measured",
          by_key["inproc/vision"]["vram_bytes_planned"] == 2_000_000_000)
    check("snapshot: subprocess row has no planned estimate key",
          "vram_bytes_planned" not in by_key["slot/gguf"])
    check("snapshot: all rows alive under guard",
          all(m["alive"] for m in snap["models"]))
    check("snapshot: unattributed squatter carried through",
          foreign_pid in {u["pid"] for u in snap["unattributed"]})

    # A dead model_key in the log reads alive=False after its process exits.
    del fake.table[slot_pid]
    snap2 = r.snapshot_for_heartbeat()
    by_key2 = {m["model_key"]: m for m in snap2["models"]}
    check("snapshot: exited slot child reads alive=False",
          by_key2["slot/gguf"]["alive"] is False)


# ── degrade-to-empty: no GPU / no models ────────────────────────────────────
def test_degrade_empty():
    fake = FakeProc()
    r = new_registry(fake)
    res = r.reconcile({}, {}, None)
    check("degrade: empty reconcile attributed {}", res["attributed"] == {})
    check("degrade: empty reconcile unattributed []", res["unattributed"] == [])
    snap = r.snapshot_for_heartbeat()
    check("degrade: empty snapshot models []", snap["models"] == [])
    check("degrade: empty snapshot unattributed []", snap["unattributed"] == [])

    # None inputs (no nvidia-smi at all) also degrade cleanly.
    res2 = r.reconcile(None, None, None)
    check("degrade: None inputs -> empty attributed", res2["attributed"] == {})
    check("degrade: None inputs -> empty unattributed", res2["unattributed"] == [])


# ── idempotent heartbeat-driven population ──────────────────────────────────
def test_record_idempotent_same_process():
    fake = FakeProc()
    fake.table[7000] = {"starttime": 77, "cmdline": "llama-server -m x.gguf", "name": "llama-server"}
    r = new_registry(fake)
    first = r.record_launch("m/x", 7000, "subprocess")
    launched_at = first["launched_at"]
    # Re-observe the SAME live process (heartbeat calls record each beat).
    again = r.record_launch("m/x", 7000, "subprocess")
    check("idempotent: launched_at preserved on same pid+starttime",
          again["launched_at"] == launched_at)
    # A genuine reload (new pid, new starttime) replaces the record.
    fake.table[7001] = {"starttime": 88, "cmdline": "llama-server -m x.gguf", "name": "llama-server"}
    reloaded = r.record_launch("m/x", 7001, "subprocess")
    check("idempotent: new launch replaces record", reloaded["pid"] == 7001)
    check("idempotent: verify tracks the new pid", r.verify("m/x") == 7001)


# ── PART A: worker's own pids tagged cuda_context, not unattributed ──────────
def test_own_pid_cuda_context():
    fake = FakeProc()
    r = new_registry(fake)
    agent_pid = 100          # the worker agent process (os.getpid())
    idle_slot_pid = 101      # a slot child sharing the venv, not a recorded model
    squatter_pid = 200       # a genuine foreign squatter
    gpu_procs = {
        agent_pid: {"name": "/opt/hugpy-worker/venv/bin/python3", "mib": 120},
        idle_slot_pid: {"name": "/opt/hugpy-worker/venv/bin/python3", "mib": 90},
        squatter_pid: {"name": "/usr/bin/xmrig", "mib": 8000},
    }
    res = r.reconcile(gpu_procs, {}, None,
                      own_pids={agent_pid},
                      self_venv_marker="/opt/hugpy-worker/venv")
    frows = {f["pid"]: f for f in res["foreign"]}
    check("own: agent pid tagged cuda_context",
          frows.get(agent_pid, {}).get("host_mode") == "cuda_context")
    check("own: idle slot (venv marker) tagged cuda_context",
          frows.get(idle_slot_pid, {}).get("host_mode") == "cuda_context")
    check("own: cuda_context carries the label",
          frows.get(agent_pid, {}).get("label") == "agent CUDA context")
    check("own: cuda_context carries its mib as vram_bytes",
          frows.get(agent_pid, {}).get("vram_bytes") == 120 * _MIB)
    unatt = {u["pid"] for u in res["unattributed"]}
    check("own: agent pid NOT unattributed", agent_pid not in unatt)
    check("own: idle slot NOT unattributed", idle_slot_pid not in unatt)
    check("own: genuine squatter STILL unattributed", squatter_pid in unatt)
    # cuda_context rows ride into snapshot models (attributed, not anonymous).
    snap = r.snapshot_for_heartbeat()
    hostmodes = {m.get("pid"): m.get("host_mode") for m in snap["models"]}
    check("own: cuda_context rows appear in snapshot models",
          hostmodes.get(agent_pid) == "cuda_context"
          and hostmodes.get(idle_slot_pid) == "cuda_context")


# ── PART B: comfy call-time attribution ──────────────────────────────────────
def test_comfy_call_attribution():
    fake = FakeProc()
    r = new_registry(fake)
    comfy_pid = 300
    gpu_procs = {comfy_pid: {"name": "/opt/ComfyUI/venv/bin/python", "mib": 2400}}

    # No active call -> recognized ComfyUI, idle/unknown-model (NOT unattributed).
    res_idle = r.reconcile(gpu_procs, {}, None)
    fidle = {f["pid"]: f for f in res_idle["foreign"]}
    check("comfy: recognized when no call", fidle.get(comfy_pid, {}).get("host_mode") == "comfy")
    check("comfy: idle model_key is None", fidle.get(comfy_pid, {}).get("model_key") is None)
    check("comfy: idle carries an idle label",
          "idle" in (fidle.get(comfy_pid, {}).get("label") or ""))
    check("comfy: idle NOT unattributed",
          comfy_pid not in {u["pid"] for u in res_idle["unattributed"]})

    # An active comfy call stamps its model_key + job_id onto the comfy pid.
    r.record_foreign_call("comfy", "sdxl/juggernaut", job_id="req-123")
    res = r.reconcile(gpu_procs, {}, None)
    f = {ff["pid"]: ff for ff in res["foreign"]}
    check("comfy: active call attributes model_key",
          f.get(comfy_pid, {}).get("model_key") == "sdxl/juggernaut")
    check("comfy: active call stamps job_id",
          f.get(comfy_pid, {}).get("job_id") == "req-123")
    check("comfy: attributed vram = pid mib",
          f.get(comfy_pid, {}).get("vram_bytes") == 2400 * _MIB)

    # Ending the call returns the pid to recognized-idle.
    r.end_foreign_call("comfy", job_id="req-123")
    res2 = r.reconcile(gpu_procs, {}, None)
    f2 = {ff["pid"]: ff for ff in res2["foreign"]}
    check("comfy: after end_foreign_call -> idle again",
          f2.get(comfy_pid, {}).get("model_key") is None)


def test_foreign_call_ttl():
    fake = FakeProc()
    r = new_registry(fake)
    comfy_pid = 400
    gpu_procs = {comfy_pid: {"name": "/opt/ComfyUI/venv/bin/python", "mib": 100}}
    r.record_foreign_call("comfy", "m/x", job_id="j")
    # ttl=0 => the just-recorded call is already "leaked" -> not trusted.
    res = r.reconcile(gpu_procs, {}, None, foreign_call_ttl=0.0)
    f = {ff["pid"]: ff for ff in res["foreign"]}
    check("ttl: expired call -> recognized-idle (not stale-attributed)",
          f.get(comfy_pid, {}).get("model_key") is None)


# ── E/M byte-exact: rows sum to the nvidia-smi total, multi-in-process split ──
def test_measured_sum_byte_exact():
    """Acceptance (E/M 2026-07-31): every attributed/foreign/unattributed row's
    measured bytes sum BYTE-FOR-BYTE to nvidia-smi's compute-apps total, and a
    worker PID hosting TWO in-process models splits its measured lump between them
    (proportional to their torch estimates) rather than counting the lump twice."""
    fake = FakeProc()
    worker_pid, slot_pid, foreign_pid = 1000, 2000, 4000
    fake.table[worker_pid] = {"starttime": 10, "cmdline": "python -m ...agent", "name": "python"}
    fake.table[slot_pid] = {"starttime": 20, "cmdline": "llama-server", "name": "llama-server"}
    fake.table[foreign_pid] = {"starttime": 40, "cmdline": "./miner", "name": "xmrig"}
    r = new_registry(fake)
    r.record_launch("slot/gguf", slot_pid, "subprocess")
    r.record_launch("inproc/a", worker_pid, "in_process")
    r.record_launch("inproc/b", worker_pid, "in_process")

    gpu_procs = {
        worker_pid: {"name": "python", "mib": 3000},   # lump shared by a + b
        slot_pid: {"name": "llama-server", "mib": 5120},
        foreign_pid: {"name": "xmrig", "mib": 12000},
    }
    # a estimated 3x b -> a gets 3/4 of the lump, b 1/4 (last peer takes remainder).
    inprocess_bytes = {"inproc/a": {"vram_bytes": 900, "device": "cuda"},
                       "inproc/b": {"vram_bytes": 300, "device": "cuda"}}
    res = r.reconcile(gpu_procs, inprocess_bytes, None)
    att = res["attributed"]
    lump = 3000 * _MIB
    check("split: two in-process shares sum to the measured lump (exact)",
          att["inproc/a"] + att["inproc/b"] == lump)
    check("split: proportional to estimate (a=3/4 of lump)",
          att["inproc/a"] == lump * 900 // 1200)
    check("split: b takes the remainder", att["inproc/b"] == lump - att["inproc/a"])

    total_smi = sum(m["mib"] for m in gpu_procs.values()) * _MIB
    attributed_sum = sum(att.values())
    foreign_sum = sum(int(f.get("vram_bytes") or 0) for f in res["foreign"])
    unattr_sum = sum(int(u.get("mib") or 0) * _MIB for u in res["unattributed"])
    check("byte-exact: attributed + foreign + unattributed == nvidia-smi total",
          attributed_sum + foreign_sum + unattr_sum == total_smi)


def main():
    test_record_verify_forget()
    test_recycled_pid_guard()
    test_cmdline_fallback_when_no_starttime()
    test_reconcile_attribution()
    test_degrade_empty()
    test_record_idempotent_same_process()
    test_own_pid_cuda_context()
    test_comfy_call_attribution()
    test_foreign_call_ttl()
    test_measured_sum_byte_exact()
    print("\n%d ok, %d failed" % (ok, fail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

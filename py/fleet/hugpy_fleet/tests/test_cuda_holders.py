"""/ops/cuda-holders is read-only and answers without CUDA."""
import sys

from hugpy_fleet.worker import agent as A


def test_no_torch_cuda_reports_false(monkeypatch):
    class _T:
        class cuda:
            @staticmethod
            def is_initialized():
                return False
    monkeypatch.setitem(sys.modules, "torch", _T)
    assert A._cuda_holders() == {"torch_cuda": False}


def test_route_answers(monkeypatch):
    monkeypatch.setattr(A, "_cuda_holders", lambda limit=12: {"torch_cuda": False, "limit": limit})
    c = A.build_app(A.WorkerState(name="t", url=None, worker_id="w")).test_client()
    r = c.get("/ops/cuda-holders?limit=3")
    assert r.status_code == 200 and r.get_json()["limit"] == 3


def test_referrer_names_finds_a_module_global():
    import types
    mod = types.ModuleType("holder_mod")
    obj = object.__new__(type("Big", (), {}))
    mod.KEEP = obj
    sys.modules["holder_mod"] = mod
    try:
        assert any("holder_mod.KEEP" in n for n in A._referrer_names(obj))
    finally:
        sys.modules.pop("holder_mod", None)

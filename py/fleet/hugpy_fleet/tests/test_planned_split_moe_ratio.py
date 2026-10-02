"""The Memory projection uses the model's tensor ratio, not layer capacity."""
import importlib

W = importlib.import_module("hugpy_fleet.central.workers")


def test_enabled_moe_projects_model_size_by_expert_tensor_ratio(monkeypatch):
    monkeypatch.setattr(W, "_model_size_bytes", lambda _key: 1000)
    monkeypatch.setattr(W, "bnb_enabled", lambda *_args: False)
    monkeypatch.setattr(W, "derived_default_allocation",
                        lambda *_args, **_kwargs: {"mode": "max-ram", "spill": {}})
    monkeypatch.setattr(W, "planned_need", lambda *_args, **_kwargs: {
        "is_moe": True, "gpu_bytes": 7000, "ram_bytes": 9000,
        "kv_bytes": 400, "n_cpu_moe": 39, "block_count": 48,
    })
    monkeypatch.setattr(W, "moe_effective", lambda *_args: True)
    monkeypatch.setattr(W, "_model_moe_detail", lambda _key: {
        "is_moe": True, "expert_bytes": 800, "non_expert_bytes": 200,
    })
    monkeypatch.setattr(W, "_resident_row", lambda *_args: None)

    plan = W.planned_split({"models": ["test-moe"]}, "test-moe")

    assert plan["split_basis"] == "model-moe-ratio"
    assert plan["gpu_bytes"] == 200
    assert plan["ram_bytes"] == 800
    assert plan["gpu_bytes"] + plan["ram_bytes"] == plan["size_bytes"] == 1000
    assert plan["moe_ratio"] == {
        "shared_bytes": 200, "expert_bytes": 800,
        "gpu_fraction": 0.2, "ram_fraction": 0.8,
    }


def test_moe_off_keeps_the_normal_allocation_projection(monkeypatch):
    monkeypatch.setattr(W, "_model_size_bytes", lambda _key: 1000)
    monkeypatch.setattr(W, "bnb_enabled", lambda *_args: False)
    monkeypatch.setattr(W, "derived_default_allocation",
                        lambda *_args, **_kwargs: {"mode": "max-ram", "spill": {}})
    monkeypatch.setattr(W, "planned_need", lambda *_args, **_kwargs: {
        "is_moe": True, "gpu_bytes": 700, "ram_bytes": 900,
        "kv_bytes": 40, "n_cpu_moe": 39, "block_count": 48,
    })
    monkeypatch.setattr(W, "moe_effective", lambda *_args: False)
    monkeypatch.setattr(W, "_model_moe_detail", lambda _key: {
        "is_moe": True, "expert_bytes": 800, "non_expert_bytes": 200,
    })
    monkeypatch.setattr(W, "_resident_row", lambda *_args: None)

    plan = W.planned_split({"models": ["test-moe"]}, "test-moe")

    assert plan["split_basis"] == "need-function"
    assert (plan["gpu_bytes"], plan["ram_bytes"]) == (700, 900)
    assert "moe_ratio" not in plan

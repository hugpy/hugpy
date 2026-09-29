from hugpy_fleet.worker.host_activity import inventory
from hugpy_fleet.worker import external_residents


def test_host_inventory_keeps_unknown_gpu_processes_visible():
    external = [{"model_key": "vllm:qwen", "pid": 10, "pids": [10, 11],
                 "gpu_indices": [2, 3], "evictable": True},
                {"model_key": "coder-next:q4", "pid": 12, "pids": [12],
                 "gpu_indices": [0], "evictable": False, "immutable": True,
                 "api_url": "http://127.0.0.1:7005"}]
    processes = [{"pid": 10, "process": "VLLM::Worker_TP0", "gpu_indices": [2], "vram_mib": 100},
                 {"pid": 11, "process": "VLLM::Worker_TP1", "gpu_indices": [3], "vram_mib": 100},
                 {"pid": 12, "process": "llama-server", "gpu_indices": [0], "vram_mib": 200},
                 {"pid": 13, "process": "foreign-server", "gpu_indices": [1], "vram_mib": 50}]
    rows = inventory(external, [], processes)
    assert len(rows) == 3
    assert next(r for r in rows if r["model_key"] == "vllm:qwen")["vram_mib"] == 200
    coder = next(r for r in rows if r["model_key"] == "coder-next:q4")
    assert coder["immutable"] and coder["api_available"] and not coder["evictable"]
    unknown = next(r for r in rows if r["kind"] == "observed")
    assert unknown["pids"] == [13] and unknown["evictable"] is None


def test_immutable_external_policy_cannot_be_toggled():
    external_residents.clear()
    external_residents.register("coder-next:q4", 12, immutable=True, evictable=False)
    result = external_residents.set_policy("coder-next:q4", evictable=True)
    assert result["evictable"] is False
    external_residents.clear()

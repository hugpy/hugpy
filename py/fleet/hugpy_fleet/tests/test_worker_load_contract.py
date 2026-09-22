from hugpy_fleet.worker import agent


def test_same_contract_reuses_resident(monkeypatch):
    agent._LOAD_CONTRACTS.clear()
    evictions = []
    monkeypatch.setattr(agent, "loaded_model_keys", lambda: ["m"])
    monkeypatch.setattr(agent, "_evict_model",
                        lambda *_a, **_k: evictions.append(True) or {"evicted": True})
    agent._LOAD_CONTRACTS["m"] = agent._load_contract({"n_gpu_layers": -1})
    agent._prepare_load_contract(object(), "m", {"n_gpu_layers": -1})
    assert evictions == []


def test_changed_contract_evicts_before_reuse(monkeypatch):
    agent._LOAD_CONTRACTS.clear()
    evictions = []
    monkeypatch.setattr(agent, "loaded_model_keys", lambda: ["m"])
    monkeypatch.setattr(agent, "_evict_model",
                        lambda *_a, **_k: evictions.append(True) or {"evicted": True})
    agent._LOAD_CONTRACTS["m"] = agent._load_contract({"n_gpu_layers": -1})
    agent._prepare_load_contract(object(), "m", {"n_gpu_layers": "off"})
    assert evictions == [True]


def test_first_explicit_contract_rebuilds_unknown_resident(monkeypatch):
    agent._LOAD_CONTRACTS.clear()
    evictions = []
    monkeypatch.setattr(agent, "loaded_model_keys", lambda: ["m"])
    monkeypatch.setattr(agent, "_evict_model",
                        lambda *_a, **_k: evictions.append(True) or {"evicted": True})
    agent._prepare_load_contract(object(), "m", {"n_gpu_layers": 0})
    assert evictions == [True]

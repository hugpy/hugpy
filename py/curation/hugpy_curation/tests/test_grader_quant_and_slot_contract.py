from hugpy_curation.review.fleet_grading import _quants
from hugpy_engine.serve.slots import status_satisfies_opts


def test_grader_excludes_incomplete_shard_groups():
    serving = {"available_gguf_detail": [
        {"filename": "model-00002-of-00002.gguf", "bytes": 10,
         "complete": False},
        {"filename": "model-Q2_K.gguf", "bytes": 20},
    ]}
    assert _quants({}, serving, "w") == [
        {"quant": "model-Q2_K.gguf", "size_bytes": 20}]


def test_slot_reuse_requires_requested_allocation_and_quant():
    status = {"n_gpu_layers": -1, "n_cpu_moe": 0,
              "model_path": "/models/model-Q2_K.gguf"}
    assert status_satisfies_opts(
        status, {"n_gpu_layers": -1, "path": "model-Q2_K.gguf"})
    assert not status_satisfies_opts(status, {"n_gpu_layers": "off"})
    assert not status_satisfies_opts(status, {"path": "model-IQ1_S.gguf"})

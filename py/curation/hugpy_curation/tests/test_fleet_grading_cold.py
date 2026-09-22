"""Cold grading prepares state but lets ordinary inference provision it."""
from unittest.mock import Mock, call

from hugpy_curation.review import fleet_grading


def test_cold_reset_only_evicts_memory_and_worker_cache():
    client = Mock()
    client.request.side_effect = [{"evicted": True}, {"removed": True}]
    lane = {"worker_id": "w/1", "model": "org/model"}

    result = fleet_grading._cold_reset(client, lane)

    assert client.request.call_args_list == [
        call("/llm/workers/w%2F1/evict", "POST", {"model_key": "org/model", "force": True}),
        call("/llm/workers/w%2F1/cache-evict", "POST", {"model_key": "org/model"}),
    ]
    assert result == {"evict": {"evicted": True},
                      "cache_evict": {"removed": True}}

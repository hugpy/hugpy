"""Live worker activity: heartbeat/edge merge rules and the feed scheduler."""
from concurrent.futures import ThreadPoolExecutor
import threading

from hugpy_fleet.central import feeds
from hugpy_fleet.central import heartbeat_db as hb


def test_finish_edge_beats_delayed_busy_heartbeat():
    records = [{'worker_id': 'w', 'ts': 120, 'payload': {
        'activity_sampled_at': 100,
        'allocations': [{'slot_id': '1', 'model_key': 'qwen', 'busy': True}]}}]
    live = [{'id': 'w', 'status': 'online', 'answering': ['qwen']}]
    assert hb._merge_activity(live, records, [('w', '1', 'qwen', False, 110)])[0]['answering'] == []


def test_newer_full_sample_supersedes_old_start_edge():
    records = [{'worker_id': 'w', 'ts': 120, 'payload': {'activity_sampled_at': 115}}]
    live = [{'id': 'w', 'status': 'online', 'answering': []}]
    assert hb._merge_activity(live, records, [('w', '1', 'qwen', True, 110)])[0]['answering'] == []


def test_start_edge_updates_only_its_slot():
    records = [{'worker_id': 'w', 'ts': 100, 'payload': {'allocations': [
        {'slot_id': '1', 'model_key': 'old'}, {'slot_id': '2', 'model_key': 'other'}]}}]
    live = [{'id': 'w', 'status': 'online', 'answering': ['old', 'other']}]
    assert hb._merge_activity(live, records, [('w', '1', 'qwen', True, 110)])[0]['answering'] == ['other', 'qwen']


def test_slow_roster_does_not_block_or_duplicate_liveness(monkeypatch):
    release = threading.Event()
    roster_started = threading.Event()
    live_done = threading.Event()
    counts = {'workers': 0, 'liveness': 0}
    def build(app, feed, builder, transform):
        counts[feed] += 1
        if feed == 'workers':
            roster_started.set()
            release.wait(3)
        else:
            live_done.set()
    monkeypatch.setattr(feeds, 'FEEDS', {'workers': ('slow', 1, None), 'liveness': ('fast', 1, None)})
    monkeypatch.setattr(feeds, '_refresh_one', build)
    pending, last = {}, {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            feeds._schedule_due(None, pool, pending, last, 10)
            assert roster_started.wait(1)
            assert live_done.wait(1)
            pending['liveness'].result(timeout=1)
            live_done.clear()
            feeds._schedule_due(None, pool, pending, last, 12)
            assert live_done.wait(1)
            assert counts == {'workers': 1, 'liveness': 2}
        finally:
            release.set()

"""Control-plane setup fetches (the worker roster, the model catalog) ride out a
WARMING/BLIPPING central with bounded backoff and never kill the run on one blip
— the 2026-09-24 incident (a resumed benchmark died on a single /llm/workers
timeout seconds after a promotion restarted central). A hard error still fails
fast; a persistent one gives up after the window."""
import urllib.error

import pytest

from hugpy_curation.review.fleet_grading import (
    FleetError, retrying_control_request, _transient_control_error)


class _Client:
    """Loopback client stub: ``request`` replays a scripted list of outcomes,
    raising the exceptions and returning the payloads in order."""
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def request(self, path, method="GET", body=None, timeout=None):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _sleeps():
    waited = []
    return waited, (lambda s: waited.append(s))


def test_transient_then_success_rides_it_out():
    # Two warm-up faults (a timeout, a refused connection) then the real roster.
    client = _Client([TimeoutError("timed out"),
                      urllib.error.URLError("Connection refused"),
                      {"workers": [{"id": "w1"}]}])
    waited, sleep = _sleeps()
    logs = []
    rows = retrying_control_request(client, "/llm/workers", "workers",
                                    sleep=sleep, log=lambda k, v: logs.append((k, v)))
    assert rows == [{"id": "w1"}]
    assert client.calls == 3
    assert len(waited) == 2                       # backed off once per transient
    assert waited[0] <= waited[1]                 # bounded, non-decreasing backoff
    assert logs and logs[0][0] == "notice" and logs[0][1]["path"] == "/llm/workers"


def test_transient_5xx_from_fleeterror_is_retried():
    client = _Client([FleetError("HTTP 502 at /llm/workers: Bad Gateway"),
                      {"workers": []}])
    waited, sleep = _sleeps()
    assert retrying_control_request(client, "/llm/workers", "workers", sleep=sleep) == []
    assert client.calls == 2 and len(waited) == 1


def test_hard_4xx_fails_fast_without_retry():
    client = _Client([FleetError("HTTP 400 at /llm/workers: bad request")])
    waited, sleep = _sleeps()
    with pytest.raises(FleetError):
        retrying_control_request(client, "/llm/workers", "workers", sleep=sleep)
    assert client.calls == 1 and waited == []     # a genuine no is not retried


def test_persistent_transient_gives_up_after_the_window():
    client = _Client([TimeoutError("timed out")])
    waited, sleep = _sleeps()
    # total_s=0: the first failure is already past the deadline -> raise, bounded.
    with pytest.raises(TimeoutError):
        retrying_control_request(client, "/llm/workers", "workers", total_s=0.0, sleep=sleep)
    assert client.calls == 1


def test_transient_classifier():
    assert _transient_control_error(TimeoutError("timed out"))
    assert _transient_control_error(ConnectionResetError("reset"))
    assert _transient_control_error(urllib.error.URLError("Connection refused"))
    assert _transient_control_error(FleetError("HTTP 503 at /x: unavailable"))
    assert not _transient_control_error(FleetError("HTTP 404 at /x: gone"))
    assert not _transient_control_error(ValueError("nonsense"))

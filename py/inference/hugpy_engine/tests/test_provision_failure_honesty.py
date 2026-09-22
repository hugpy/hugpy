"""The cause of a provisioning failure must survive to the operator's chat.

THE INCIDENT (2026-07-28). computron's drive was 100% full. ``_provision_now``
caught ``OSError: [Errno 28] No space left on device`` from BOTH the central
parallel transfer and the archive fallback, wrote the diagnosis into
``central_reason`` — and then returned a bare ``False``, at which point every
word of it was gone. What reached the operator was:

    The 'computron' worker could not complete this request:
    could not fetch model Qwen2.5-7B-Instruct-GGUF from central or HF

— a sentence equally true of a full disk, a revoked token, a 404 and a dead NIC.
Finding the truth cost an ssh session and a journalctl read.
"""
import errno

import pytest

from hugpy_storage import provision as pv


@pytest.fixture(autouse=True)
def _clean():
    with pv._FAILURES_LOCK:
        pv._FAILURES.clear()
    yield
    with pv._FAILURES_LOCK:
        pv._FAILURES.clear()


def test_enospc_is_recorded_as_a_human_sentence():
    exc = OSError(errno.ENOSPC, "No space left on device")
    pv._record_failure("Q", "central-transfer",
                       "parallel transfer failed: OSError: [Errno 28] "
                       "No space left on device", exc=exc, dest_path="/")
    got = pv.last_failure("Q")
    assert got["errno_name"] == "ENOSPC"
    assert got["error_class"] == "OSError"
    assert got["source"] == "central-transfer"
    assert got["human"].startswith("disk full (ENOSPC) on /")
    assert " free of " in got["human"]


def test_a_reason_with_no_exception_still_survives():
    pv._record_failure("Q", "archive",
                       "central cannot provide the files (archive refused)")
    got = pv.last_failure("Q")
    assert got["errno_name"] is None
    # `human` falls back to the reason — still infinitely better than the
    # flattened "from central or HF".
    assert got["human"] == "central cannot provide the files (archive refused)"


def test_last_failure_is_none_for_an_unknown_model():
    assert pv.last_failure("never-heard-of-it") is None


def test_a_successful_provision_clears_a_stale_cause():
    """A "disk full" must never be reported against a model that has since
    landed — the operator would chase a problem that no longer exists."""
    pv._record_failure("Q", "hf", "boom",
                       exc=OSError(errno.ENOSPC, "x"), dest_path="/")
    assert pv.last_failure("Q") is not None
    pv.clear_failure("Q")
    assert pv.last_failure("Q") is None


def test_the_record_is_bounded():
    """A worker that fails to provision thousands of distinct keys must not
    grow a dict forever."""
    for i in range(pv._FAILURES_MAX + 50):
        pv._record_failure(f"m{i}", "hf", "nope")
    with pv._FAILURES_LOCK:
        assert len(pv._FAILURES) <= pv._FAILURES_MAX
    # The NEWEST cause is the one that matters and must have survived.
    assert pv.last_failure(f"m{pv._FAILURES_MAX + 49}") is not None


def test_record_failure_never_raises_on_a_hostile_exception():
    class Weird(OSError):
        def __str__(self):
            raise RuntimeError("str is broken")

    pv._record_failure("Q", "hf", "reason", exc=Weird(), dest_path="/")


# --------------------------------------------------------------------------- #
# the chat message the relay ends up classifying and rendering
# --------------------------------------------------------------------------- #

def test_the_honest_message_is_still_a_permanent_load_failure():
    """The richer wording must NOT become retryable.

    Retrying a request against a 100%-full drive is a storm, not a recovery, so
    the central relay has to keep classifying this as permanent. Both the old
    generic wording and the new named one are markers."""
    from hugpy_engine.resolvers import remote

    honest = ("could not provision Qwen2.5-7B-Instruct-GGUF: disk full "
              "(ENOSPC) on /mnt/storage — 0 B free of 938 GB")
    generic = "could not fetch model Qwen2.5-7B-Instruct-GGUF from central or HF"
    assert remote._is_permanent_load_error(honest)
    assert remote._is_permanent_load_error(generic)
    # And an ordinary transient failure is still transient (hold + retry).
    assert not remote._is_permanent_load_error("connection reset by peer")


def test_the_message_does_not_double_the_worker_name():
    """The relay's _humanize_worker_error already stamps "The '<worker>'
    worker could not complete this request: " on the front. The worker-side
    message must not name itself again."""
    from hugpy_engine.resolvers import remote

    worker_side = ("could not provision Qwen2.5-7B-Instruct-GGUF: disk full "
                   "(ENOSPC) on /mnt/storage — 0 B free of 938 GB")
    assert "computron" not in worker_side
    final = remote._humanize_worker_error("computron", worker_side)
    assert final.count("computron") == 1
    assert "disk full (ENOSPC)" in final
    assert "\n" not in final            # one line, no traceback

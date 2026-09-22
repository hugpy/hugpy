import functools
import logging
import traceback
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Distinct from None so a real None return isn't mistaken for failure.
FAILED = object()


def caught(fn, *args, default=FAILED, label=None, reraise=False, **kwargs):
    """Call fn(*args, **kwargs); on exception, log the full traceback.

    Returns fn's result, or `default` (sentinel FAILED) if it raised.
    Set reraise=True to log-and-propagate instead of swallowing.
    """
    label = label or getattr(fn, "__name__", repr(fn))
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("caught: %s failed", label)   # logger.exception => full stack
        if reraise:
            raise
        return default


@contextmanager
def caught_block(label, *, reraise=False):
    """Wrap an arbitrary block:  with caught_block('loading config'): ..."""
    try:
        yield
    except Exception:
        logger.exception("caught: %s failed", label)
        if reraise:
            raise


def catching(label=None, *, default=FAILED, reraise=False):
    """Decorator form:  @catching()  on any function."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return caught(fn, *args, default=default,
                          label=label or fn.__name__, reraise=reraise, **kwargs)
        return wrapper
    return deco


def attempt(fn, *args, label=None, **kwargs):
    """Run fn; return (ok, value, exc). Logs full traceback on failure.

    Unlike caught(), this never swallows the exception object — the caller
    gets it back so it can build a typed error result, decide log level by
    exception type, etc. Never reraises; that's the caller's call.
    """
    label = label or getattr(fn, "__name__", repr(fn))
    try:
        return True, fn(*args, **kwargs), None
    except Exception as exc:
        logger.exception("attempt: %s failed", label)
        return False, None, exc

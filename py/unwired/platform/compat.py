"""Compatibility shims for abstract_utilities / abstract_security / abstract_webtools.

These packages require ezodf/geopandas/moviepy which can't always be built.
This module provides the subset of their APIs actually used in this codebase,
implemented with only stdlib + pydantic + requests.
"""
from __future__ import annotations

import base64
import copy
import glob
import importlib
import json
import logging
import os
import os.path as osp
import re
import sys
import tempfile
import threading
import unicodedata
import urllib
import uuid
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlunparse, unquote, quote, urlparse, parse_qs


# ---------------------------------------------------------------------------
# get_env_value  (abstract_security.get_env_value)
# ---------------------------------------------------------------------------

def get_env_value(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Logging  (abstract_utilities.get_logFile)
# ---------------------------------------------------------------------------

def get_logFile(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# make_list  (abstract_utilities.make_list)
# ---------------------------------------------------------------------------

def make_list(obj: Any) -> list:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, (tuple, set, frozenset)):
        return list(obj)
    return [obj]


# ---------------------------------------------------------------------------
# get_any_value  (abstract_utilities.get_any_value)
# ---------------------------------------------------------------------------

def get_any_value(obj: Any, keys: Any, default: Any = None) -> Any:
    keys = make_list(keys)
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
    else:
        for k in keys:
            v = getattr(obj, k, None)
            if v is not None:
                return v
    return default


# ---------------------------------------------------------------------------
# JSON helpers  (abstract_utilities.safe_read_from_json / safe_load_from_json)
# ---------------------------------------------------------------------------

def safe_read_from_json(path: str) -> Optional[dict]:
    try:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


safe_load_from_json = safe_read_from_json   # alias


# ---------------------------------------------------------------------------
# SingletonMeta  (abstract_utilities.SingletonMeta)
# ---------------------------------------------------------------------------

class SingletonMeta(type):
    """Thread-safe singleton metaclass."""
    _instances: dict = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# ---------------------------------------------------------------------------
# nullProxy  (abstract_utilities.import_utils.…nullProxy.nullProxy)
# ---------------------------------------------------------------------------

class nullProxy:
    """Returned by lazy_import when a package is not installed.

    Every attribute access returns self so chained imports work without
    raising at import time. Truthiness is False so callers can check
    `if isinstance(obj, nullProxy)`.
    """
    def __init__(self, name: str = ""):
        object.__setattr__(self, "_name", name)

    def __getattr__(self, item: str):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __bool__(self):
        return False

    def __repr__(self):
        return f"<nullProxy: {object.__getattribute__(self, '_name')}>"

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


# ---------------------------------------------------------------------------
# lazy_import  (abstract_utilities.lazy_import)
# ---------------------------------------------------------------------------

def lazy_import(name: str) -> Any:
    """Import *name* and return it, or return nullProxy if not installed."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return nullProxy(name)
    except Exception:
        return nullProxy(name)


# ---------------------------------------------------------------------------
# requests  (re-export standard requests)
# ---------------------------------------------------------------------------

try:
    import requests as _requests
    requests = _requests
except ImportError:
    requests = nullProxy("requests")


# ---------------------------------------------------------------------------
# derive_approved_headers_user_agent_session_for_url
# (abstract_webtools stub — returns a basic requests.Session)
# ---------------------------------------------------------------------------

def derive_approved_headers_user_agent_session_for_url(url: str):
    try:
        import requests as _r
        session = _r.Session()
        session.headers.update({"User-Agent": "abstract_hugpy/1.0"})
        return {}, session
    except ImportError:
        return {}, None

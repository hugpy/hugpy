"""The dossier store as the oracle sees it — ``hugpy_oracle.providers.DossierSource``.

The oracle's interim ledger lists every discovery dossier as an interim
record, but the oracle sits BELOW curation and must not read the store's
files. It declares a narrow read-only Protocol instead::

    root() -> str | None                      the store root, for provenance
    iter_dossiers() -> Iterable[Mapping]      {criteria, hub_id, payload, path}

:class:`DossierStoreSource` implements it over :mod:`hugpy_curation.dossier
.store` — the same ``root_dir()`` / layout the store writes, so a dossier
filed by the review shows up in the ledger without a second index. Files the
store keeps beside dossiers (``_radar.json``, ``*.tmp``) are skipped; a file
that will not parse is skipped too (the ledger reports what it can read, a
corrupt file is the store's problem, not the ledger's).

Installed by :func:`hugpy_curation.install_providers` at composition time
(``hugpy_server`` startup, or ``hugpy-curation install-providers``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = ["DossierStoreSource", "install_dossier_source"]


class DossierStoreSource:
    """Read-only view of the discovery dossier store.

    ``root`` pins the store directory (tests); by default the store's own
    :func:`~hugpy_curation.dossier.store.root_dir` is asked on every call, so
    a ``DOSSIER_DIR`` change is honoured without re-installing the source.
    """

    def __init__(self, root: Optional[str] = None) -> None:
        self._root = root

    def root(self) -> Optional[str]:
        if self._root:
            return self._root
        try:
            from hugpy_curation.dossier.store import root_dir
            return root_dir()
        except Exception as exc:  # noqa: BLE001 — "no root" is the honest answer
            logger.debug("dossier source: root unavailable (%s)", exc)
            return None

    def iter_dossiers(self) -> Iterator[Mapping[str, Any]]:
        root = self.root()
        if not root or not os.path.isdir(root):
            return
        try:
            criteria_names = sorted(os.listdir(root))
        except OSError:
            return
        for criteria in criteria_names:
            directory = os.path.join(root, criteria)
            if not os.path.isdir(directory):
                continue
            try:
                names = sorted(os.listdir(directory))
            except OSError:
                continue
            for name in names:
                if not name.endswith(".json") or name.startswith("_"):
                    continue
                path = os.path.join(directory, name)
                payload = _read_json(path)
                if payload is None:
                    continue
                hub_id = str(payload.get("hub_id") or name[:-5])
                record_criteria = str(payload.get("criteria") or criteria)
                yield {"criteria": record_criteria, "hub_id": hub_id,
                       "payload": payload, "path": path}


def _read_json(path: str) -> Optional[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.debug("dossier source: skipping %s (%s)", path, exc)
        return None
    return data if isinstance(data, dict) else None


def install_dossier_source(root: Optional[str] = None) -> DossierStoreSource:
    """Register a :class:`DossierStoreSource` with the oracle and return it."""
    from hugpy_oracle.providers import set_dossier_source

    source = DossierStoreSource(root)
    set_dossier_source(source)
    return source

"""hugpy_curation — discovery dossiers and the HF model review pipeline.

Two subpackages, one lifecycle:

* :mod:`hugpy_curation.review` — search, screen, download (through
  ``hugpy_storage``), smoke-load, judge; the on-box review record and the
  worker->central push.
* :mod:`hugpy_curation.dossier` — the comprehensive per-model dossier the
  review files for every survivor (card digest, weights, research,
  community, trial, verdict) and the gem radar.

Seams:

* :mod:`hugpy_curation.config` — every state root, env-overridable.
* :mod:`hugpy_curation.providers` — the fleet doctrine the dossier judge
  reads (``DoctrineSource``; null default).
* :mod:`hugpy_curation.dossier.oracle_source` — the dossier store as the
  oracle's ``DossierSource``; :func:`install_providers` registers it.

This ``__init__`` is deliberately light: nothing heavy (huggingface_hub,
llama_cpp, the review sqlite, the job mirror) is imported here.
"""

from __future__ import annotations

from typing import Any, Optional

try:  # the installed distribution's version: the workspace tag/commit, never a literal
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("hugpy-curation")
except Exception:  # noqa: BLE001 — source tree without metadata
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "install_providers"]


def install_providers(doctrine_source: Optional[Any] = None,
                      dossier_root: Optional[str] = None) -> dict[str, Any]:
    """Wire curation's seams in this process; returns what was wired.

    * Installs :class:`~hugpy_curation.dossier.oracle_source.DossierStoreSource`
      into ``hugpy_oracle.providers`` (``set_dossier_source``) so the oracle's
      interim ledger can list discovery dossiers.
    * Installs ``doctrine_source`` (anything with ``latest()``) into
      :mod:`hugpy_curation.providers` when given; the server passes the
      fleet's ``hugpy_fleet.doctrine`` adapter here. Without one, the
      doctrine falls back to whatever the oracle has installed, else null.

    Called by ``hugpy_server`` at startup and by
    ``hugpy-curation install-providers``.
    """
    from hugpy_curation.dossier.oracle_source import install_dossier_source
    from hugpy_curation.providers import set_doctrine_source

    report: dict[str, Any] = {}
    source = install_dossier_source(dossier_root)
    report["dossier_source"] = {"installed": True, "root": source.root()}
    if doctrine_source is not None:
        set_doctrine_source(doctrine_source)
        report["doctrine_source"] = {"installed": True,
                                     "impl": type(doctrine_source).__name__}
    else:
        report["doctrine_source"] = {"installed": False, "impl": "default"}
    return report

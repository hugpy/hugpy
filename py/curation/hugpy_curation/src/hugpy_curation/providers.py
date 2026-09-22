"""Provider seams: what curation may ask of packages it must not import.

Curation (dossiers and the review pipeline) sits below ``hugpy_fleet`` and
``hugpy_server``. The one thing it used to reach up for is the fleet's
environment DOCTRINE (k118): the dossier judge prompt carries a one-line
"fleet note" saying whether a doctrine snapshot exists and which version, so
the judge can flag a candidate that needs a dependency the reference box
lacks. That is doctrine *data*, not fleet policy — the verdict rule itself
("no evidence, no verdict") is curation's and stays in
:mod:`hugpy_curation.dossier.verdicts`.

So the doctrine is read through the Protocol below and the composition root
installs the real implementation:

* :class:`DoctrineSource` — ``latest()`` returns the current doctrine record
  (anything with a ``version`` attribute or key) or ``None`` when central
  holds none. Implemented by ``hugpy_fleet.doctrine`` (``latest``).

Default (nothing installed): the source installed in
``hugpy_oracle.providers`` is used when there is one — the server already
installs the fleet's doctrine there and curation may import the oracle, so a
single ``set_doctrine_source`` on the oracle serves both packages — else
:class:`NullDoctrineSource`, whose ``latest()`` is ``None`` ("no snapshot
yet"). The judge note then says so and the verdict is judged from the
dossier's own numbers, exactly as before on a box without the fleet package.

Wiring: ``hugpy_server`` calls :func:`hugpy_curation.install_providers`
(which also registers the dossier store with the oracle) or
:func:`set_doctrine_source` directly; tests call :func:`reset_providers`.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

__all__ = [
    "DoctrineSource",
    "NullDoctrineSource",
    "get_doctrine_source",
    "set_doctrine_source",
    "reset_providers",
]


@runtime_checkable
class DoctrineSource(Protocol):
    """The fleet environment doctrine central currently holds (read-only)."""

    def latest(self) -> Any:
        """The current doctrine record, or ``None`` when central holds none."""
        ...


class NullDoctrineSource:
    """No doctrine: ``latest()`` is ``None``."""

    def latest(self):
        return None


_NULL = NullDoctrineSource()
_installed: Optional[DoctrineSource] = None


def _oracle_installed() -> Optional[DoctrineSource]:
    """The doctrine source the composition root gave the ORACLE, if any.

    The oracle's default is its own null source; only an explicitly installed
    one is worth delegating to, so the oracle's registry is inspected rather
    than its accessor called (its null default would mask ours for no gain)."""
    try:
        from hugpy_oracle import providers as oracle_providers
    except Exception:  # noqa: BLE001 — the oracle is optional at runtime here
        return None
    impl = getattr(oracle_providers, "_providers", {}).get("doctrine_source")
    if impl is None or not hasattr(impl, "latest"):
        return None
    return impl


def get_doctrine_source() -> DoctrineSource:
    """The installed source, else the oracle's installed source, else null."""
    if _installed is not None:
        return _installed
    return _oracle_installed() or _NULL


def set_doctrine_source(impl: Optional[DoctrineSource]) -> None:
    """Install (or, with ``None``, remove) the doctrine source."""
    global _installed
    if impl is not None and not hasattr(impl, "latest"):
        raise TypeError("doctrine source must provide latest()")
    _installed = impl


def reset_providers() -> None:
    """Drop every installed provider (tests)."""
    set_doctrine_source(None)

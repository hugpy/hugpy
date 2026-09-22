"""Ecosystem distribution versions, resolved lazily through ``importlib``.

The monolith's sentinel compared workers against the monolith's own ``__version__``.
After the partition the code a worker runs is ``hugpy-fleet`` and the code
central runs is ``hugpy-server``, so a version-skew case now carries those
distribution versions (as installed in the sentinel's own environment) as
evidence. Nothing here imports the packages themselves — only their
installed metadata — so the sentinel stays runnable on a box where neither
is installed (the value is then ``None``).
"""

from __future__ import annotations

from typing import Iterable, Optional

DEFAULT_DISTRIBUTIONS = ("hugpy-fleet", "hugpy-server")


def distribution_version(name: str) -> Optional[str]:
    """Installed version of one distribution, or None when it is absent."""
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 — metadata lookups must never break a check
        return None


def distribution_versions(names: Iterable[str] = DEFAULT_DISTRIBUTIONS) -> dict[str, Optional[str]]:
    """``{distribution: version-or-None}`` for the given distributions."""
    return {name: distribution_version(name) for name in names}

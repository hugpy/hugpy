"""The public meta wheel must mirror the complete local Hugpy module stack."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[4]


def test_base_install_requires_every_lockstep_distribution():
    with (ROOT / "py" / "partition.toml").open("rb") as manifest:
        published = {
            entry["distribution"]
            for entry in tomllib.load(manifest)["package"]
            # The meta distribution cannot declare itself as a dependency.
            if entry["distribution"] != "hugpy"
        }
    with (ROOT / "py" / "meta" / "hugpy" / "pyproject.toml").open("rb") as project:
        dependencies = set(tomllib.load(project)["project"]["dependencies"])

    assert published <= dependencies

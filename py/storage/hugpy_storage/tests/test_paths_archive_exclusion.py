"""Archive-exclusion regression for resolve_model_dir() / route_destination().

Confirmed live bug: candidate_model_dirs() legitimately SURFACES a model's
entry under <root>/models/_archive/dedupes/... (reconcile's archive/de-dupe
area — reconcile IS the reconcile survey set and needs to see these), but the
weight file there is often a SYMLINK back into /checkpoints. resolve_model_dir()
must NEVER hand that path out as a live serve/transfer source, even when it is
listed first (e.g. because the registry's ``folder`` field still points at it)
and exists on disk ahead of the real flat dir.
"""
import os

import pytest

from hugpy_storage import model_paths as paths


def wfile(path, size=1024 * 1024 + 1):
    # model_looks_downloaded's generic (no config.json) branch requires each
    # *.safetensors file to exceed 1 MiB to distinguish a real weight from a
    # Git-LFS pointer stub; default to just over that floor so "complete"
    # checks in this test genuinely exercise the completeness gate.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * size)


@pytest.fixture()
def root(tmp_path):
    return str(tmp_path)


def test_archive_candidate_listed_first_but_real_dir_wins(root):
    # Mirrors the confirmed live bug — a registry entry whose ``folder`` points
    # into models/_archive/dedupes/..., AND the real flat dir also exists.
    archive_dir = os.path.join(root, "models", "_archive", "dedupes", "misc", "comfy", "foo")
    real_dir = os.path.join(root, "models", "misc", "comfy", "foo")
    wfile(os.path.join(archive_dir, "model.safetensors"))
    wfile(os.path.join(real_dir, "model.safetensors"))
    model = {
        "hub_id": "comfy/foo", "framework": "misc", "primary_task": "misc",
        "tasks": ["misc"], "folder": "_archive/dedupes/misc/comfy/foo",
    }
    cands = paths.candidate_model_dirs(model, root)
    assert archive_dir in cands, "survey set must still surface the archive dir"
    assert cands.index(archive_dir) < cands.index(real_dir)

    assert paths.resolve_model_dir(model, root, require_complete=False) == real_dir
    assert paths.resolve_model_dir(model, root, require_complete=True) == real_dir
    assert paths.route_destination(model, root) == real_dir


def test_archive_only_falls_through_to_flat_destination(root):
    # ONLY the archive copy exists on disk (no live dir anywhere). Even though
    # it is the sole EXISTING candidate, resolve_model_dir must not return it.
    only_archive_model = {
        "hub_id": "comfy/bar", "framework": "misc", "primary_task": "misc",
        "tasks": ["misc"], "folder": "_archive/dedupes/misc/comfy/bar",
    }
    only_archive_dir = os.path.join(root, "models", "_archive", "dedupes", "misc", "comfy", "bar")
    wfile(os.path.join(only_archive_dir, "model.safetensors"))

    resolved = paths.resolve_model_dir(only_archive_model, root, require_complete=False)
    assert resolved != only_archive_dir
    assert resolved == paths.flat_destination(only_archive_model, root)
    assert paths.resolve_model_dir(only_archive_model, root, require_complete=True) is None


def test_plain_model_unaffected(root):
    plain_dir = os.path.join(root, "models", "misc", "comfy", "baz")
    wfile(os.path.join(plain_dir, "model.safetensors"))
    plain_model = {"hub_id": "comfy/baz", "framework": "misc",
                   "primary_task": "misc", "tasks": ["misc"]}
    assert paths.resolve_model_dir(plain_model, root, require_complete=False) == plain_dir
    assert paths.route_destination(plain_model, root) == plain_dir


def test_is_archived_path_is_component_match():
    assert paths._is_archived_path("/root/models/_archive/dedupes/misc/comfy/foo")
    assert not paths._is_archived_path("/root/models/misc/comfy/my_archive_tool")
    assert not paths._is_archived_path("/root/models/misc/comfy/foo")
    assert not paths._is_archived_path("")

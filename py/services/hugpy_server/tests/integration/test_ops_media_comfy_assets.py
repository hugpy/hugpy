"""Cross-package consistency: the provisioner's §5b comfy asset manifest
(hugpy_ops) names exactly the files hugpy_media's comfy builder wires into
the IPAdapter graph. Ops may not import media, so this lives with the server
integration tests (moved out of the monolith's test_provisioner.py by the
ops agent, 2026-09-22)."""
from __future__ import annotations


def test_comfy_shared_asset_dest_names_match_the_media_builder():
    from hugpy_media.comfy.comfy_runner import _IPADAPTER_FILES
    from hugpy_ops.provisioner import COMFY_SHARED_ASSETS
    built = {COMFY_SHARED_ASSETS["ipadapter:sd15"].dest_name,
             COMFY_SHARED_ASSETS["ipadapter:sdxl"].dest_name,
             COMFY_SHARED_ASSETS["clip_vision:vit-h"].dest_name}
    runner_files = {f for pair in _IPADAPTER_FILES.values() for f in pair}
    assert runner_files <= built

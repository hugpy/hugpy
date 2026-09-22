import os
import shutil
from datetime import datetime, timezone

from hugpy_platform.utils import slugify, unique_path

from hugpy_media.schemas.whisper_schemas import MediaArtifactManifest, MediaWorkspace

def create_media_workspace(
    source_path: str,
    output_root: str | None = None,
    copy_source: bool = False,
    overwrite: bool = False,
) -> MediaWorkspace:
    source = os.path.abspath(os.path.expanduser(source_path))

    if not os.path.isfile(source):
        raise FileNotFoundError(f"Source media file not found: {source}")

    source_parent = os.path.dirname(source)
    parent = (
        os.path.abspath(os.path.expanduser(output_root))
        if output_root
        else source_parent
    )

    source_stem = os.path.splitext(os.path.basename(source))[0]
    workspace_name = f"{slugify(source_stem)}.assets"
    workspace_dir = os.path.join(parent, workspace_name)

    if os.path.exists(workspace_dir) and not overwrite:
        workspace_dir = unique_path(workspace_dir)

    os.makedirs(workspace_dir, exist_ok=True)

    manifest = MediaArtifactManifest(
        source_path=source,
        workspace_dir=workspace_dir,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    if copy_source:
        copied_source = os.path.join(workspace_dir, os.path.basename(source))
        shutil.copy2(source, copied_source)
        manifest.set_file("source_copy", copied_source)

    manifest.set_file("source", source)

    workspace = MediaWorkspace(
        source_path=source,
        root_dir=workspace_dir,
        manifest=manifest,
    )

    return workspace

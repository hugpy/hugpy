// Upload → ingest → guard-kind helpers for the Identities station (lifted out of
// IdentitiesStation.tsx in k94 so the one-path create panel can share them
// without a circular import). Behavior/wording UNCHANGED from the originals:
// the EXACT upload→ingest→guard-kind flow StudioGenerateTab's onPickImageFile /
// StudioMovieComposer's run (POST /uploads → POST /video/ingest → MediaRef),
// joining the session library on success so an asset picked here is pickable
// from the library everywhere else too.
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import { hugpyConfig } from "../../config";
import { addToLibrary } from "../../video/mediaLibrary";
import { mediaRefSchema, uploadResultSchema } from "../../video/contract";
import type { MediaRef } from "../../video/contract";
import { getSessionId } from "../../session";

async function uploadAndIngest(
  file: File,
  kind: "image" | "video",
  ops: { upload: string; ingest: string },
): Promise<{ ref?: MediaRef; error?: string }> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("sid", getSessionId());
  const up = await request<unknown>(hugpyConfig.uploadUrl, {
    method: "POST",
    body: fd,
    meta: { specKey: "studio", operation: ops.upload },
  });
  if (!up.ok) return { error: describeAppError(errorOf(up)) };
  const upParsed = uploadResultSchema.safeParse(okValue(up));
  if (!upParsed.success) return { error: "Malformed upload response." };
  const ing = await request<unknown>(hugpyConfig.videoIngestUrl, {
    method: "POST",
    body: JSON.stringify({ path: upParsed.data.path }),
    headers: { "Content-Type": "application/json" },
    meta: { specKey: "studio", operation: ops.ingest },
  });
  if (!ing.ok) return { error: describeAppError(errorOf(ing)) };
  const media = mediaRefSchema.safeParse(okValue(ing));
  if (!media.success) return { error: "Malformed ingest response." };
  if (media.data.kind !== kind) {
    return {
      error:
        kind === "image"
          ? `Ingested asset is "${media.data.kind}", not an image — pick a still.`
          : `Ingested asset is "${media.data.kind}", not a video — pick a video file.`,
    };
  }
  addToLibrary(media.data, "upload", media.data.mime);
  return { ref: media.data };
}

/** Upload → ingest → guard kind=image. */
export function uploadAndIngestImage(file: File): Promise<{ ref?: MediaRef; error?: string }> {
  return uploadAndIngest(file, "image", {
    upload: "identity.station.upload",
    ingest: "identity.station.ingest",
  });
}

/** char360 S4 — the same chain guarding the INVERSE kind: the source MUST be a video. */
export function uploadAndIngestVideo(file: File): Promise<{ ref?: MediaRef; error?: string }> {
  return uploadAndIngest(file, "video", {
    upload: "identity.video_extract.upload",
    ingest: "identity.video_extract.ingest",
  });
}

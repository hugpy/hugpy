// Frames & Models station (Phase 4) — the sample image's left rail + frame grid.
// Flow:
//   video file → POST /uploads (multipart) → {path}
//              → POST /video/ingest {path} → MediaRef (guard kind==="video")
//              → FrameExtractCore: knob rail + frame_extract job + frame grid + pick
// Every hop goes through request<T>() + config URLs; errors surface via
// describeAppError.
//
// Studio slice 2a: the Options knob rail + useFrameJobs + frame grid + frame-pick
// were extracted VERBATIM into stations/ops/FrameExtractCore. Because the Frames
// station has an Options panel (whose knob state is shared with the working view),
// the Core owns the whole SectionTabs and takes this station's add-file bar node +
// `busy` flag as props. This station now owns only the upload/ingest lifecycle.
// Behaviour is unchanged.
import { useState } from "react";
import type { StationSpec } from "./types";
import { AddFileBar } from "./AddFileBar";
import { FrameExtractCore } from "./ops/FrameExtractCore";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";
import { getSessionId } from "../session";
import {
  mediaRefSchema,
  uploadResultSchema,
  type MediaRef,
} from "../video/contract";

type Phase = "idle" | "uploading" | "ingesting" | "ready";

const PHASE_LABEL: Record<Phase, string> = {
  idle: "",
  uploading: "Uploading…",
  ingesting: "Ingesting…",
  ready: "",
};

export function FrameExtractStation({ spec }: { spec: StationSpec }) {
  const [source, setSource] = useState<MediaRef | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);

  const busy = phase === "uploading" || phase === "ingesting";

  async function onPick(file: File) {
    setError(null);
    setFileName(file.name);
    setSource(null);
    setPhase("uploading");

    // (a) multipart upload — field `file` + field `sid`. No Content-Type header:
    // the browser sets the multipart boundary itself.
    const fd = new FormData();
    fd.append("file", file);
    fd.append("sid", getSessionId());
    const up = await request<unknown>(hugpyConfig.uploadUrl, {
      method: "POST",
      body: fd,
      meta: { specKey: "frames", operation: "upload" },
    });
    if (!up.ok) {
      setError(describeAppError(errorOf(up)));
      setPhase("idle");
      return;
    }
    const upParsed = uploadResultSchema.safeParse(okValue(up));
    if (!upParsed.success) {
      setError("Malformed upload response.");
      setPhase("idle");
      return;
    }

    // (b) ingest the server path into a typed MediaRef.
    setPhase("ingesting");
    const ing = await request<unknown>(hugpyConfig.videoIngestUrl, {
      method: "POST",
      body: JSON.stringify({ path: upParsed.data.path }),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "frames", operation: "ingest" },
    });
    if (!ing.ok) {
      setError(describeAppError(errorOf(ing)));
      setPhase("idle");
      return;
    }
    const media = mediaRefSchema.safeParse(okValue(ing));
    if (!media.success) {
      setError("Malformed ingest response.");
      setPhase("idle");
      return;
    }
    if (media.data.kind !== "video") {
      setError(
        `Ingested asset is "${media.data.kind}", not a video — use the Image Crop or Audio station for that.`,
      );
      setPhase("idle");
      return;
    }
    setSource(media.data);
    setPhase("ready");
  }

  // Clear the loaded source back to the empty state. FrameExtractCore resets the job
  // + selection when the source reference changes (here → null).
  function clearSource() {
    setSource(null);
    setFileName(null);
    setError(null);
  }

  return (
    <section className="station-card station-wide">
      <FrameExtractCore
        source={source}
        busy={busy}
        addFile={
          <>
            <AddFileBar
              accept="video/*"
              onFile={onPick}
              busy={busy}
              buttonLabel={source ? "Choose another video" : "Choose video"}
              onClear={source && !busy ? clearSource : undefined}
            >
              {busy && <span className="vi-crop-status">{PHASE_LABEL[phase]}</span>}
              {!busy && source && (
                <>
                  {fileName && (
                    <span className="vi-addbar-name" title={fileName}>
                      {fileName}
                    </span>
                  )}
                  <span className="vi-crop-meta">
                    <code>{source.mime}</code>
                    {source.width && source.height ? ` · ${source.width}×${source.height}` : ""}
                    {source.duration_s != null ? ` · ${source.duration_s}s` : ""}
                    {source.fps_native != null ? ` · ${source.fps_native} fps native` : ""}
                  </span>
                </>
              )}
            </AddFileBar>
            {error && <p className="vi-error" role="alert">{error}</p>}
          </>
        }
      />
    </section>
  );
}

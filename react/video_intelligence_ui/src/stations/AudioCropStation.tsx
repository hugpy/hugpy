// Audio Crop station (Phase 5) — the temporal sibling of the Image Crop station.
// Flow:
//   file (audio OR video) → POST /uploads (multipart) → {path}
//        → POST /video/ingest {path} → MediaRef
//        → AudioCropCore: (video→audio extract) + TemporalRegionEditor + enqueue
//          + result grid
// Every hop goes through request<T>() + config URLs; errors surface via
// describeAppError.
//
// Studio slice 2a: the extract step + region editor + enqueue hook + result grid
// were extracted VERBATIM into stations/ops/AudioCropCore (the reusable "Components
// view minus upload"), including the video→audio extract path. This station now owns
// only the upload/ingest lifecycle + the persistent add-file bar; it renders
// <AudioCropCore/> in the Components slot. Behaviour is unchanged.
import { useState } from "react";
import type { StationSpec } from "./types";
import { SectionTabs } from "./SectionTabs";
import { AddFileBar } from "./AddFileBar";
import { AudioCropCore } from "./ops/AudioCropCore";
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

export function AudioCropStation({ spec }: { spec: StationSpec }) {
  const [source, setSource] = useState<MediaRef | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);

  const busy = phase === "uploading" || phase === "ingesting";

  // Clear the loaded source back to the empty state. AudioCropCore resets its queued
  // clips + extract + tracker rows when the source reference changes (here → null).
  function clearSource() {
    setSource(null);
    setFileName(null);
    setError(null);
  }

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
      meta: { specKey: "audio-crop", operation: "upload" },
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
      meta: { specKey: "audio-crop", operation: "ingest" },
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
    if (media.data.kind !== "audio" && media.data.kind !== "video") {
      setError(
        `Ingested asset is "${media.data.kind}", not audio or video — use the Image Crop station for that.`,
      );
      setPhase("idle");
      return;
    }
    setSource(media.data);
    setPhase("ready");
    // A video's audio track is pulled out by AudioCropCore once the source lands.
  }

  const isVideo = source?.kind === "video";

  return (
    <section className="station-card station-wide">
      {/* Same shared section framing as Frames/Generate, but with NO Options
          panel — the crop params live inline in the Components view below the
          waveform, so no [Components|Options] toggle / rail renders. */}
      <SectionTabs
        ariaLabel="Audio crop station"
        addFile={
          <>
            {/* ONE compact add control — a Choose button + filename, no dropbox.
                Accepts audio OR video; onPick routes a video through the
                audio_extract chain before cropping. */}
            <AddFileBar
              accept="audio/*,video/*"
              onFile={onPick}
              busy={busy}
              buttonLabel={source ? "Choose another file" : "Choose audio or video"}
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
                    {source.duration_s != null ? ` · ${source.duration_s}s` : ""}
                    {isVideo ? " · video (extracting audio)" : ""}
                  </span>
                </>
              )}
            </AddFileBar>
            {error && <p className="vi-error" role="alert">{error}</p>}
          </>
        }
        components={<AudioCropCore source={source} multiRegion />}
      />
    </section>
  );
}

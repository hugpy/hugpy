// StudioItemCard — the first VISIBLE piece of the Studio redesign (slice 2b). It
// renders a first-class StudioItem inline with its own per-item OPS, so a staged
// studio item is no longer a loose `source` ref but a card that shows explicit,
// kind-gated operations (Image Crop / Video Crop / Audio Crop / Frames & Models)
// which either MUTATE the item in place or BRANCH a new child item.
//
// The card reuses the slice-2a op "cores" verbatim (ImageCropCore / VideoCropCore /
// AudioCropCore / FrameExtractCore): selecting an op opens the matching Core in an
// inline drawer, wired via each Core's additive `onProduced` sink. The card contains
// NO composer (prompt / negative / parts) — that is slice 3's per-item composer,
// deliberately kept out here to bound the recursion. Ops only.
//
// The DAG rule (routing): an op whose PRODUCED media kind equals the item's working
// kind (image item + Image Crop; video item + Video Crop; audio item + Audio Crop) is
// SAME-KIND → onApplyOp replaces the parent's `working`. An op that CHANGES the kind
// or FANS OUT (video item + Audio Crop → audio; video item + Frames → images) →
// onSpawn mints NEW child items and never overwrites the parent's `working` (that
// would corrupt its kind).
import { useRef, useState, type ReactNode } from "react";
import { mediaBytesUrl } from "../../config";
import type { MediaRef } from "../../video/contract";
import type { SpatialRegion, TemporalRegion } from "../../regions/types";
import { getJobs, type StationId } from "../../video/jobTracker";
import { updateLibraryProvenance } from "../../video/mediaLibrary";
import type { StudioItem, StudioItemOp } from "./studioItem";
import { ImageCropCore } from "../ops/ImageCropCore";
import { VideoCropCore } from "../ops/VideoCropCore";
import { AudioCropCore } from "../ops/AudioCropCore";
import { FrameExtractCore } from "../ops/FrameExtractCore";

type OpKind = StudioItemOp["kind"];

/** A produced ref + the axes the op ran over — the shape every Core's onProduced
 *  hands us (the frame sink omits `meta`, which is why it is optional). */
type ProducedMeta = { spatial?: SpatialRegion | null; temporal?: TemporalRegion | null };
type ProducedSink = (op: OpKind, ref: MediaRef, meta?: ProducedMeta) => void;

// Fresh op id — mirrors the arm's monotonic-counter + base-36-timestamp idiom
// (studioItem.ts itemKey / useCropJobs.makeKey), unique within a session with no
// new dependency.
let _opSeq = 0;
function opKey(): string {
  _opSeq += 1;
  return `op_${_opSeq.toString(36)}_${Date.now().toString(36)}`;
}

// The media kind an op PRODUCES — the DAG routing pivot. When it equals the item's
// working kind the op mutates in place (same-kind); otherwise it branches a child.
function producedKind(op: OpKind): string {
  switch (op) {
    case "image_crop":
      return "image";
    case "video_crop":
      return "video";
    case "audio_crop":
      return "audio";
    case "frame_extract":
      return "image";
  }
}

// Which tracker station each op enqueues on — used to SEED the re-emit guard below.
const STATION_FOR_OP: Record<OpKind, StationId> = {
  image_crop: "image-crop",
  video_crop: "video-crop",
  audio_crop: "audio-crop",
  frame_extract: "frames",
};

const OP_LABEL: Record<OpKind, string> = {
  image_crop: "Image Crop",
  video_crop: "Video Crop",
  audio_crop: "Audio Crop",
  frame_extract: "Frames & Models",
};

// The ops offered for a given working kind (the menu gate).
//   image → Image Crop · video → Video Crop / Audio Crop / Frames · audio → Audio Crop
function opsFor(kind: string): OpKind[] {
  if (kind === "image") return ["image_crop"];
  if (kind === "video") return ["video_crop", "audio_crop", "frame_extract"];
  if (kind === "audio") return ["audio_crop"];
  return [];
}

// Last path segment of a uri, for the compact (immutable) provenance line.
function shortUri(uri: string): string {
  const clean = uri.split(/[?#]/)[0];
  const seg = clean.split("/").filter(Boolean).pop() ?? uri;
  return seg.length > 40 ? `…${seg.slice(-39)}` : seg;
}

// Poster/preview for the item's working ref, per kind — the arm's inline idiom (a
// muted seeked <video>, an <img>, or an <audio>), not the non-exported StudioGenerateTab
// SourceThumb (kept private to that surface).
function renderMedia(ref: MediaRef): ReactNode {
  const src = mediaBytesUrl(ref.uri);
  if (ref.kind === "image") {
    return <img className="vi-studio-item-img" src={src} alt="Studio item" />;
  }
  if (ref.kind === "video") {
    return (
      <video
        className="vi-studio-item-img"
        src={`${src}#t=0.1`}
        muted
        playsInline
        controls
        preload="metadata"
        aria-label="Studio item preview"
      />
    );
  }
  if (ref.kind === "audio") {
    return (
      <audio
        className="vi-studio-item-audio"
        src={src}
        controls
        preload="metadata"
        aria-label="Studio item preview"
      />
    );
  }
  return <span className="vi-crop-hint">No preview for {ref.kind}.</span>;
}

// The immutable "where it came from" line — read-only, never editable.
function ProvenanceLine({ item }: { item: StudioItem }) {
  const { provenance } = item;
  return (
    <div className="vi-studio-item-prov" aria-label="Provenance (immutable)">
      <span className="vi-studio-item-prov-origin">{provenance.origin}</span>
      <code className="vi-studio-item-prov-kind">{item.working.kind}</code>
      {provenance.parentUri && (
        <span className="vi-studio-item-prov-parent" title={provenance.parentUri}>
          ← {shortUri(provenance.parentUri)}
        </span>
      )}
      {provenance.model && (
        <span
          className="vi-studio-item-prov-model"
          title={`model: ${provenance.model}`}
        >
          · {provenance.model}
        </span>
      )}
    </div>
  );
}

// Mount the matching Core in the drawer. Each arrow's `meta` is contextually typed
// from the Core's own onProduced prop, so the axes flow through with their exact
// shape; the crop cores run SINGLE-region (multiRegion={false}) inline-on-item.
function renderCore(op: OpKind, working: MediaRef, onProduced: ProducedSink): ReactNode {
  switch (op) {
    case "image_crop":
      return (
        <ImageCropCore
          source={working}
          multiRegion={false}
          onProduced={(ref, meta) => onProduced(op, ref, meta)}
        />
      );
    case "video_crop":
      return (
        <VideoCropCore
          source={working}
          onProduced={(ref, meta) => onProduced(op, ref, meta)}
        />
      );
    case "audio_crop":
      return (
        <AudioCropCore
          source={working}
          multiRegion={false}
          onProduced={(ref, meta) => onProduced(op, ref, meta)}
        />
      );
    case "frame_extract":
      return (
        <FrameExtractCore
          source={working}
          busy={false}
          addFile={null}
          onProduced={(ref) => onProduced(op, ref)}
        />
      );
  }
}

export interface StudioItemCardProps {
  item: StudioItem;
  /** same-kind op → parent working replaced (mutate in place). */
  onApplyOp: (item: StudioItem, op: StudioItemOp) => void;
  /** kind-change / fan-out → new child item(s) carrying parent provenance. */
  onSpawn: (parent: StudioItem, produced: MediaRef[], kind: OpKind) => void;
  onRevert: (item: StudioItem) => void;
  onRemove: (item: StudioItem) => void;
}

export function StudioItemCard({
  item,
  onApplyOp,
  onSpawn,
  onRevert,
  onRemove,
}: StudioItemCardProps) {
  const [activeOp, setActiveOp] = useState<OpKind | null>(null);

  // Per-card guard against the shared per-station tracker RE-EMITTING pre-existing
  // done jobs. The crop cores fire onProduced from a useEffect over the tracker's
  // rows for their station, so a freshly-mounted core (every drawer open) would
  // re-notify EVERY historical done crop of that station — spawning phantom children
  // or re-applying stale results. We remember each uri this card has already routed,
  // and SEED that set at drawer-open with the station's current outputs, so only NEW
  // completions route. (frame_extract's sink is pick-triggered, not job-triggered, so
  // it is inherently immune; seeding it is harmless.)
  const routedUris = useRef<Set<string>>(new Set());

  const working = item.working;
  const ops = opsFor(working.kind);
  // Revert is meaningful only once an op has diverged `working` from the source.
  const canRevert = working !== item.source;

  function openOp(op: OpKind) {
    const station = STATION_FOR_OP[op];
    const seen = routedUris.current;
    for (const rec of getJobs()) {
      if (rec.station === station) {
        for (const out of rec.outputs) seen.add(out.uri);
      }
    }
    setActiveOp(op);
  }

  // The single onProduced sink the mounted core drives. Routes ONCE per uri.
  function handleProduced(op: OpKind, ref: MediaRef, meta?: ProducedMeta) {
    if (!ref || routedUris.current.has(ref.uri)) return;
    routedUris.current.add(ref.uri);

    // Light provenance enrichment: tie the produced library entry back to its parent
    // item (groupId) and carry any inherited prompt/model. LibraryProvenance has no
    // free "op kind" field (genKind is a closed generation union), so the op kind
    // itself is recorded on the StudioItemOp / the spawned child's provenance instead.
    updateLibraryProvenance(ref.uri, {
      groupId: item.key,
      ...(item.provenance.prompt != null ? { prompt: item.provenance.prompt } : {}),
      ...(item.provenance.model != null ? { model: item.provenance.model } : {}),
    });

    const sameKind = op !== "frame_extract" && producedKind(op) === working.kind;
    if (sameKind) {
      onApplyOp(item, {
        id: opKey(),
        kind: op,
        at: Date.now(),
        spatial: meta?.spatial ?? null,
        temporal: meta?.temporal ?? null,
        result: ref,
      });
    } else {
      onSpawn(item, [ref], op);
    }
    // The op resolved — close the drawer.
    setActiveOp(null);
  }

  return (
    <div className="vi-studio-item">
      <div className="vi-studio-item-body">
        <div className="vi-studio-item-media">{renderMedia(working)}</div>
        <div className="vi-studio-item-meta">
          <ProvenanceLine item={item} />
          <div
            className="vi-studio-item-ops"
            role="group"
            aria-label="Item operations"
          >
            {ops.map((op) => (
              <button
                key={op}
                type="button"
                className={
                  activeOp === op
                    ? "vi-btn vi-btn-sm vi-btn-accent"
                    : "vi-btn vi-btn-sm"
                }
                aria-pressed={activeOp === op}
                onClick={() => (activeOp === op ? setActiveOp(null) : openOp(op))}
              >
                {OP_LABEL[op]}
              </button>
            ))}
            {canRevert && (
              <button
                type="button"
                className="vi-btn vi-btn-sm vi-btn-ghost"
                onClick={() => onRevert(item)}
                title="Restore the working ref back to the immutable source"
              >
                Revert to source
              </button>
            )}
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              onClick={() => onRemove(item)}
              title="Remove this item from the studio (forgets the local card only)"
            >
              Remove
            </button>
          </div>
        </div>
      </div>

      {activeOp && (
        <div className="vi-studio-item-drawer">
          <div className="vi-studio-item-drawer-head">
            <span className="vi-studio-item-drawer-title">{OP_LABEL[activeOp]}</span>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              onClick={() => setActiveOp(null)}
            >
              Close
            </button>
          </div>
          {renderCore(activeOp, working, handleProduced)}
        </div>
      )}
    </div>
  );
}

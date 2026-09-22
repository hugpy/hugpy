// The single STUDIO WORKBENCH (Round 9) — the movie-page shape with ONE shared left
// sidebar. The studio no longer carries its own [ Library | Active | Settings ] sidebar
// (that produced the dual-sidebar redundancy): it now folds into the arm shell's ONE
// WorkbenchSidebar via the sidebar registry —
//
//   • Settings  — the geometry / sampler / model knobs + the template dropdown render
//     into the shell sidebar's guarded "Settings" tab (the studio surface portals its
//     knob grid into `reg.settingsHost`; the knob STATE stays in the surface).
//   • Active    — studio's in-flight renders portal into the shell "Active Processes"
//     tab (studio jobs are not in the jobTracker, so they ride in as an extra section).
//   • Library   — the shell "Session Library" IS the one library; produced studio clips
//     already flow into it (genKind studio_i2v) with extend ("Send to Studio") / restyle
//     ("Restyle") on every clip row. (The former inline StudioLibrarySection is retained
//     in the tree, unused, for history — per-row Details / capability+model filters stay
//     deferred there. select-to-viewer is NO LONGER deferred: it exists in that dead
//     file's code but was never reachable from either mount, which is exactly why the
//     viewer read as stuck — see StudioLibraryTab.tsx's header note. The reachable
//     version now lives on the viewer itself, below.)
//
// So the CENTER column is just the movie skeleton: the VIEWER on top (plays the selected
// clip, with its own header controls — Play from library / Clear / follow-newest, operator
// ask 2026-07-12 — so it never reads as a fixed, unchangeable default) → the GENERATOR
// below it (composer prompt + negative, with the shape heroes / reference slots / attach
// affordances as the input block, hero gating intact) → the Run row. Template selection
// lives in the Settings dropdown.
//
// This ONE component mounts in BOTH places the studio appears — the standalone Studio station
// (StudioClipsStation) and the Generate station's "studio" mode (StudioGenerateMode) — so the
// workbench is identical and a job enqueued from either is byte-identical. The only per-mount
// difference is `registerBridge` (the standalone station owns the "Send to Studio" target).
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
// Sub-tab input memory for the staged source + item strip — see
// video/sessionForm.ts.
import { useSessionState } from "../../video/sessionForm";
import { createPortal } from "react-dom";
import { registerStudioStager, type StudioMode } from "../../video/studioBridge";
import { StudioAssistLogPanel } from "./StudioAssistLogPanel";
import type { MediaRef } from "../../video/contract";
import { useStudioPresets } from "../../video/useStudioPresets";
import { useStudioClips } from "./studioShared";
import { useMediaLibrary } from "../../video/mediaLibrary";
import {
  makeStudioItem,
  applyOp,
  revertItem,
  type StudioItem,
  type StudioItemOp,
} from "./studioItem";
import { StudioItemCard } from "./StudioItemCard";
import { StudioGenerateSurface } from "./StudioGenerateTab";
import { StudioViewer } from "./StudioViewer";
import { useSidebarRegistry } from "../SidebarRegistry";

export interface StudioPlaneProps {
  /** Register the studioBridge stager (the "Send to Studio" target). The standalone
   *  Studio station passes true; the Generate-station studio mode passes false — the
   *  shell library's "Send to Studio" navigates to /studio-clips (the standalone station),
   *  so registering in the Generate mode would only race that navigation. */
  registerBridge?: boolean;
  /** When set by a top-level Generate tab (Clip / Cinema), lock the generate surface to
   *  that sub-mode and hide its internal Clip|Cinema switcher. Omitted by the standalone
   *  /studio-clips station, which keeps the internal switcher. */
  lockedSurfaceMode?: "clip" | "movie";
}

export function StudioPlane({ registerBridge = false, lockedSurfaceMode }: StudioPlaneProps) {
  const reg = useSidebarRegistry();
  const { presets, loading: presetsLoading } = useStudioPresets(true);
  const clipsState = useStudioClips();

  // Source clip + its staged consume mode; `stageSeq` bumps on each external stage so the
  // surface adopts the capability exactly once per stage.
  //
  // The staged SOURCE and its mode are session-scoped (operator ask 2026-08-06:
  // inputs survive a sub-tab switch) — they are operator-staged inputs to the
  // generate surface below, and restoring the surface's capability/knobs while
  // silently dropping the clip they were chosen for would be a half-restore.
  // `stageSeq` deliberately is NOT persisted: it counts STAGE EVENTS, and the
  // surface's capability-resolution effect keys off it. Restoring a non-zero
  // value would re-fire that handoff on every mount and overwrite exactly the
  // capability we just restored.
  const [source, setSource] = useSessionState<MediaRef | null>("studio.plane.source", null);
  const [sourceMode, setSourceMode] = useSessionState<StudioMode>(
    "studio.plane.sourceMode",
    "extend",
  );
  const [stageSeq, setStageSeq] = useState(0);

  // First-class studio items (Studio redesign, slice 1). Each external stage mints a
  // provenance-carrying StudioItem here. Slice 2b RENDERS these as a strip of
  // StudioItemCards with inline per-item ops (crop / frames), but they still do NOT
  // feed the generate surface — the legacy `source`/`setSource` wiring below is passed
  // to StudioGenerateSurface exactly as before, so today's single-source behavior
  // (extend vs restyle, cross-tier carry) is byte-identical. Slice 3 folds the surface
  // into the item (a per-item composer) and makes generation item-driven.
  const [items, setItems] = useSessionState<StudioItem[]>("studio.plane.items", []);

  // LIFTED clip selection so the player-forward viewer (top of the centre column) follows
  // the newest playable clip as a fresh enqueue completes — see `pinned` immediately
  // below for the manual override (operator ask 2026-07-12: "can the change be made
  // THROUGH to change the initial scene or video or make it 'new'" — the viewer used to
  // have no controls of its own and would read as permanently stuck on one clip).
  const [selected, setSelected] = useState<string | null>(null);

  // PINNED = the operator has taken manual control of the viewer (a library pick via
  // "▤ Play from library", or an explicit "✕ Clear viewer") — the auto-follow effect
  // below stands down while this is true, so a manual choice sticks instead of being
  // overwritten by the next 6s clip-list poll. "↻ follow newest" flips it back off and
  // hands control back to auto-follow. This is the one semantic switch StudioViewer's
  // header reads/toggles; see its matching half of this note.
  const [pinned, setPinned] = useState(false);

  const topRef = useRef<HTMLDivElement | null>(null);

  // Round 9: fold into the ONE shell sidebar — reveal its guarded Settings tab (knobs +
  // template dropdown portal there). registerSettings is stable, so this runs once per
  // mount and unregisters on unmount.
  //
  // Studio's in-flight renders are NO LONGER portaled into the shared Active Processes
  // tab: those same jobs now arrive fleet-wide via GET /llm/jobs (kinds studio_i2v /
  // generate_studio_movie / generate_scene / generate_movie, transport "media"), which
  // the panel polls and renders directly — so keeping the portal would double-count each
  // studio render. The in-plane "Active renders" component (StudioActiveProcesses) is
  // retained in the tree for history; it just no longer feeds the global sidebar.
  useEffect(() => reg.registerSettings(), [reg.registerSettings]);

  // GENERATION LOG → the sidebar's ACTIVE tab (operator, 2026-08-05): the
  // studio-assist log rides the existing activeExtra portal (the always-rendered
  // host at the foot of the Active Processes panel), so generation attempts sit
  // with the in-flight renders — one "what is the studio doing" place. Register
  // so the panel suppresses its empty copy while the log is mounted.
  useEffect(() => reg.registerActiveExtra(), [reg.registerActiveExtra]);

  // Auto-follow the newest playable clip WHILE UNPINNED. Previously this only re-picked
  // when the selected job disappeared from the clip list entirely — since a completed
  // job id never leaves the durable catalog, that meant it latched onto the FIRST
  // playable clip of the session and then never moved again, no matter how many newer
  // clips completed after it. That is the reported bug ("sits on the same clip
  // indefinitely and reads as a stuck, unchangeable default"): the header comment always
  // CLAIMED auto-follow, but the code only implemented sticky-keep-unless-deleted. Fixed:
  // now it re-syncs to the newest playable clip on every clip-list change, same as the
  // comment always claimed — gated on `pinned` so a manual pick/clear (see above) is
  // respected instead of being clobbered by the next poll tick.
  // Session-library media from OTHER tabs (cinema movies, scene clips/frames, crops,
  // …) — everything that finished anywhere and carries pixels, deduped against the
  // studio clip catalog by uri (the studio-clips poll mirrors every playable clip
  // into the library, so without the dedup each clip would list twice). Operator
  // ask 2026-08-27: generated media from ANY tab auto-displays in this viewer, and
  // "Play from library" offers recent generations — not just studio_i2v clips.
  const libraryAll = useMediaLibrary();
  const libraryMedia = useMemo(() => {
    const clipUris = new Set(
      clipsState.clips.map((c) => c.output?.uri).filter((u): u is string => !!u),
    );
    return libraryAll
      .filter(
        (it) =>
          (it.ref.kind === "video" || it.ref.kind === "image") &&
          !!it.ref.uri &&
          !clipUris.has(it.ref.uri),
      )
      .sort((a, b) => b.addedAt - a.addedAt);
  }, [libraryAll, clipsState.clips]);

  useEffect(() => {
    if (pinned) return;
    const clip = clipsState.clips.find((c) => c.playable) ?? null;
    const clipAt = clip ? (clip.updated ?? clip.created ?? 0) * 1000 : -1;
    // "history" items are server back-fill stamped with the MOUNT time, not the
    // generation time — following them would surface stale renders as "newest".
    const item = libraryMedia.find((it) => it.origin !== "history") ?? null;
    if (item && item.addedAt > clipAt) {
      setSelected(`lib:${item.ref.asset_id}`);
    } else {
      setSelected(clip?.job_id ?? null);
    }
  }, [clipsState.clips, libraryMedia, pinned]);

  // Manual pick from the viewer's "▤ Play from library" panel — pins so auto-follow
  // leaves the choice alone.
  const pickViewerClip = useCallback((jobId: string) => {
    setSelected(jobId);
    setPinned(true);
  }, []);

  // "✕ Clear viewer" — empties AND pins, so StudioViewer's explicit empty state
  // ("nothing playing") sticks instead of being silently repopulated by the next poll
  // (StudioViewer no longer has an "any playable clip" fallback that would undo this —
  // see that file).
  const clearViewer = useCallback(() => {
    setSelected(null);
    setPinned(true);
  }, []);

  // "↻ follow newest" toggle — flips `pinned`. Turning it back on (true → false) hands
  // control back to the auto-follow effect above, which fires on the very next render
  // and re-syncs to the newest playable clip immediately.
  const toggleFollowNewest = useCallback(() => {
    setPinned((v) => !v);
  }, []);

  const applyStage = useCallback((ref: MediaRef | null, mode: StudioMode) => {
    setSource(ref);
    setSourceMode(mode);
    setStageSeq((n) => n + 1);
    // Bring the generator into view so the staged source shows.
    topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  // ── Slice-2b item mutation (matched by the item's stable `key`, NOT its uri —
  // a ref may legitimately repeat) ────────────────────────────────────────────

  // SAME-KIND op → replace the matching item's `working` via applyOp (mutate in place).
  const onApplyOp = useCallback((target: StudioItem, op: StudioItemOp) => {
    setItems((prev) =>
      prev.map((it) => (it.key === target.key ? applyOp(it, op) : it)),
    );
  }, []);

  // KIND-CHANGE / FAN-OUT op → append a NEW child item per produced ref, carrying the
  // forward-only provenance edge (parentUri = the parent's working uri) + the op kind
  // as origin, and inheriting any prompt/model/genKind lineage from the parent.
  const onSpawn = useCallback(
    (parent: StudioItem, produced: MediaRef[], kind: StudioItemOp["kind"]) => {
      if (produced.length === 0) return;
      setItems((prev) => [
        ...prev,
        ...produced.map((ref) =>
          makeStudioItem(ref, {
            origin: `op:${kind}`,
            parentUri: parent.working.uri,
            ...(parent.provenance.prompt != null
              ? { prompt: parent.provenance.prompt }
              : {}),
            ...(parent.provenance.model != null
              ? { model: parent.provenance.model }
              : {}),
            ...(parent.provenance.genKind != null
              ? { genKind: parent.provenance.genKind }
              : {}),
          }),
        ),
      ]);
    },
    [],
  );

  // Revert the item's `working` back to its immutable source (ops history preserved).
  const onRevert = useCallback((target: StudioItem) => {
    setItems((prev) =>
      prev.map((it) => (it.key === target.key ? revertItem(it) : it)),
    );
  }, []);

  // Forget one item's card (local only — never deletes the underlying server asset).
  const onRemove = useCallback((target: StudioItem) => {
    setItems((prev) => prev.filter((it) => it.key !== target.key));
  }, []);

  // studioBridge "Send to Studio" / "Restyle": only the registering mount (the standalone
  // station) receives staged sources; the Generate-mode mount passes registerBridge=false.
  useEffect(() => {
    if (!registerBridge) return;
    return registerStudioStager((req) => {
      const ref = req.sourceVideo ?? null;
      applyStage(ref, req.mode ?? "extend");
      // Mint the first-class StudioItem for this handoff, carrying the forwarded
      // provenance (or a "send-to-studio" default). Additive — see the `items` note.
      if (ref) {
        const provenance = req.provenance ?? { origin: "send-to-studio", parentUri: ref.uri };
        setItems((prev) => [...prev, makeStudioItem(ref, provenance)]);
      }
    });
  }, [registerBridge, applyStage]);

  return (
    <div className="vi-studio-workbench" ref={topRef}>
      {/* CENTER, movie skeleton: viewer on top → generator below. Viewer header controls
          (Play from library / Clear / follow-newest toggle) are wired here so both
          plane mounts get them for free — see the `pinned` note above. */}
      <StudioViewer
        clips={clipsState.clips}
        library={libraryMedia}
        selected={selected}
        pinned={pinned}
        onPick={pickViewerClip}
        onClear={clearViewer}
        onToggleFollow={toggleFollowNewest}
      />

      {/* Slice 2b: the staged studio items as a strip of first-class cards with inline
          per-item ops. Additive — it sits ABOVE the generate surface and does not touch
          the surface or its props (slice 3 folds these together). Rendered in BOTH plane
          mounts identically (this is the one shared StudioPlane). Each card is wrapped in
          a keyed intrinsic <div> so the map key never lands on a component (which the
          bare-tsc env flags as a spurious prop). */}
      {items.length > 0 && (
        <div className="vi-studio-item-strip" aria-label="Studio items">
          {items.map((it) => (
            <div className="vi-studio-item-cell" key={it.key}>
              <StudioItemCard
                item={it}
                onApplyOp={onApplyOp}
                onSpawn={onSpawn}
                onRevert={onRevert}
                onRemove={onRemove}
              />
            </div>
          ))}
        </div>
      )}

      <StudioGenerateSurface
        presets={presets}
        presetsLoading={presetsLoading}
        clips={clipsState.clips}
        source={source}
        setSource={setSource}
        sourceMode={sourceMode}
        stageSeq={stageSeq}
        onEnqueued={clipsState.refresh}
        settingsHost={reg.settingsHost}
        lockedSurfaceMode={lockedSurfaceMode}
      />
      {/* STUDIO-ASSIST LIVE LOG → portaled into the sidebar's ACTIVE (Active
          Processes) tab via the registry's activeExtra host (operator correction
          2026-08-05: "active tab" = Active Processes). Single mount, shared stream. */}
      {reg.activeExtraHost ? createPortal(<StudioAssistLogPanel />, reg.activeExtraHost) : null}
      {/* Studio's in-flight renders are shown by the shared Active Processes panel via the
          fleet-wide GET /llm/jobs feed (no per-plane portal — that would double-count each
          render, since the same job appears in /llm/jobs with transport "media"). */}
    </div>
  );
}

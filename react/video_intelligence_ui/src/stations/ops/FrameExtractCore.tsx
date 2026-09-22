// FrameExtractCore — the reusable body of the Frames & Models station (Studio
// slice 2a). The station's working body MINUS the upload lifecycle: the Options
// knob rail (fps / quality / fmt / window / max_frames + the shared image-model
// dropdown) + the useFrameJobs enqueue hook + the paginated frame grid + the
// frame-pick → library flow, parameterised by a `source` ref.
//
// Unlike the crop cores (which fill a single `components` slot), the Frames station
// has an Options panel whose knob state is shared with the working view, so this
// Core owns the whole `SectionTabs` (both slots) and takes the station's `addFile`
// node + `busy` flag as props. Behaviour is IDENTICAL to the pre-slice-2a
// FrameExtractStation body: same Options rail, same preview/grid/pager/footer, same
// classes, same run gate. `onProduced` is an OPTIONAL slice-2b sink fired for each
// frame picked into the library; when OMITTED (the station case) the pick still
// pushes to the shared library exactly as before.
//
// Source lifecycle: the station owned `reset()` + clearing the selection/page inside
// onPick/clearSource. The Core now owns the job hook + selection, so it resets them
// whenever the `source` REFERENCE changes (a new pick or a clear). The initial mount
// is skipped (prevSource seed) so a remount after a sub-tab switch keeps the durable
// tracker's finished frames.
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useFrameJobs } from "../useFrameJobs";
import { SectionTabs } from "../SectionTabs";
import { useSidebarRegistry } from "../SidebarRegistry";
// t25: char360 character view-set extraction, reused WHOLESALE from the identities
// station (zero new backend — same extractFromVideo + GET /video/jobs/<id> plumbing).
import { ExtractFromVideoPanel } from "../studio/IdentitiesStation";
import { CharacterGroupsPanel } from "./CharacterGroupsPanel";
import { useIdentityProfiles } from "../studio/useIdentityProfiles";
import { STILL_RUNNING_MESSAGE } from "../pollCaps";
import { useImageModels } from "../../video/useModels";
import { addToLibrary, useMediaLibrary } from "../../video/mediaLibrary";
import { mediaBytesUrl } from "../../config";
import {
  FRAME_FORMATS,
  type FrameFormat,
  type FrameExtractRequest,
  type MediaRef,
} from "../../video/contract";

const PAGE_SIZE = 60;
const MODEL_KEY = "vi.frames.imageModel.v1";

// quality semantics differ by fmt — surfaced, never silently clamped.
function qualityRange(fmt: FrameFormat): { min: number; max: number; hint: string } {
  if (fmt === "png") {
    return {
      min: 0,
      max: 9,
      hint: "PNG · 0–9 compression level (0 = fastest/largest, 9 = smallest).",
    };
  }
  return {
    min: 1,
    max: 100,
    hint: `${fmt.toUpperCase()} · 1–100 quality (higher = better, larger file).`,
  };
}

// Parse a text field into a positive number, or null when blank/invalid.
function toNumberOrNull(raw: string): number | null {
  const s = raw.trim();
  if (s === "") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

// ── INTELLIGENT DEFAULTS (t23; defaults-are-promises) ────────────────────────
// A blank knob resolves to a real, source-informed value so a bare Run on a
// typical clip yields a useful extraction with zero operator input. The resolved
// values are shown in the UI (never a field that "secretly computes").
//
// fps ← clip DURATION. We aim for a total frame count in a useful review band and
// back out the fps that lands there, clamped to sane bounds:
//   target ≈ 180 frames (mid of a ~100–300 band), fps = target / duration_s,
//   clamped to [0.5, 8] fps. Rounded to a friendly step (≥2fps → integer;
//   <2fps → one decimal) so the hint reads cleanly. With no duration (unknown
//   source) we fall back to a flat 2 fps — a sane general default.
const FPS_TARGET_FRAMES = 180;   // mid of the ~100–300 useful band
const FPS_MIN = 0.5;
const FPS_MAX = 8;
const FPS_FALLBACK = 2;          // duration unknown → a sane flat default

function deriveFps(durationS: number | null | undefined): number {
  if (!durationS || durationS <= 0) return FPS_FALLBACK;
  const raw = FPS_TARGET_FRAMES / durationS;
  const clamped = Math.min(FPS_MAX, Math.max(FPS_MIN, raw));
  // Friendly rounding: whole fps at ≥2, one decimal below (0.5 steps read well).
  return clamped >= 2 ? Math.round(clamped) : Math.round(clamped * 2) / 2;
}

// The resulting frame count at a given fps over a duration (for the honest hint).
function framesAt(fps: number, durationS: number | null | undefined): number | null {
  if (!durationS || durationS <= 0 || !fps || fps <= 0) return null;
  return Math.max(1, Math.round(fps * durationS));
}

// quality ← a high, format-appropriate constant. jpg/webp are 1..100 (higher =
// better) so 90 is near-max fidelity at a reasonable file size; png is 0..9
// COMPRESSION (higher = smaller) so 6 is a balanced size/speed default. These are
// the "good result on a typical clip" constants — never left blank.
function defaultQuality(fmt: FrameFormat): number {
  return fmt === "png" ? 6 : 90;
}

export interface FrameExtractCoreProps {
  /** Source video (kind==="video"), or null before one is loaded. */
  source: MediaRef | null;
  /** Upload/ingest in flight (station-owned) — gates Run, exactly as before. */
  busy: boolean;
  /** The persistent add-file bar node (SectionTabs' addFile slot) — station-owned. */
  addFile: ReactNode;
  /** Additive (slice 2b): notified for each frame picked into the library.
      Omitted by the station — picks still push to the shared library regardless. */
  onProduced?: (ref: MediaRef) => void;
}

export function FrameExtractCore({
  source,
  busy,
  addFile,
  onProduced,
}: FrameExtractCoreProps) {
  // ---- knob state (all explicit; fps/quality start empty and must be set) ----
  const [fps, setFps] = useState<string>("");
  const [fmt, setFmt] = useState<FrameFormat>("jpg");
  const [quality, setQuality] = useState<string>("");
  const [windowOn, setWindowOn] = useState<boolean>(false);
  const [startS, setStartS] = useState<string>("");
  const [endS, setEndS] = useState<string>("");
  const [maxFrames, setMaxFrames] = useState<string>("");

  // ---- shared image-model dropdown (config surface reused by Generate) ----
  const { models, defaultId, loading: modelsLoading, error: modelsError, refresh: refreshModels } =
    useImageModels();
  const [modelId, setModelId] = useState<string>(() => {
    try {
      return sessionStorage.getItem(MODEL_KEY) ?? "";
    } catch {
      return "";
    }
  });
  // Seed the selection from the task default once models load (only if unset).
  useEffect(() => {
    if (!modelId && defaultId) setModelId(defaultId);
  }, [defaultId, modelId]);
  useEffect(() => {
    if (!modelId) return;
    try {
      sessionStorage.setItem(MODEL_KEY, modelId);
    } catch {
      /* ignore */
    }
  }, [modelId]);

  // ---- job lifecycle ----
  const { status, frames, error: jobError, capped, running, run, resume, reset } =
    useFrameJobs();

  // ---- Settings tab in the LEFT sidebar (t23) ----
  // The Options knob rail folds into the arm shell's ONE left sidebar "Settings"
  // tab (the same pattern GenerateStation uses): register reveals the guarded
  // Settings tab, and SectionTabs portals the options node into reg.settingsHost
  // — the work column then goes full width. Same shell, no variant.
  const reg = useSidebarRegistry();
  useEffect(() => reg.registerSettings(), [reg.registerSettings]);

  // ---- CHARACTER tools (t25): char360 view-set extraction over the loaded video ----
  // Reuses the identities station's ExtractFromVideoPanel + its extractFromVideo
  // hook wholesale (same GET /video/jobs/<id> polling). The station already loads a
  // VIDEO source, so char360 runs against that same source — no second upload hop.
  const { profiles: idProfiles, extractFromVideo, reload: reloadIdentities } =
    useIdentityProfiles();
  const sourceIsVideo = source?.kind === "video";

  // ---- frame selection + pagination ----
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [page, setPage] = useState<number>(0);
  const library = useMediaLibrary();
  const inLibrary = useMemo(
    () => new Set(library.map((it) => it.ref.uri)),
    [library],
  );

  const qr = qualityRange(fmt);

  // Reset the job + selection when a NEW source is picked (or cleared) — keyed on the
  // source reference so it fires exactly on a pick/clear. The mount is skipped
  // (prevSource seed) so a remount after a tab switch keeps the durable tracker's
  // finished frames.
  const prevSource = useRef<MediaRef | null>(source);
  useEffect(() => {
    if (prevSource.current === source) return;
    prevSource.current = source;
    reset();
    setSelected(new Set());
    setPage(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  // ---- run validation (no silent defaults; every gate is visible) ----
  const fpsNum = toNumberOrNull(fps);
  const qualityNum = toNumberOrNull(quality);
  const maxFramesNum = toNumberOrNull(maxFrames);
  const startNum = toNumberOrNull(startS);
  const endNum = toNumberOrNull(endS);

  // ---- resolved (effective) values (t23) ----
  // A BLANK field resolves to its intelligent default: fps from the source's
  // duration, quality from the format. A TYPED value overrides. These are what
  // Run sends and what the UI surfaces as the honest "(auto: …)" hint, so a bare
  // Run on a typical clip produces a good extraction with no operator input.
  const autoFps = deriveFps(source?.duration_s);
  const autoQuality = defaultQuality(fmt);
  const effectiveFps = fpsNum != null ? fpsNum : autoFps;
  const effectiveQuality = qualityNum != null ? qualityNum : autoQuality;
  const fpsIsAuto = fpsNum == null;         // blank → showing the derived default
  const qualityIsAuto = qualityNum == null;
  const autoFrames = framesAt(effectiveFps, source?.duration_s);

  // A field is INVALID only when NON-BLANK and out of range — blank is valid now
  // (it means "use the auto default"), never a Run-blocking empty.
  const fpsValid = fpsNum == null || fpsNum > 0;
  const qualityValid =
    qualityNum == null ||
    (Number.isInteger(qualityNum) && qualityNum >= qr.min && qualityNum <= qr.max);
  const maxFramesValid =
    maxFrames.trim() === "" ||
    (maxFramesNum != null && Number.isInteger(maxFramesNum) && maxFramesNum > 0);
  const windowValid =
    !windowOn ||
    (startNum != null && endNum != null && startNum >= 0 && endNum > startNum);

  const canRun =
    !!source &&
    !busy &&
    !running &&
    fpsValid &&
    qualityValid &&
    maxFramesValid &&
    windowValid;

  function onRun() {
    if (!source) return;
    setSelected(new Set());
    setPage(0);
    const req: FrameExtractRequest = {
      source,
      // Send the RESOLVED values — a blank field ships its intelligent default.
      fps: effectiveFps,
      quality: effectiveQuality,
      fmt,
      window: windowOn && startNum != null && endNum != null
        ? { start_s: startNum, end_s: endNum }
        : null,
      max_frames: maxFrames.trim() === "" ? null : maxFramesNum,
    };
    run(req);
  }

  // ---- selection helpers ----
  function toggleFrame(uri: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(uri)) next.delete(uri);
      else next.add(uri);
      return next;
    });
  }

  const pageCount = Math.max(1, Math.ceil(frames.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount - 1);
  const pageStart = clampedPage * PAGE_SIZE;
  const pageFrames = frames.slice(pageStart, pageStart + PAGE_SIZE);

  function selectAllOnPage() {
    setSelected((prev) => {
      const next = new Set(prev);
      for (const f of pageFrames) next.add(f.uri);
      return next;
    });
  }
  function clearSelection() {
    setSelected(new Set());
  }

  // Push every selected frame into the shared library (dedupes by uri). Frames the
  // user unselects are NOT removed from the library (house policy: no deletion).
  function addSelectedToGenerate() {
    let n = 0;
    frames.forEach((f, i) => {
      if (selected.has(f.uri)) {
        addToLibrary(f, "frames", `frame ${i + 1}`);
        onProduced?.(f);
        n += 1;
      }
    });
    return n;
  }

  const selectedCount = selected.size;
  const showGrid = frames.length > 0;

  // The large center preview mirrors the strip: the first selected frame, or the
  // first extracted frame when nothing is selected yet. Pure derivation — no new
  // interaction, no change to the poll/selection contract.
  const focusFrame = useMemo(() => {
    const sel = frames.find((f) => selected.has(f.uri));
    return sel ?? frames[0] ?? null;
  }, [frames, selected]);

  return (
    <SectionTabs
      ariaLabel="Frames station"
      initialTab="components"
      addFile={addFile}
      optionsLabel="Settings"
      optionsHost={reg.settingsHost}
      options={
        /* OPTIONS — explicit knobs, no silent defaults. */
        <aside className="vi-frames-rail">
          <div className="vi-knob">
            <label htmlFor="vi-model">Image model</label>
            <select
              id="vi-model"
              className="vi-knob-select"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={modelsLoading}
            >
              {modelsLoading && <option value="">Loading models…</option>}
              {!modelsLoading && models.every((m) => m.disabled) && (
                <option value="">No image models available</option>
              )}
              {/* Adapters / pipeline components / unclassified rows appear greyed
                  with the backend's reason instead of being offered (k61). */}
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={m.disabled} title={m.reason}>
                  {m.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              onClick={refreshModels}
              disabled={modelsLoading}
              title="Re-scan the registry for image models adopted this session"
            >
              ↻ Refresh models
            </button>
            <span className="vi-knob-hint">
              Shared image model — carried to the Generate station. Not used to
              extract frames.
            </span>
            {modelsError && <span className="vi-error">{modelsError}</span>}
          </div>

          <div className="vi-knob">
            <label htmlFor="vi-fps">fps</label>
            <input
              id="vi-fps"
              type="number"
              min="0"
              step="any"
              inputMode="decimal"
              className="vi-knob-input"
              placeholder={`auto: ${autoFps}`}
              value={fps}
              onChange={(e) => setFps(e.target.value)}
            />
            <span className="vi-knob-hint">
              Frames sampled per second of video.{" "}
              {fpsIsAuto ? (
                <span className="vi-knob-auto">
                  Using {effectiveFps} fps (auto
                  {autoFrames != null && source?.duration_s
                    ? `: ~${autoFrames} frames from ${Math.round(source.duration_s)}s`
                    : ""}
                  ). Type to override.
                </span>
              ) : (
                autoFrames != null && (
                  <span className="vi-knob-auto">≈ {autoFrames} frames.</span>
                )
              )}
              {!fpsValid && fps.trim() !== "" && (
                <span className="vi-knob-flag"> Enter a positive number.</span>
              )}
            </span>
          </div>

          <div className="vi-knob">
            <label htmlFor="vi-fmt">format</label>
            <select
              id="vi-fmt"
              className="vi-knob-select"
              value={fmt}
              onChange={(e) => setFmt(e.target.value as FrameFormat)}
            >
              {FRAME_FORMATS.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          </div>

          <div className="vi-knob">
            <label htmlFor="vi-quality">quality</label>
            <input
              id="vi-quality"
              type="number"
              min={qr.min}
              max={qr.max}
              step="1"
              inputMode="numeric"
              className="vi-knob-input"
              placeholder={`auto: ${autoQuality}`}
              value={quality}
              onChange={(e) => setQuality(e.target.value)}
            />
            <span className="vi-knob-hint vi-knob-range">{qr.hint}</span>
            {qualityIsAuto && (
              <span className="vi-knob-auto">
                Using {effectiveQuality} ({fmt === "png" ? "balanced compression" : "high quality"}). Type to override.
              </span>
            )}
            {!qualityValid && quality.trim() !== "" && (
              <span className="vi-knob-flag">
                Enter an integer in {qr.min}–{qr.max} for {fmt.toUpperCase()}.
              </span>
            )}
          </div>

          <div className="vi-knob">
            <label className="vi-knob-check">
              <input
                type="checkbox"
                checked={windowOn}
                onChange={(e) => setWindowOn(e.target.checked)}
              />
              window (time slice)
            </label>
            {windowOn ? (
              <div className="vi-knob-window">
                <label className="vi-knob-sub">
                  start_s
                  <input
                    type="number"
                    min="0"
                    step="any"
                    className="vi-knob-input vi-knob-input-sm"
                    value={startS}
                    onChange={(e) => setStartS(e.target.value)}
                  />
                </label>
                <label className="vi-knob-sub">
                  end_s
                  <input
                    type="number"
                    min="0"
                    step="any"
                    className="vi-knob-input vi-knob-input-sm"
                    value={endS}
                    onChange={(e) => setEndS(e.target.value)}
                  />
                </label>
              </div>
            ) : (
              <span className="vi-knob-hint">
                Off — the WHOLE video is extracted (window: null).
              </span>
            )}
            {windowOn && !windowValid && (
              <span className="vi-knob-flag">
                Need 0 ≤ start &lt; end (seconds).
              </span>
            )}
          </div>

          <div className="vi-knob">
            <label htmlFor="vi-maxframes">max_frames (optional)</label>
            <input
              id="vi-maxframes"
              type="number"
              min="1"
              step="1"
              inputMode="numeric"
              className="vi-knob-input"
              placeholder="uncapped"
              value={maxFrames}
              onChange={(e) => setMaxFrames(e.target.value)}
            />
            <span className="vi-knob-hint vi-knob-flag">
              Hard cap. If the extraction would exceed it, the job FAILS LOUDLY
              (status: failed) — it never silently truncates.
            </span>
            {!maxFramesValid && (
              <span className="vi-knob-flag">Enter a positive integer or leave blank.</span>
            )}
          </div>

        </aside>
      }
      components={
        /* COMPONENTS — the working UI: preview over the frame grid, with the
           post-extract toolbar + Run action as an in-panel footer. */
        <div className="vi-station-components">
          {/* Preview area — ALWAYS rendered (blank frame area when no video is
              loaded). The Options, add-file bar, and Run footer stay visible
              regardless. */}
          <div className="vi-studio-preview">
            {focusFrame ? (
              <img
                className="vi-studio-preview-img"
                src={mediaBytesUrl(focusFrame.uri)}
                alt="Selected frame"
              />
            ) : running ? (
              <p className="vi-crop-hint">
                Working… frames appear here when the job finishes.
              </p>
            ) : source ? (
              <p className="vi-crop-hint">
                Set the options and run to extract frames. Each frame lands as a
                MediaRef (never inlined); pick the ones you want for Generate.
              </p>
            ) : null}
          </div>

          {showGrid && (
            <div className="vi-frames-grid vi-studio-strip">
              {pageFrames.map((f, i) => {
                const globalIdx = pageStart + i;
                const picked = selected.has(f.uri);
                const saved = inLibrary.has(f.uri);
                return (
                  <button
                    type="button"
                    key={f.uri}
                    className={
                      picked ? "vi-frame-cell vi-frame-cell-selected" : "vi-frame-cell"
                    }
                    onClick={() => toggleFrame(f.uri)}
                    aria-pressed={picked}
                    title={f.uri}
                  >
                    <img
                      className="vi-frame-img"
                      src={mediaBytesUrl(f.uri)}
                      alt={`Frame ${globalIdx + 1}`}
                      loading="lazy"
                    />
                    <span className="vi-frame-tag">#{globalIdx + 1}</span>
                    {picked && <span className="vi-frame-check">✓</span>}
                    {saved && <span className="vi-frame-saved">in library</span>}
                  </button>
                );
              })}
            </div>
          )}

          {showGrid && pageCount > 1 && (
            <div className="vi-pager">
              <button
                type="button"
                className="vi-btn vi-btn-sm"
                disabled={clampedPage === 0}
                onClick={() => setPage(clampedPage - 1)}
              >
                ‹ Prev
              </button>
              <span className="vi-pager-label">
                page {clampedPage + 1} / {pageCount}
              </span>
              <button
                type="button"
                className="vi-btn vi-btn-sm"
                disabled={clampedPage >= pageCount - 1}
                onClick={() => setPage(clampedPage + 1)}
              >
                Next ›
              </button>
            </div>
          )}

        {/* FOOTER — post-extract toolbar + primary Run action, INSIDE the
            components panel (nothing floats outside the panels). */}
        <div className="vi-station-footer">
          <div className="vi-station-footer-main">
            {showGrid ? (
              <div className="vi-frames-toolbar">
                <span className="vi-frames-summary">
                  {frames.length} frame{frames.length === 1 ? "" : "s"} ·{" "}
                  {selectedCount} selected
                </span>
                <div className="vi-frames-actions">
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-ghost"
                    onClick={selectAllOnPage}
                  >
                    Select page
                  </button>
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-ghost"
                    onClick={clearSelection}
                    disabled={selectedCount === 0}
                  >
                    Clear
                  </button>
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-accent"
                    onClick={addSelectedToGenerate}
                    disabled={selectedCount === 0}
                  >
                    Add {selectedCount} to Generate
                  </button>
                </div>
              </div>
            ) : (
              <p className="vi-crop-hint">
                Set the options, then Extract to fill the strip above.
              </p>
            )}
          </div>

          <div className="vi-station-footer-run">
            <div className="vi-frames-run">
              <button
                type="button"
                className="vi-btn vi-btn-accent"
                disabled={!canRun}
                onClick={onRun}
              >
                {running ? "Extracting…" : "Extract frames"}
              </button>
              {status && (
                <span className={`vi-status vi-status-${status}`}>{status}</span>
              )}
            </div>
            {jobError && <p className="vi-error" role="alert">{jobError}</p>}
            {capped && (
              <div className="vi-frames-run">
                <span className="vi-knob-flag">{STILL_RUNNING_MESSAGE}</span>
                <button type="button" className="vi-btn vi-btn-sm" onClick={resume}>
                  Check again
                </button>
              </div>
            )}
          </div>
        </div>

        {/* ── CHARACTER TOOLS (t25) ──────────────────────────────────────────
            A first-class group within the Frames station for character-centric
            work on the loaded video / extracted frames. char360 view-set
            extraction is its FIRST resident. This group is deliberately laid out
            as a titled area so the pending character-aggregation modes (t19,
            approved-pending: a during-processing toggle, an "aggregate now"
            action, frame-multi-select→associate, a seek-identity picker) slot in
            HERE as additional residents later — no aggregation is built now, just
            the structural room. Rendered only when the source is a VIDEO (char360
            operates on video); a still-frame source hides it honestly. */}
        {sourceIsVideo && (
          <section className="vi-character-tools" aria-label="Character tools">
            <div className="vi-character-tools-head">
              <span className="vi-comfy-label">Character tools</span>
              <span className="vi-knob-hint" style={{ opacity: 0.7 }}>
                Character-centric extraction over the loaded video.
              </span>
            </div>
            {/* Resident 1 — char360 view-set extraction (reused wholesale). */}
            <ExtractFromVideoPanel
              profiles={idProfiles}
              extractFromVideo={extractFromVideo}
              reload={reloadIdentities}
              presetVideo={source}
            />
            {/* Resident 2 (S2+S4) — char360 REVIEW + curate: non-committing extraction
                → editable per-character groups → commit selected to identity
                profiles → per-group "Generate 3D" (S4). Shares the SINGLE
                useIdentityProfiles hook's extractFromVideo (never a second mount).
                The panel owns its 3D job state internally; onGenerate3D is left
                omitted — nothing outside /video/frames needs the notify hook. */}
            <CharacterGroupsPanel source={source} extractFromVideo={extractFromVideo} />
          </section>
        )}
        </div>
      }
    />
  );
}

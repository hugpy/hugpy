// SpatialRegionEditor — a draggable/resizable bbox over a source image, the first
// concrete implementation of the RegionEditor contract (regions/types.ts). Used by
// the Image Crop station in multiRegion mode: the user shapes a working box (drag
// to move, corner handles to resize, aspect lock, preset buttons, numeric fields),
// adds it to a queue, and the queue is surfaced to the station via onChange.
//
// CRITICAL CORRECTNESS: the box the user manipulates lives in DISPLAY pixels (the
// image is scaled to fit), but CropSpec regions MUST be in the image's NATIVE pixel
// space. We keep the working region in native px at all times and convert only for
// painting/hit-testing, using scale = source.width / renderedWidth. Every emitted
// region is clamped to integers within [0..width] x [0..height].
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RegionEditorProps, SpatialPreset, SpatialRegion } from "./types";
import type { MediaRef } from "../video/contract";
import { mediaBytesUrl } from "../config";

/** Model-native starting boxes offered by default. */
export const DEFAULT_SPATIAL_PRESETS: readonly SpatialPreset[] = [
  { label: "1:1", aspect: 1 },
  { label: "16:9", aspect: 16 / 9 },
  { label: "9:16", aspect: 9 / 16 },
  { label: "4:3", aspect: 4 / 3 },
  { label: "1024²", aspect: 1, size: { w: 1024, h: 1024 } },
  { label: "Native", aspect: null },
] as const;

/** Fallback canvas dimensions used to lay out the editor when no image is
    loaded yet — the chrome renders full, just over a blank canvas. */
const DEFAULT_DIM = 1024;

type Corner = "nw" | "ne" | "sw" | "se";

interface DragState {
  mode: "move" | "resize";
  corner: Corner | null;
  startClientX: number;
  startClientY: number;
  startBox: SpatialRegion;
  /** native px per display px, captured at drag start. */
  scaleX: number;
  scaleY: number;
  /** locked width/height ratio during a resize, or null when unlocked. */
  aspect: number | null;
}

interface Size {
  w: number;
  h: number;
}

// ---- pure geometry (native px) ------------------------------------------------

/** Clamp a region to integer pixels inside [0..W] x [0..H], min 1x1. */
function clampRegion(r: SpatialRegion, W: number, H: number): SpatialRegion {
  const w = Math.max(1, Math.min(Math.round(r.w), W));
  const h = Math.max(1, Math.min(Math.round(r.h), H));
  const x = Math.max(0, Math.min(Math.round(r.x), W - w));
  const y = Math.max(0, Math.min(Math.round(r.y), H - h));
  return { x, y, w, h };
}

/** Seed a centered region for a preset, clamped to the source. */
function presetToRegion(p: SpatialPreset, W: number, H: number): SpatialRegion {
  let w: number;
  let h: number;
  if (p.size) {
    w = p.size.w;
    h = p.size.h;
  } else if (p.aspect == null) {
    w = W;
    h = H;
  } else if (W / H > p.aspect) {
    // source is wider than target aspect → height-bound
    h = H;
    w = H * p.aspect;
  } else {
    w = W;
    h = W / p.aspect;
  }
  w = Math.min(w, W);
  h = Math.min(h, H);
  return clampRegion({ x: (W - w) / 2, y: (H - h) / 2, w, h }, W, H);
}

/** The corner opposite the dragged one — stays fixed during a resize. */
function anchorOf(corner: Corner, b: SpatialRegion): { ax: number; ay: number } {
  const right = b.x + b.w;
  const bottom = b.y + b.h;
  switch (corner) {
    case "nw":
      return { ax: right, ay: bottom };
    case "ne":
      return { ax: b.x, ay: bottom };
    case "sw":
      return { ax: right, ay: b.y };
    case "se":
      return { ax: b.x, ay: b.y };
  }
}

const CORNERS: Corner[] = ["nw", "ne", "sw", "se"];

// ---- rendered-size hook -------------------------------------------------------

/** Track an element's rendered (display) box, updating on load + resize. */
function useRenderedSize(ref: React.RefObject<HTMLElement>, dep: unknown): Size {
  const [size, setSize] = useState<Size>({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dep]);
  return size;
}

// ---- component ----------------------------------------------------------------

interface SpatialRegionEditorProps
  extends RegionEditorProps<SpatialRegion, SpatialPreset> {
  /** Source image (kind==="image"), or null before one is loaded — the editor
      then renders its full chrome over a blank default-sized canvas. */
  source: MediaRef | null;
}

export function SpatialRegionEditor({
  source,
  values,
  onChange,
  presets = DEFAULT_SPATIAL_PRESETS as SpatialPreset[],
  numericInputs = true,
  multiRegion = true,
}: SpatialRegionEditorProps) {
  const W = source?.width || DEFAULT_DIM;
  const H = source?.height || DEFAULT_DIM;

  // One measuring ref for whichever canvas element renders (the <img>, or the
  // blank placeholder when no image is loaded).
  const canvasElRef = useRef<HTMLElement | null>(null);
  const measureRef = useCallback((el: HTMLElement | null) => {
    canvasElRef.current = el;
  }, []);
  const rendered = useRenderedSize(canvasElRef, source?.uri);

  const [working, setWorking] = useState<SpatialRegion>(() =>
    presetToRegion({ label: "seed", aspect: null }, W, H),
  );
  const [aspectLock, setAspectLock] = useState(false);
  const [drag, setDrag] = useState<DragState | null>(null);

  // Reset the working box (to a slightly inset native frame) when the source swaps.
  useEffect(() => {
    setWorking(clampRegion({ x: W * 0.1, y: H * 0.1, w: W * 0.8, h: H * 0.8 }, W, H));
  }, [source?.uri, W, H]);

  const queue = values ?? [];

  // display px per native px — the inverse of the native-space scale factor.
  const dispScaleX = rendered.w > 0 && W > 0 ? rendered.w / W : 0;
  const dispScaleY = rendered.h > 0 && H > 0 ? rendered.h / H : 0;

  const setClamped = useCallback(
    (r: SpatialRegion) => setWorking(clampRegion(r, W, H)),
    [W, H],
  );

  // ---- drag lifecycle: window listeners while a pointer is down ----
  useEffect(() => {
    if (!drag) return;
    const onMove = (e: PointerEvent) => {
      const dxN = (e.clientX - drag.startClientX) * drag.scaleX;
      const dyN = (e.clientY - drag.startClientY) * drag.scaleY;
      if (drag.mode === "move") {
        setClamped({ ...drag.startBox, x: drag.startBox.x + dxN, y: drag.startBox.y + dyN });
        return;
      }
      const corner = drag.corner as Corner;
      const { ax, ay } = anchorOf(corner, drag.startBox);
      // dragged corner's new native position
      const sc =
        corner === "nw"
          ? { x: drag.startBox.x, y: drag.startBox.y }
          : corner === "ne"
            ? { x: drag.startBox.x + drag.startBox.w, y: drag.startBox.y }
            : corner === "sw"
              ? { x: drag.startBox.x, y: drag.startBox.y + drag.startBox.h }
              : { x: drag.startBox.x + drag.startBox.w, y: drag.startBox.y + drag.startBox.h };
      const cx = sc.x + dxN;
      const cy = sc.y + dyN;
      let w = Math.abs(cx - ax);
      let h = Math.abs(cy - ay);
      if (drag.aspect != null && drag.aspect > 0) h = w / drag.aspect;
      const nx = cx >= ax ? ax : ax - w;
      const ny = cy >= ay ? ay : ay - h;
      setClamped({ x: nx, y: ny, w, h });
    };
    const onUp = () => setDrag(null);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [drag, setClamped]);

  const startMove = (e: React.PointerEvent) => {
    if (dispScaleX <= 0) return;
    e.preventDefault();
    setDrag({
      mode: "move",
      corner: null,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startBox: working,
      scaleX: 1 / dispScaleX,
      scaleY: 1 / dispScaleY,
      aspect: null,
    });
  };

  const startResize = (corner: Corner) => (e: React.PointerEvent) => {
    if (dispScaleX <= 0) return;
    e.preventDefault();
    e.stopPropagation();
    setDrag({
      mode: "resize",
      corner,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startBox: working,
      scaleX: 1 / dispScaleX,
      scaleY: 1 / dispScaleY,
      aspect: aspectLock && working.h > 0 ? working.w / working.h : null,
    });
  };

  // ---- numeric field editing (native px) ----
  const setField = (key: keyof SpatialRegion) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = Number(e.target.value);
    const v = Number.isFinite(raw) ? raw : 0;
    setClamped({ ...working, [key]: v });
  };

  // ---- queue mutations (surface via onChange) ----
  const addToQueue = () => {
    const region = clampRegion(working, W, H);
    onChange(multiRegion ? [...queue, region] : region);
  };
  const removeAt = (i: number) => onChange(queue.filter((_, idx) => idx !== i));
  const applyPreset = (p: SpatialPreset) => setWorking(presetToRegion(p, W, H));

  // display geometry for the overlay box
  const boxStyle = useMemo(
    () => ({
      left: `${working.x * dispScaleX}px`,
      top: `${working.y * dispScaleY}px`,
      width: `${working.w * dispScaleX}px`,
      height: `${working.h * dispScaleY}px`,
    }),
    [working, dispScaleX, dispScaleY],
  );

  return (
    <div className="vi-crop-editor">
      <div className="vi-crop-canvas">
        {source ? (
          <img
            ref={measureRef}
            className="vi-crop-img"
            src={mediaBytesUrl(source.uri)}
            alt="Crop source"
            draggable={false}
          />
        ) : (
          // No image yet — a blank canvas of the default aspect keeps the full
          // editor chrome laid out; only the media itself is missing.
          <div
            ref={measureRef}
            className="vi-crop-img vi-crop-img--blank"
            style={{ aspectRatio: `${W} / ${H}` }}
            aria-hidden
          />
        )}
        <div className="vi-crop-overlay">
          <div
            className={drag?.mode === "move" ? "vi-crop-box vi-crop-box-active" : "vi-crop-box"}
            style={boxStyle}
            onPointerDown={startMove}
          >
            {CORNERS.map((c) => (
              <span
                key={c}
                className={`vi-crop-handle vi-crop-handle-${c}`}
                onPointerDown={startResize(c)}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="vi-crop-controls">
        <div className="vi-crop-presets">
          {presets.map((p) => (
            <button
              key={p.label}
              type="button"
              className="vi-btn vi-btn-ghost"
              onClick={() => applyPreset(p)}
            >
              {p.label}
            </button>
          ))}
          <label className="vi-crop-lock">
            <input
              type="checkbox"
              checked={aspectLock}
              onChange={(e) => setAspectLock(e.target.checked)}
            />
            Lock aspect
          </label>
        </div>

        {numericInputs && (
          <div className="vi-crop-fields">
            {(["x", "y", "w", "h"] as const).map((k) => (
              <label key={k} className="vi-crop-field">
                <span>{k}</span>
                <input
                  type="number"
                  min={0}
                  max={k === "x" || k === "w" ? W : H}
                  value={working[k]}
                  onChange={setField(k)}
                />
              </label>
            ))}
            <span className="vi-crop-native">
              native {W}×{H}
            </span>
          </div>
        )}

        <div className="vi-crop-actions">
          <button type="button" className="vi-btn vi-btn-accent" onClick={addToQueue}>
            {multiRegion ? "Add crop to queue" : "Set crop"}
          </button>
        </div>

        {multiRegion && queue.length > 0 && (
          <ul className="vi-region-list">
            {queue.map((r, i) => (
              <li key={i} className="vi-region-item">
                <span className="vi-region-idx">#{i + 1}</span>
                <code className="vi-region-dims">
                  {r.w}×{r.h} @ {r.x},{r.y}
                </code>
                <button
                  type="button"
                  className="vi-btn vi-btn-ghost vi-btn-sm"
                  onClick={() => removeAt(i)}
                >
                  remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

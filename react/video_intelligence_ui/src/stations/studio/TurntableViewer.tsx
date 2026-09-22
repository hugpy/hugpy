// TURNTABLE VIEWER — the drag-to-rotate half of the identity 360° orbit turntable.
//
// A turntable reconstruction stores ONE orbit clip's frames in its `views[]` array
// IN ANGULAR ORDER (frame 0 = start angle … frame N ≈ 360°). This component shows
// exactly ONE frame at a time and scrubs across the ordered frames as the user drags
// horizontally (pointer events) or moves the range slider — dragging/sliding left↔
// right maps position across [0..N-1] = rotating the character 0..360°. Every frame
// is a degree view, so scrubbing IS the rotation.
//
// Frames are PRELOADED (a `new Image()` per uri) on mount/whenever the frame set
// changes, so once warm the on-screen <img> src swap is instant (served from cache)
// and scrubbing is smooth. Served bytes come through `mediaBytesUrl(uri)`, the same
// media-bytes seam every other identity thumb uses.
//
// Presentational + self-contained: it owns only the current-frame index and drag
// bookkeeping; the caller (IdentitiesStation) gates rendering this vs the static
// sheet grid on the reconstruction's `mode`, and keeps its own approve-to-canonical
// control alongside.
import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { mediaBytesUrl } from "../../config";

function clampIndex(i: number, n: number): number {
  if (n <= 0) return 0;
  if (i < 0) return 0;
  if (i > n - 1) return n - 1;
  return i;
}

export function TurntableViewer({
  frames,
  degreesPerFrame,
  frameCount,
}: {
  /** The reconstruction's `views[]` — the orbit clip's frame uris, in angular order. */
  frames: string[];
  /** Optional manifest hint: degrees advanced per frame (else derived from count). */
  degreesPerFrame?: number | null;
  /** Optional manifest hint: authoritative frame count (else `frames.length`). */
  frameCount?: number | null;
}) {
  const n = frames.length;
  // `frame_count` is a manifest convenience; the frames we can actually SHOW are the
  // uris we were handed, so scrubbing is always bounded by `frames.length`.
  const count = frameCount && frameCount > 0 ? frameCount : n;
  const dpf = useMemo(() => {
    if (degreesPerFrame && degreesPerFrame > 0) return degreesPerFrame;
    return count > 0 ? 360 / count : 0;
  }, [degreesPerFrame, count]);

  const [index, setIndex] = useState(0);
  const [loaded, setLoaded] = useState(0);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ startX: number; startIndex: number } | null>(null);

  // Keep the index valid if the frame set changes out from under us (e.g. a fresh
  // turntable replaces an older one on this same card).
  useEffect(() => {
    setIndex((i) => clampIndex(i, n));
  }, [n]);

  // PRELOAD every frame. Hold the Image objects in a ref for the effect's lifetime so
  // the browser keeps them warm (and doesn't GC an in-flight decode); count decoded
  // frames for a small "loading" readout while the orbit warms up.
  useEffect(() => {
    setLoaded(0);
    if (n === 0) return;
    let live = true;
    const imgs: HTMLImageElement[] = [];
    frames.forEach((uri) => {
      const img = new Image();
      img.onload = () => {
        if (live) setLoaded((c) => c + 1);
      };
      img.onerror = () => {
        // A broken frame still "settles" so the readout can reach N and not hang.
        if (live) setLoaded((c) => c + 1);
      };
      img.src = mediaBytesUrl(uri);
      imgs.push(img);
    });
    return () => {
      live = false;
      imgs.forEach((img) => {
        img.onload = null;
        img.onerror = null;
      });
    };
  }, [frames, n]);

  if (n === 0) return null;

  // Map a horizontal drag across the FULL stage width to the FULL frame range: a drag
  // spanning the whole viewer = a full 0..360° turn. Dragging RIGHT advances frames
  // (the character rotates), dragging LEFT rewinds.
  function onPointerDown(e: ReactPointerEvent<HTMLDivElement>) {
    (e.currentTarget as HTMLDivElement).setPointerCapture?.(e.pointerId);
    dragRef.current = { startX: e.clientX, startIndex: index };
  }
  function onPointerMove(e: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    const width = stageRef.current?.clientWidth || 1;
    const delta = e.clientX - drag.startX;
    const framesMoved = Math.round((delta / width) * (count - 1 || 1));
    setIndex(clampIndex(drag.startIndex + framesMoved, n));
  }
  function endDrag(e: ReactPointerEvent<HTMLDivElement>) {
    (e.currentTarget as HTMLDivElement).releasePointerCapture?.(e.pointerId);
    dragRef.current = null;
  }

  const angle = Math.round(index * dpf) % 360;
  const warming = loaded < n;

  return (
    <div style={{ marginTop: "0.4rem" }}>
      <p className="vi-comfy-label" style={{ margin: "0 0 0.3rem" }}>
        360° turntable
      </p>
      <div
        ref={stageRef}
        className="vi-turntable-stage"
        role="slider"
        aria-label="Drag to rotate the character"
        aria-valuemin={0}
        aria-valuemax={n - 1}
        aria-valuenow={index}
        aria-valuetext={`${angle}°`}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") setIndex((i) => clampIndex(i + 1, n));
          else if (e.key === "ArrowLeft") setIndex((i) => clampIndex(i - 1, n));
        }}
        style={{
          position: "relative",
          width: "100%",
          maxWidth: "18rem",
          aspectRatio: "1 / 1",
          borderRadius: "0.5rem",
          overflow: "hidden",
          background: "#000",
          border: "1px solid var(--vi-border)",
          touchAction: "none",
          userSelect: "none",
        }}
      >
        {frames.map((uri, i) => (
          // Render ALL frames stacked, toggling visibility, so the already-decoded
          // frame is simply revealed on scrub (no src swap / reflow flicker at all).
          <img
            key={uri}
            src={mediaBytesUrl(uri)}
            alt={i === index ? `orbit frame ${i + 1} of ${n} (${angle}°)` : ""}
            draggable={false}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              opacity: i === index ? 1 : 0,
              pointerEvents: "none",
            }}
          />
        ))}
        <span
          className="vi-comfy-hint"
          style={{
            position: "absolute",
            top: 4,
            right: 6,
            fontSize: "0.7rem",
            padding: "0 0.3rem",
            borderRadius: "0.3rem",
            background: "rgba(0,0,0,0.55)",
            textShadow: "0 0 3px #000",
          }}
        >
          {angle}°
        </span>
        {warming && (
          <span
            className="vi-comfy-hint"
            style={{
              position: "absolute",
              left: 6,
              bottom: 4,
              fontSize: "0.7rem",
              padding: "0 0.3rem",
              borderRadius: "0.3rem",
              background: "rgba(0,0,0,0.55)",
              opacity: 0.9,
            }}
          >
            preloading {loaded}/{n}…
          </span>
        )}
      </div>
      <input
        type="range"
        className="vi-turntable-scrub"
        min={0}
        max={n - 1}
        step={1}
        value={index}
        aria-label="Rotate the character"
        onChange={(e) => setIndex(clampIndex(Number(e.target.value), n))}
        style={{ width: "100%", maxWidth: "18rem", marginTop: "0.4rem", display: "block", cursor: "ew-resize" }}
      />
      <p className="vi-comfy-hint" style={{ margin: "0.25rem 0 0", opacity: 0.8 }}>
        {n} frame{n === 1 ? "" : "s"} · ~{dpf ? dpf.toFixed(dpf < 10 ? 1 : 0) : "?"}°/frame · drag or scrub to rotate
      </p>
    </div>
  );
}

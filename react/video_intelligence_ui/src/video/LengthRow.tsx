// LENGTH ROW (operator 2026-08-13, linkage reworked 2026-08-27): one dedicated
// row for clip length — fps | length-frames | length-seconds — any TWO set the
// THIRD. The derived field is the one LEAST-RECENTLY touched (and starts as
// FRAMES = fps × seconds); it renders dimmed but stays editable — typing in it
// makes IT authoritative and demotes the stalest of the other two. Seconds
// never leaves the browser: the payload stays fps + frames; the server still
// clamps to the model ceiling and snaps to the 4k+1 cadence (the hint says so).
// ONE component for every frames-based surface (clip, cinema movie-level,
// cinema per-goal — where fps is movie-level and renders fixed/grey, so
// frames<->seconds derive against it).
import { useEffect, useRef, useState } from "react";

type Field = "fps" | "frames" | "seconds";

export function LengthRow({
  idPrefix,
  fps,
  onFps,
  fpsFixedReason,
  frames,
  onFrames,
  defaultFrames = 81,
  framesPlaceholder = "model default (81)",
}: {
  idPrefix: string;
  /** Current fps (host state, a number). */
  fps: number;
  /** Omit to render fps FIXED (grey) — e.g. cinema per-goal, where fps is movie-level. */
  onFps?: (n: number) => void;
  /** Tooltip explaining WHY fps is fixed, when it is. */
  fpsFixedReason?: string;
  /** Frames as the host stores it: a string, "" = model default. */
  frames: string;
  onFrames: (v: string) => void;
  defaultFrames?: number;
  framesPlaceholder?: string;
}) {
  const [secondsText, setSecondsText] = useState("");
  // Last-two-touched ring; the missing one is DERIVED. Starts seconds+fps, so
  // FRAMES opens as the derived display (frames = fps × seconds, e.g. 24 × 3.4)
  // and an fps edit keeps the duration by recomputing frames — while the first
  // direct frames edit demotes seconds (fps stays put), matching "the derived
  // slot is whichever the user adjusted longest ago".
  const touchRef = useRef<Field[]>(["seconds", "fps"]);
  const touch = (f: Field) => {
    const t = touchRef.current.filter((x) => x !== f);
    t.push(f);
    touchRef.current = t.slice(-2);
  };
  const fields: Field[] = ["fps", "frames", "seconds"];
  // The derived slot is the field touched LONGEST ago. When fps is fixed it can
  // never be the derived (writable) target: fall back to the STALER of
  // frames/seconds — touchRef is ordered oldest→newest, so that's [0].
  const deriveNow = (): Field => {
    const missing = fields.find((f) => !touchRef.current.includes(f)) ?? "seconds";
    if (!onFps && missing === "fps") return touchRef.current[0];
    return missing;
  };
  const derived = deriveNow();

  const framesNum = frames.trim() === "" ? null : Number(frames);
  const effFrames = framesNum != null && Number.isFinite(framesNum) ? framesNum : defaultFrames;
  const secondsNum = secondsText.trim() === "" ? null : Number(secondsText);

  const secondsDisplay =
    derived === "seconds"
      ? fps > 0
        ? (effFrames / fps).toFixed(1)
        : ""
      : secondsText;
  const framesDisplay = derived === "frames" && frames.trim() === "" ? String(effFrames) : frames;

  const dim = (f: Field) =>
    derived === f ? { opacity: 0.55 } : undefined;
  const derivedTitle =
    "Derived from the other two — type here to make this the authoritative value.";

  // Handlers recompute the derived slot AFTER recording the touch — the
  // render-time `derived` is one edit behind by definition (touchRef is a ref),
  // and using it here is what used to make a seconds edit silently rewrite fps.
  const selfFpsRef = useRef(false);
  const changeFps = (v: string) => {
    const n = Number(v);
    if (!onFps || !Number.isFinite(n) || n < 1) return;
    selfFpsRef.current = true;
    onFps(Math.round(n));
    touch("fps");
    if (deriveNow() === "frames") {
      // frames = fps × seconds; when the user never typed seconds, derive it
      // from what the row currently shows so the duration they see is kept.
      const s = secondsNum != null && secondsNum > 0 ? secondsNum : fps > 0 ? effFrames / fps : null;
      if (s != null && s > 0) onFrames(String(Math.max(1, Math.round(s * Math.round(n)))));
    }
  };
  const changeFrames = (v: string) => {
    onFrames(v);
    touch("frames");
    const n = Number(v);
    if (!Number.isFinite(n) || n <= 0) return;
    if (deriveNow() === "fps" && onFps && secondsNum != null && secondsNum > 0) {
      onFps(Math.max(1, Math.round(n / secondsNum)));
    }
    // derived "seconds" needs no write — its display recomputes from frames/fps.
  };
  const changeSeconds = (v: string) => {
    setSecondsText(v);
    touch("seconds");
    const n = Number(v);
    if (!Number.isFinite(n) || n <= 0) return;
    if (deriveNow() === "fps" && onFps && framesNum != null && framesNum > 0) {
      onFps(Math.max(1, Math.round(framesNum / n)));
    } else if (fps > 0) {
      onFrames(String(Math.max(1, Math.round(n * fps))));
    }
  };
  // fps can also change OUTSIDE this row (the sidebar fps knob shares the same
  // host state). Treat that exactly like typing in the fps cell here: fps is
  // now the freshest value, and if frames is the derived slot it follows
  // fps × seconds so the duration the user chose is preserved.
  const lastFpsRef = useRef(fps);
  useEffect(() => {
    if (fps === lastFpsRef.current) return;
    lastFpsRef.current = fps;
    if (selfFpsRef.current) {
      selfFpsRef.current = false;
      return;
    }
    touch("fps");
    if (deriveNow() === "frames" && secondsNum != null && secondsNum > 0 && fps > 0) {
      onFrames(String(Math.max(1, Math.round(secondsNum * fps))));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fps]);

  return (
    <div className="vi-knob vi-length-row" style={{ flexBasis: "100%" }}>
      <label>length</label>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
        <span className="vi-length-cell">
          <label className="vi-knob-mini-label" htmlFor={`${idPrefix}-fps`}>fps</label>
          <input
            id={`${idPrefix}-fps`}
            className="vi-knob-input"
            style={{ maxWidth: "4.5rem", ...(onFps ? dim("fps") : { opacity: 0.55 }) }}
            type="number"
            min={1}
            disabled={!onFps}
            value={fps}
            title={!onFps ? fpsFixedReason : derived === "fps" ? derivedTitle : undefined}
            onChange={(e) => changeFps(e.target.value)}
          />
        </span>
        <span className="vi-length-cell">
          <label className="vi-knob-mini-label" htmlFor={`${idPrefix}-frames`}>frames</label>
          <input
            id={`${idPrefix}-frames`}
            className="vi-knob-input"
            style={{ maxWidth: "7rem", ...dim("frames") }}
            type="number"
            min={1}
            value={framesDisplay}
            placeholder={framesPlaceholder}
            title={derived === "frames" ? derivedTitle : undefined}
            onChange={(e) => changeFrames(e.target.value)}
          />
        </span>
        <span className="vi-length-cell">
          <label className="vi-knob-mini-label" htmlFor={`${idPrefix}-seconds`}>seconds</label>
          <input
            id={`${idPrefix}-seconds`}
            className="vi-knob-input"
            style={{ maxWidth: "6rem", ...dim("seconds") }}
            type="number"
            min={0.1}
            step={0.1}
            value={secondsDisplay}
            placeholder={fps > 0 ? (defaultFrames / fps).toFixed(1) : ""}
            title={derived === "seconds" ? derivedTitle : undefined}
            onChange={(e) => changeSeconds(e.target.value)}
          />
        </span>
      </div>
      <span className="vi-knob-hint">
        any two set the third — the dimmed one is derived (last touched wins) ·
        server clamps to the model ceiling ({defaultFrames}) and snaps to 4k+1
      </span>
    </div>
  );
}

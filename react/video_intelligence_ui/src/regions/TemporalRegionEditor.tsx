// TemporalRegionEditor — a draggable time-window over an audio waveform, the
// temporal sibling of SpatialRegionEditor and the second concrete implementation
// of the RegionEditor contract (regions/types.ts). Used by the Audio Crop station
// in multiRegion mode: the user shapes a working window (drag the band to move it,
// edge handles to change start/end, preset buttons, numeric fields), scrubs/loops
// it with an <audio> element, adds it to a queue, and the queue is surfaced to the
// station via onChange — exactly like the spatial editor, only the axis differs.
//
// The working window lives in SECONDS at all times; we convert to display pixels
// only for painting/hit-testing, using the timeline scale renderedWidth/duration.
// Every emitted window is clamped to 0 ≤ start < end ≤ duration.
//
// Waveform: the raw audio bytes are fetched through request() (expect:
// "arraybuffer" — the transport's binary path) and decoded with Web Audio's
// AudioContext.decodeAudioData; channel-0 is downsampled to per-column peaks and
// painted to a <canvas>. Decode failure surfaces as a clean inline message and
// never throws (numeric editing + playback still work without the waveform).
//
// FUTURE WORK (stretch, intentionally skipped this phase): a spectrogram toggle
// (FFT per column → magnitude heatmap) over the same canvas. Not built — it must
// not threaten the build; the waveform is the shipping view.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  RegionEditorProps,
  TemporalPreset,
  TemporalRegion,
} from "./types";
import type { MediaRef } from "../video/contract";
import { mediaBytesUrl } from "../config";
import { request, okValue, errorOf, describeAppError } from "../transport/client";

/** One-tap starting windows offered by default. */
export const DEFAULT_TEMPORAL_PRESETS: readonly TemporalPreset[] = [
  { label: "1s", seconds: 1 },
  { label: "5s", seconds: 5 },
  { label: "Full" }, // whole track — no fixed length
] as const;

/** Shortest window we allow (seconds) — keeps start strictly below end. */
const MIN_WINDOW_S = 0.05;
/** Fixed peak resolution computed once per decode; sampled down to canvas width. */
const PEAK_BUCKETS = 1600;
/** Canvas drawing-buffer height (px). CSS height matches for crisp 1:1 painting. */
const CANVAS_H = 128;

interface Size {
  w: number;
  h: number;
}

interface AudioDragState {
  mode: "move" | "start" | "end";
  startClientX: number;
  startWindow: TemporalRegion;
  /** seconds per display px, captured at drag start. */
  secPerPx: number;
}

// ---- pure helpers -------------------------------------------------------------

/** Clamp a window to a valid [0..dur] range with start + MIN_WINDOW_S ≤ end. */
function clampTemporal(r: TemporalRegion, dur: number): TemporalRegion {
  const d = dur > 0 ? dur : 0;
  let start = Math.max(0, Math.min(r.start_s, d));
  let end = Math.max(0, Math.min(r.end_s, d));
  if (end < start) {
    // Tolerate an inverted numeric edit by swapping rather than rejecting.
    const t = start;
    start = end;
    end = t;
  }
  if (end - start < MIN_WINDOW_S) {
    end = Math.min(d, start + MIN_WINDOW_S);
    if (end - start < MIN_WINDOW_S) start = Math.max(0, end - MIN_WINDOW_S);
  }
  return { start_s: start, end_s: end };
}

/** Downsample channel-0 to normalised [0..1] peaks (abs-max per bucket). */
function computePeaks(buf: AudioBuffer, buckets: number): Float32Array {
  const data = buf.getChannelData(0);
  const peaks = new Float32Array(buckets);
  const block = Math.max(1, Math.floor(data.length / buckets));
  for (let i = 0; i < buckets; i++) {
    let max = 0;
    const start = i * block;
    const end = Math.min(data.length, start + block);
    for (let j = start; j < end; j++) {
      const v = Math.abs(data[j]);
      if (v > max) max = v;
    }
    peaks[i] = max;
  }
  let peak = 0;
  for (let i = 0; i < buckets; i++) if (peaks[i] > peak) peak = peaks[i];
  if (peak > 0) for (let i = 0; i < buckets; i++) peaks[i] /= peak;
  return peaks;
}

// ---- rendered-size hook (mirrors SpatialRegionEditor) -------------------------

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

interface TemporalRegionEditorProps
  extends RegionEditorProps<TemporalRegion, TemporalPreset> {
  /** Source audio (kind==="audio"), or null before one is loaded — the editor
      then renders its full chrome over a blank (flat) waveform. */
  source: MediaRef | null;
  /** Total duration in seconds — the timeline scale (0 when no source). */
  duration: number;
}

export function TemporalRegionEditor({
  source,
  duration,
  values,
  onChange,
  presets = DEFAULT_TEMPORAL_PRESETS as TemporalPreset[],
  numericInputs = true,
  multiRegion = true,
}: TemporalRegionEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const rendered = useRenderedSize(canvasRef, source?.uri);

  const [working, setWorking] = useState<TemporalRegion>(() =>
    clampTemporal({ start_s: 0, end_s: Math.min(duration, 5) }, duration),
  );
  const workingRef = useRef<TemporalRegion>(working);
  workingRef.current = working;

  const [drag, setDrag] = useState<AudioDragState | null>(null);
  const [peaks, setPeaks] = useState<Float32Array | null>(null);
  const [decoding, setDecoding] = useState<boolean>(false);
  const [decodeError, setDecodeError] = useState<string | null>(null);

  const [playing, setPlaying] = useState<boolean>(false);
  const [playhead, setPlayhead] = useState<number>(0);

  const queue = values ?? [];

  // display px per second — the timeline scale used for painting/hit-testing.
  const dispScale = rendered.w > 0 && duration > 0 ? rendered.w / duration : 0;

  const setClamped = useCallback(
    (r: TemporalRegion) => setWorking(clampTemporal(r, duration)),
    [duration],
  );

  // Reset the working window (and playback) when the source or duration swaps.
  useEffect(() => {
    setWorking(clampTemporal({ start_s: 0, end_s: Math.min(duration, 5) }, duration));
    setPlaying(false);
    setPlayhead(0);
  }, [source?.uri, duration]);

  // ---- fetch bytes → decode → peaks (never throws) ----
  useEffect(() => {
    // No source yet — leave the waveform flat (baseline only), no fetch.
    if (!source) {
      setDecoding(false);
      setDecodeError(null);
      setPeaks(null);
      return;
    }
    const uri = source.uri;
    let cancelled = false;
    const ctrl = new AbortController();
    setDecoding(true);
    setDecodeError(null);
    setPeaks(null);

    void (async () => {
      const r = await request<ArrayBuffer>(mediaBytesUrl(uri), {
        expect: "arraybuffer",
        signal: ctrl.signal,
        meta: { specKey: "audio-crop", operation: "waveform.fetch" },
      });
      if (cancelled) return;
      if (!r.ok) {
        setDecodeError(describeAppError(errorOf(r)));
        setDecoding(false);
        return;
      }
      const Ctor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!Ctor) {
        setDecodeError("Web Audio is unavailable — waveform preview disabled.");
        setDecoding(false);
        return;
      }
      const ac = new Ctor();
      try {
        const audioBuf = await ac.decodeAudioData(okValue(r));
        if (!cancelled) setPeaks(computePeaks(audioBuf, PEAK_BUCKETS));
      } catch {
        if (!cancelled) setDecodeError("Could not decode this audio for a waveform.");
      } finally {
        void ac.close();
        if (!cancelled) setDecoding(false);
      }
    })();

    return () => {
      cancelled = true;
      ctrl.abort();
    };
  }, [source?.uri]);

  // ---- paint the waveform (+ baseline) to the canvas ----
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = Math.max(1, Math.round(rendered.w));
    const h = CANVAS_H;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    const mid = h / 2;
    // baseline
    ctx.strokeStyle = "#26262a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid + 0.5);
    ctx.lineTo(w, mid + 0.5);
    ctx.stroke();
    if (!peaks || peaks.length === 0) return;
    // waveform bars
    ctx.strokeStyle = "#8ab4ff";
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const idx = Math.min(peaks.length - 1, Math.floor((x / w) * peaks.length));
      const barH = Math.max(1, peaks[idx] * (h * 0.9));
      ctx.moveTo(x + 0.5, mid - barH / 2);
      ctx.lineTo(x + 0.5, mid + barH / 2);
    }
    ctx.stroke();
  }, [peaks, rendered.w]);

  // ---- drag lifecycle: window listeners while a pointer is down ----
  useEffect(() => {
    if (!drag) return;
    const onMove = (e: PointerEvent) => {
      const dSec = (e.clientX - drag.startClientX) * drag.secPerPx;
      if (drag.mode === "move") {
        const len = drag.startWindow.end_s - drag.startWindow.start_s;
        let start = drag.startWindow.start_s + dSec;
        start = Math.max(0, Math.min(start, duration - len));
        setWorking({ start_s: start, end_s: start + len });
        return;
      }
      if (drag.mode === "start") {
        let start = drag.startWindow.start_s + dSec;
        start = Math.max(0, Math.min(start, drag.startWindow.end_s - MIN_WINDOW_S));
        setWorking({ start_s: start, end_s: drag.startWindow.end_s });
        return;
      }
      let end = drag.startWindow.end_s + dSec;
      end = Math.min(duration, Math.max(end, drag.startWindow.start_s + MIN_WINDOW_S));
      setWorking({ start_s: drag.startWindow.start_s, end_s: end });
    };
    const onUp = () => setDrag(null);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [drag, duration]);

  const startDrag = (mode: AudioDragState["mode"]) => (e: React.PointerEvent) => {
    if (dispScale <= 0) return;
    e.preventDefault();
    if (mode !== "move") e.stopPropagation();
    setDrag({
      mode,
      startClientX: e.clientX,
      startWindow: working,
      secPerPx: 1 / dispScale,
    });
  };

  // ---- region-loop playback (single <audio> element) ----
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const loop = () => {
      const a = audioRef.current;
      if (a) {
        const win = workingRef.current;
        if (a.currentTime >= win.end_s) a.currentTime = win.start_s;
        setPlayhead(a.currentTime);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const togglePlay = () => {
    const a = audioRef.current;
    if (!a) return;
    if (playing) {
      a.pause();
      setPlaying(false);
      setPlayhead(working.start_s);
      return;
    }
    try {
      a.currentTime = working.start_s;
    } catch {
      /* currentTime may not be settable until metadata loads — play anyway */
    }
    void a
      .play()
      .then(() => setPlaying(true))
      .catch(() => setPlaying(false));
  };

  // ---- numeric field editing (seconds) ----
  const setField =
    (key: keyof TemporalRegion) => (e: React.ChangeEvent<HTMLInputElement>) => {
      const raw = Number(e.target.value);
      const v = Number.isFinite(raw) ? raw : 0;
      setClamped({ ...working, [key]: v });
    };

  // ---- queue mutations (surface via onChange) ----
  const addToQueue = () => {
    const region = clampTemporal(working, duration);
    onChange(multiRegion ? [...queue, region] : region);
  };
  const removeAt = (i: number) => onChange(queue.filter((_, idx) => idx !== i));
  const applyPreset = (p: TemporalPreset) => {
    if (p.seconds == null) {
      setClamped({ start_s: 0, end_s: duration });
      return;
    }
    setClamped({ start_s: working.start_s, end_s: working.start_s + p.seconds });
  };

  // display geometry for the region band + playhead
  const regionStyle = useMemo(
    () => ({
      left: `${working.start_s * dispScale}px`,
      width: `${(working.end_s - working.start_s) * dispScale}px`,
    }),
    [working, dispScale],
  );
  const playheadStyle = useMemo(
    () => ({ left: `${playhead * dispScale}px` }),
    [playhead, dispScale],
  );

  const windowLen = Math.max(0, working.end_s - working.start_s);

  return (
    <div className="vi-crop-editor">
      <div className="vi-audio-stage">
        <canvas ref={canvasRef} className="vi-audio-canvas" />
        <div className="vi-audio-overlay">
          <div
            className={
              drag?.mode === "move" ? "vi-audio-region vi-audio-region-active" : "vi-audio-region"
            }
            style={regionStyle}
            onPointerDown={startDrag("move")}
          >
            <span
              className="vi-audio-handle vi-audio-handle-start"
              onPointerDown={startDrag("start")}
            />
            <span
              className="vi-audio-handle vi-audio-handle-end"
              onPointerDown={startDrag("end")}
            />
          </div>
          {playing && <div className="vi-audio-playhead" style={playheadStyle} />}
        </div>
        {decoding && <span className="vi-audio-note">Decoding waveform…</span>}
        {decodeError && !decoding && (
          <span className="vi-audio-note vi-audio-note-warn">{decodeError}</span>
        )}
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
          <button
            type="button"
            className={playing ? "vi-btn vi-btn-accent" : "vi-btn"}
            onClick={togglePlay}
          >
            {playing ? "■ Stop" : "▶ Play region"}
          </button>
        </div>

        {numericInputs && (
          <div className="vi-crop-fields">
            {(["start_s", "end_s"] as const).map((k) => (
              <label key={k} className="vi-crop-field">
                <span>{k}</span>
                <input
                  type="number"
                  min={0}
                  max={duration}
                  step="any"
                  value={working[k]}
                  onChange={setField(k)}
                />
              </label>
            ))}
            <span className="vi-crop-native">
              {windowLen.toFixed(2)}s · of {duration.toFixed(2)}s
            </span>
          </div>
        )}

        <div className="vi-crop-actions">
          <button type="button" className="vi-btn vi-btn-accent" onClick={addToQueue}>
            {multiRegion ? "Add clip to queue" : "Set clip"}
          </button>
        </div>

        {multiRegion && queue.length > 0 && (
          <ul className="vi-region-list">
            {queue.map((r, i) => (
              <li key={i} className="vi-region-item">
                <span className="vi-region-idx">#{i + 1}</span>
                <code className="vi-region-dims">
                  {r.start_s.toFixed(2)}–{r.end_s.toFixed(2)}s
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

      {/* Scrub + region-loop playback source. Direct media URL (like <img> in the
          spatial editor) — not a fetch; playback works without HTTP Range. */}
      <audio
        ref={audioRef}
        src={source ? mediaBytesUrl(source.uri) : undefined}
        preload="metadata"
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
    </div>
  );
}

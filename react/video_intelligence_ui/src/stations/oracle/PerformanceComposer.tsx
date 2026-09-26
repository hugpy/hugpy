import { useEffect, useState } from "react";

import { hugpyConfig } from "../../config";
import { describeAppError, errorOf, okValue, request } from "../../transport/client";

type EnqueueResult = { job_id?: string; error?: string };
type PerformanceProbe = { ready: boolean; bound?: string[]; unbound?: Array<{ seam: string; requirement: string }> };
type Stage = "authority" | "snapshot" | "audio" | "lock" | "segments" | "keyframes" | "clips" | "assembly";

/** The small, explicit entry point for the Oracle's durable performance job. */
export function PerformanceComposer({ onEnqueued }: { onEnqueued: (jobId: string) => void }) {
  const [prompt, setPrompt] = useState("");
  const [dialogue, setDialogue] = useState("");
  const [speaker, setSpeaker] = useState("narrator");
  const [voiceId, setVoiceId] = useState("narrator");
  const [rawRef, setRawRef] = useState("");
  const [quality, setQuality] = useState("balanced");
  const [stopAfter, setStopAfter] = useState<Stage | "">("segments");
  const [tts, setTts] = useState(3);
  const [keyframes, setKeyframes] = useState(3);
  const [clips, setClips] = useState(3);
  const [maxSeconds, setMaxSeconds] = useState("");
  const [seed, setSeed] = useState(0);
  const [negative, setNegative] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [probe, setProbe] = useState<PerformanceProbe | null>(null);
  const [probeGap, setProbeGap] = useState("");

  useEffect(() => {
    let active = true;
    void (async () => {
      const result = await request<PerformanceProbe>(`${hugpyConfig.apiBase}/video/performance/probe`, {
        meta: { specKey: "oracle", operation: "oracle.performance.probe" },
      });
      if (!active) return;
      if (result.ok) setProbe(okValue(result));
      else setProbeGap(describeAppError(errorOf(result)));
    })();
    return () => { active = false; };
  }, []);

  async function enqueue() {
    const lines = dialogue.split("\n").map((text) => text.trim()).filter(Boolean);
    if (!prompt.trim() || !lines.length || !speaker.trim() || !voiceId.trim() || !rawRef.trim()) {
      setMessage("Add the goal, at least one dialogue line, speaker, voice ID, and raw request reference.");
      return;
    }
    const body = {
      goal: { objective: prompt.trim(), raw_prompt: prompt, quality, planner_mode: "local_only" },
      dialogue: { locked: true, lines: lines.map((text, i) => ({ line_id: `line-${i + 1}`, speaker: speaker.trim(), text })) },
      casting: [[speaker.trim(), { voice_id: voiceId.trim(), kind: "synthetic" }]],
      raw_request_ref: rawRef.trim(),
      tts_candidates: tts,
      keyframe_candidates: keyframes,
      clip_candidates: clips,
      seed_salt: seed,
      negative_prompt: negative.trim() || null,
      max_seconds: maxSeconds ? Number(maxSeconds) : null,
      stop_after: stopAfter || null,
    };
    setBusy(true);
    setMessage("");
    const result = await request<EnqueueResult>(`${hugpyConfig.apiBase}/video/jobs/performance`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      meta: { specKey: "oracle", operation: "oracle.performance.enqueue" },
    });
    setBusy(false);
    if (!result.ok) {
      setMessage(describeAppError(errorOf(result)));
      return;
    }
    const jobId = okValue(result).job_id;
    if (!jobId) {
      setMessage("The server answered without a job id.");
      return;
    }
    setMessage(`Queued ${jobId}`);
    onEnqueued(jobId);
  }

  return <section className="vi-oracle-panel vi-oracle-composer" aria-label="New performance">
    <h3>New performance <span className="vi-oracle-sub">audio first · durable media job · inspect each stage below</span></h3>
    <div className="vi-oracle-fields">
      <label className="vi-oracle-field vi-oracle-wide">Goal / visual direction
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} placeholder="Describe the film, characters, setting and action" />
      </label>
      <label className="vi-oracle-field vi-oracle-wide">Locked dialogue · one line per utterance
        <textarea value={dialogue} onChange={(e) => setDialogue(e.target.value)} rows={3} placeholder="The words to speak" />
      </label>
      <label className="vi-oracle-field">Speaker <input value={speaker} onChange={(e) => setSpeaker(e.target.value)} /></label>
      <label className="vi-oracle-field">Synthetic voice ID <input value={voiceId} onChange={(e) => setVoiceId(e.target.value)} /></label>
      <label className="vi-oracle-field vi-oracle-wide">Raw request reference
        <input value={rawRef} onChange={(e) => setRawRef(e.target.value)} placeholder="A durable reference to the original request" />
      </label>
      <label className="vi-oracle-field">Quality
        <select value={quality} onChange={(e) => setQuality(e.target.value)}><option value="preview">Preview</option><option value="balanced">Balanced</option><option value="best">Best</option></select>
      </label>
      <label className="vi-oracle-field">Stop after
        <select value={stopAfter} onChange={(e) => setStopAfter(e.target.value as Stage | "")}>
          <option value="">Full run</option>{(["authority", "snapshot", "audio", "lock", "segments", "keyframes", "clips", "assembly"] as Stage[]).map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </label>
      <label className="vi-oracle-field">TTS candidates <input type="number" min={1} max={16} value={tts} onChange={(e) => setTts(Number(e.target.value))} /></label>
      <label className="vi-oracle-field">Keyframe candidates <input type="number" min={1} max={16} value={keyframes} onChange={(e) => setKeyframes(Number(e.target.value))} /></label>
      <label className="vi-oracle-field">Clip candidates <input type="number" min={1} max={16} value={clips} onChange={(e) => setClips(Number(e.target.value))} /></label>
      <label className="vi-oracle-field">Max seconds <input type="number" min={1} value={maxSeconds} onChange={(e) => setMaxSeconds(e.target.value)} placeholder="No limit" /></label>
      <label className="vi-oracle-field">Seed salt <input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></label>
      <label className="vi-oracle-field vi-oracle-wide">Negative prompt <input value={negative} onChange={(e) => setNegative(e.target.value)} placeholder="Optional exclusions" /></label>
    </div>
    <div className="vi-oracle-note" role="status">
      {probe ? <>
        <strong>{probe.ready ? "Full recipe wired" : "Staged runs available; full recipe has gaps"}</strong>
        {probe.unbound?.length ? <ul className="vi-oracle-limits">{probe.unbound.map((gap) => <li key={gap.seam}><code>{gap.seam}</code>: {gap.requirement}</li>)}</ul> : null}
      </> : probeGap ? <>Could not read performance capability probe: {probeGap}</> : "Checking performance capabilities…"}
    </div>
    <div className="vi-oracle-row"><button type="button" className="vi-btn vi-btn-accent" disabled={busy} onClick={() => void enqueue()}>{busy ? "Queuing…" : "Queue performance"}</button>{message && <span role="status">{message}</span>}</div>
  </section>;
}

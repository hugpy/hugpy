/**
 * k114 — the Script station: the script-first pipeline, made visible.
 *
 * The doc's "Interface behavior" list, one panel each:
 *
 *  1. Run header — id, start, registry version, and the selected model ROUTE
 *     with the router's own justification for it.
 *  2. Snapshot — which prompts were captured with their hashes, and which were
 *     EXCLUDED for being persisted after the run began, with the reason. An
 *     exclusion is shown, never inferred from an absence.
 *  3. Pre-production — plot and screenplay, authored or hand-edited as JSON,
 *     with the validator's errors rendered VERBATIM. After the lock the
 *     editors go read-only and the only control is revise-with-a-reason.
 *  4. Continuity + shot plan — read-only tables.
 *  5. Segments — one parent box (the locked artifacts) fanning out to N cards.
 *     Each card carries its prompt, its provenance digests, every attempt with
 *     the model/seed/params it actually used, and its own Regenerate button
 *     that busies ONLY that card. The sibling digests are printed on the card
 *     so "regenerating 1 did not change 2" is readable rather than promised.
 *  6. Promote — with a confirmation that says what it actually does: start a
 *     NEW run's source, and never this one's.
 *
 * Every gap is rendered as a gap. Nothing on this screen is a spinner standing
 * in for a capability this fleet does not have.
 */
import { useEffect, useMemo, useState } from "react";

import type { StationSpec } from "./types";
import {
  useScriptFirstRun,
  type AttemptRow,
  type RunState,
  type SegmentRow,
} from "./useScriptFirstRun";
import { TemplateStepsPanel } from "./TemplateStepsPanel";
import "./scriptFirst.css";

const SHORT = 12;

function short(digest: unknown): string {
  const text = String(digest ?? "");
  return text ? `${text.slice(0, SHORT)}…` : "—";
}

function pretty(value: unknown): string {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return String(value ?? "");
  }
}

function asRecord(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
}

// ---------------------------------------------------------------------------
// Panel 1 — the run header
// ---------------------------------------------------------------------------

function RunHeader({ run }: { run: RunState }) {
  const route = run.models?.authoring_route ?? {};
  const reasons = route.reasons ?? [];
  return (
    <section className="vi-script-panel">
      <h3>
        Run
        <span className="vi-script-sub">{run.deliverable}</span>
      </h3>
      <dl className="vi-script-kv">
        <dt>run id</dt>
        <dd>
          <code>{run.run_id}</code>{" "}
          <span
            className={`vi-script-tag ${
              run.locked ? "vi-script-tag-on" : ""
            }`}
          >
            {run.locked
              ? `locked · revision ${
                  asRecord(run.lock?.payload).revision ?? 0
                }`
              : "open"}
          </span>
        </dd>
        <dt>started</dt>
        <dd>{run.created_at}</dd>
        <dt>registry_version</dt>
        <dd className="vi-script-digest">
          {run.models?.fleet?.registry_version ?? "— (the registry is unversioned)"}
        </dd>
        <dt>model route</dt>
        <dd>
          {route.model_id ? (
            <>
              <code>{route.model_id}</code> for <code>{route.capability}</code>{" "}
              <span className="vi-script-tag">{route.execution}</span>
            </>
          ) : (
            <span className="vi-script-tag vi-script-tag-off">
              no model route for {route.capability ?? "text.chat"}
            </span>
          )}
        </dd>
        <dt>why</dt>
        <dd>
          {route.model_rationale || "—"}
          {reasons.length > 0 && (
            <ul className="vi-script-limits">
              {reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
          <p className="vi-script-note">
            The routing justification available today is the router's own
            rationale. A measured per-task benchmark (k109) has not landed, so
            nothing here claims one.
          </p>
        </dd>
        <dt>snapshot</dt>
        <dd className="vi-script-digest">{run.snapshot_digest}</dd>
      </dl>
      {run.limitations?.length > 0 && (
        <ul className="vi-script-limits">
          {run.limitations.map((l, i) => (
            <li key={i}>{l}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Panel 2 — the immutable snapshot
// ---------------------------------------------------------------------------

function SnapshotPanel({ run }: { run: RunState }) {
  const excluded = run.sources.filter((s) => !s.included);
  return (
    <section className="vi-script-panel">
      <h3>
        Input snapshot
        <span className="vi-script-sub">
          captured before generation began — only these may influence this run
        </span>
      </h3>
      <table className="vi-script-table">
        <thead>
          <tr>
            <th>prompt id</th>
            <th>text</th>
            <th>content hash</th>
            <th>persisted</th>
            <th>in snapshot</th>
          </tr>
        </thead>
        <tbody>
          {run.sources.map((s) => (
            <tr
              key={s.prompt_id}
              className={s.included ? "" : "vi-script-excluded"}
            >
              <td>
                <code>{s.prompt_id}</code>
                {s.origin === "promoted" && (
                  <>
                    {" "}
                    <span className="vi-script-tag">promoted</span>
                  </>
                )}
              </td>
              <td>{s.text}</td>
              <td className="vi-script-digest">{short(s.digest)}</td>
              <td>{s.persisted_at ?? "—"}</td>
              <td>
                {s.included ? (
                  <span className="vi-script-tag vi-script-tag-on">captured</span>
                ) : (
                  <span className="vi-script-tag vi-script-tag-off">excluded</span>
                )}
                {s.exclusion_reason && (
                  <p className="vi-script-note">{s.exclusion_reason}</p>
                )}
              </td>
            </tr>
          ))}
          {run.sources.length === 0 && (
            <tr>
              <td colSpan={5}>
                This run captured no source prompts. Doc Stage 5's "minimal"
                mode builds a plot from the requirements alone.
              </td>
            </tr>
          )}
        </tbody>
      </table>
      {excluded.length > 0 && (
        <p className="vi-script-note">
          {excluded.length} prompt(s) were persisted after this run started and
          are shown struck through. They are excluded from the snapshot, not
          hidden — a prompt created during a run cannot become an input to a
          segment of that run.
        </p>
      )}
      <p className="vi-script-note">
        {run.ledger.length} prompt(s) have been minted inside this run and are
        recorded in its ledger. Any of them is refused re-entry into this
        snapshot by digest.
      </p>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Panel 3 — the artifact editor
// ---------------------------------------------------------------------------

function ArtifactEditor({
  stage,
  title,
  run,
  busy,
  canAuthor,
  onAuthor,
  onSave,
}: {
  stage: string;
  title: string;
  run: RunState;
  busy: string | null;
  canAuthor: boolean;
  onAuthor?: () => void;
  onSave(json: unknown): void;
}) {
  const entry = run.artifacts?.[stage];
  const [draft, setDraft] = useState<string>("");
  const [parseError, setParseError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setDraft(entry?.payload ? pretty(entry.payload) : "");
    setParseError(null);
  }, [entry?.digest, stage]);

  const locked = run.locked;
  const gap = entry?.gap ?? null;

  function save() {
    let parsed: unknown;
    try {
      parsed = JSON.parse(draft);
    } catch (e) {
      setParseError(
        `That is not JSON: ${e instanceof Error ? e.message : String(e)}`,
      );
      return;
    }
    setParseError(null);
    onSave(parsed);
  }

  return (
    <section className="vi-script-panel">
      <h3>
        {title}
        <span className="vi-script-sub">
          {entry?.digest ? (
            <>
              {entry.provenance} · <code>{short(entry.digest)}</code> ·{" "}
              {entry.at}
            </>
          ) : gap ? (
            "last attempt ended in a gap"
          ) : (
            "not built yet"
          )}
        </span>
      </h3>

      <div className="vi-script-row">
        {onAuthor && (
          <button
            type="button"
            className="vi-btn vi-btn-accent"
            disabled={locked || !canAuthor || busy != null}
            onClick={onAuthor}
          >
            {busy === `${stage}.author` ? "Authoring…" : `Author ${stage}`}
          </button>
        )}
        <button
          type="button"
          className="vi-btn"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? "Hide JSON" : locked ? "View JSON (read-only)" : "Edit JSON"}
        </button>
        {locked && (
          <span className="vi-script-tag vi-script-tag-on">
            locked — edits go through a revision
          </span>
        )}
      </div>

      {gap && (
        <div className="vi-script-gap">
          <strong>{gap.code}</strong>
          {gap.errors.map((e, i) => (
            <div key={i}>{e}</div>
          ))}
          {gap.raw && (
            <>
              <p className="vi-script-note">
                The model's raw reply is kept — nothing was coerced from it.
              </p>
              <pre className="vi-script-raw">{gap.raw}</pre>
            </>
          )}
        </div>
      )}

      {open && (
        <>
          <textarea
            className="vi-script-json"
            value={draft}
            readOnly={locked}
            disabled={busy != null}
            spellCheck={false}
            onChange={(e) => setDraft(e.target.value)}
            aria-label={`${title} JSON`}
          />
          {parseError && (
            <p className="vi-error" role="alert">
              {parseError}
            </p>
          )}
          {!locked && (
            <div className="vi-script-row">
              <button
                type="button"
                className="vi-btn vi-btn-accent"
                disabled={busy != null || !draft.trim()}
                onClick={save}
              >
                {busy === `${stage}.put` ? "Validating…" : `Save ${stage}`}
              </button>
              <span className="vi-script-note">
                Validated through the same constructor the model's reply goes
                through. There is no lenient path.
              </span>
            </div>
          )}
        </>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Panel 4 — continuity + shot plan viewers
// ---------------------------------------------------------------------------

function ContinuityViewer({ run }: { run: RunState }) {
  const entry = run.artifacts?.continuity;
  const payload = asRecord(entry?.payload);
  const entries = (payload.entries as Array<Record<string, unknown>>) ?? [];
  if (!entry) return null;
  return (
    <section className="vi-script-panel">
      <h3>
        Continuity bible
        <span className="vi-script-sub">
          derived from the screenplay, never authored ·{" "}
          <code>{short(entry.digest)}</code>
        </span>
      </h3>
      <table className="vi-script-table">
        <thead>
          <tr>
            <th>segment</th>
            <th>state_before</th>
            <th>state_after</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => (
            <tr key={String(e.segment_id ?? i)}>
              <td>
                <code>{String(e.segment_id ?? "")}</code>
              </td>
              <td>
                <pre className="vi-script-seg-prompt">
                  {pretty(e.state_before)}
                </pre>
              </td>
              <td>
                <pre className="vi-script-seg-prompt">
                  {pretty(e.state_after)}
                </pre>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function ShotPlanViewer({ run }: { run: RunState }) {
  const entry = run.artifacts?.shot_plan;
  const payload = asRecord(entry?.payload);
  const designs = (payload.designs as Array<Record<string, unknown>>) ?? [];
  if (!entry) return null;
  const audioFirst = Boolean(payload.audio_first);
  return (
    <section className="vi-script-panel">
      <h3>
        Shot plan
        <span className="vi-script-sub">
          <code>{short(entry.digest)}</code> ·{" "}
          {audioFirst
            ? "cut to the locked audio"
            : "every window is an ESTIMATE (no audio master yet)"}
        </span>
      </h3>
      <table className="vi-script-table">
        <thead>
          <tr>
            <th>segment</th>
            <th>scene</th>
            <th>window</th>
            <th>camera</th>
            <th>block / light</th>
          </tr>
        </thead>
        <tbody>
          {designs.map((d, i) => (
            <tr key={String(d.segment_id ?? i)}>
              <td>
                <code>{String(d.segment_id ?? "")}</code>
                {d.estimated ? (
                  <>
                    {" "}
                    <span className="vi-script-tag vi-script-tag-off">
                      estimated
                    </span>
                  </>
                ) : null}
              </td>
              <td>{String(d.scene_id ?? "")}</td>
              <td>
                {String(d.start_s ?? 0)}s – {String(d.end_s ?? 0)}s
              </td>
              <td className="vi-script-digest">{pretty(d.camera)}</td>
              <td>
                <div>{String(d.blocking ?? "")}</div>
                <div className="vi-script-note">{String(d.lighting ?? "")}</div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Panel 5 — the sibling fan-out
// ---------------------------------------------------------------------------

/**
 * `key` is declared on the props of the two components below because this
 * project has no `@types/react` installed (see package.json — React's types are
 * absent, not merely loose), so TS has no `JSX.IntrinsicAttributes` carrying
 * `key` and rejects it as an excess property on a locally-typed component.
 * Every other station only puts `key` on intrinsic elements or on react-router
 * components that ship their own types, which is why nothing else hits this.
 * Declaring it is the one-line fix that keeps `npx tsc --noEmit` honest without
 * adding a dependency or restructuring the markup around a limitation.
 */
function AttemptLine({ attempt }: { key?: string | number; attempt: AttemptRow }) {
  const cls = attempt.gap
    ? "vi-script-attempt-gap"
    : attempt.ok
      ? "vi-script-attempt-ok"
      : "";
  return (
    <li className={cls}>
      #{attempt.attempt} · {attempt.kind} · seed {attempt.seed} ·{" "}
      {attempt.model_id ?? "no model"} ·{" "}
      {Object.entries(attempt.params ?? {})
        .map(([k, v]) => `${k}=${String(v)}`)
        .join(" ") || "no params"}
      {attempt.gap && (
        <div className="vi-script-gap">
          <strong>
            {attempt.gap.code}
            {attempt.gap.capability ? ` · ${attempt.gap.capability}` : ""}
          </strong>
          {(attempt.gap.reasons ?? []).map((r, i) => (
            <div key={i}>{r}</div>
          ))}
          {attempt.gap.requirement && (
            <p className="vi-script-note">{attempt.gap.requirement}</p>
          )}
        </div>
      )}
      {attempt.artifacts?.length > 0 && (
        <div className="vi-script-digest">
          {attempt.artifacts
            .map((a) => String(asRecord(a).uri ?? ""))
            .filter(Boolean)
            .join(" ")}
        </div>
      )}
    </li>
  );
}

function SegmentCard({
  row,
  attempts,
  busy,
  onRegenerate,
  onPromote,
}: {
  key?: string | number;
  row: SegmentRow;
  attempts: AttemptRow[];
  busy: boolean;
  onRegenerate(kind: string): void;
  onPromote(): void;
}) {
  const [showParents, setShowParents] = useState(false);
  const last = attempts.length > 0 ? attempts[attempts.length - 1] : null;
  const siblings = last?.siblings_after ?? {};
  return (
    <article className={`vi-script-seg${busy ? " vi-script-seg-busy" : ""}`}>
      <header className="vi-script-seg-head">
        <span>
          <code>{row.segment_id}</code>{" "}
          <span className="vi-script-tag">#{row.index}</span>
        </span>
        <span className="vi-script-tag">seed {row.seed_base}</span>
      </header>
      <p className="vi-script-seg-prompt">{row.prompt}</p>
      <div className="vi-script-digest">
        spec {short(row.digest)} · lock {short(row.lock_digest)} · joint{" "}
        {row.joint_mode} · window {row.window[0]}s–{row.window[1]}s
      </div>
      <div className="vi-script-row">
        <button
          type="button"
          className="vi-btn vi-btn-sm"
          onClick={() => setShowParents((v) => !v)}
        >
          {showParents ? "Hide" : "Show"} provenance ({row.parents.length})
        </button>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-accent"
          disabled={busy}
          onClick={() => onRegenerate("keyframe")}
        >
          {busy ? "Regenerating…" : "Regenerate keyframe"}
        </button>
        <button
          type="button"
          className="vi-btn vi-btn-sm"
          disabled={busy}
          onClick={() => onRegenerate("clip")}
        >
          Render clip
        </button>
      </div>
      {showParents && (
        <div className="vi-script-digest">
          {row.parents.map((p) => (
            <div key={p}>{p}</div>
          ))}
          <p className="vi-script-note">
            Every parent is a locked artifact. No sibling segment appears here,
            and there is no field on a segment spec that one could appear in.
          </p>
        </div>
      )}
      {attempts.length > 0 ? (
        <ul className="vi-script-attempts">
          {attempts.map((a) => (
            <AttemptLine key={a.attempt} attempt={a} />
          ))}
        </ul>
      ) : (
        <p className="vi-script-note">No attempt yet.</p>
      )}
      {last && (
        <p className="vi-script-note">
          siblings at the last attempt:{" "}
          {Object.keys(siblings).length === 0
            ? "none"
            : Object.entries(siblings)
                .map(([id, d]) => `${id} ${short(d)}`)
                .join(" · ")}{" "}
          — {last.siblings_unchanged ? "unchanged" : "CHANGED"}
        </p>
      )}
      <div className="vi-script-row">
        <button
          type="button"
          className="vi-btn vi-btn-sm"
          disabled={busy}
          onClick={onPromote}
        >
          Promote…
        </button>
      </div>
    </article>
  );
}

function SegmentsPanel({
  run,
  segmentBusy,
  onCompile,
  onRegenerate,
  onPromote,
  busy,
}: {
  run: RunState;
  segmentBusy: string | null;
  busy: string | null;
  onCompile(): void;
  onRegenerate(id: string, kind: string): void;
  onPromote(id: string): void;
}) {
  const segments = run.segments;
  return (
    <section className="vi-script-panel">
      <h3>
        Segments
        <span className="vi-script-sub">
          siblings of one locked production — never a chain
        </span>
      </h3>
      <div className="vi-script-row">
        <button
          type="button"
          className="vi-btn vi-btn-accent"
          disabled={!run.locked || busy != null}
          onClick={onCompile}
        >
          {busy === "segments.compile"
            ? "Compiling…"
            : segments
              ? "Recompile from the lock"
              : "Compile segments"}
        </button>
        {!run.locked && (
          <span className="vi-script-note">
            Segments are compiled from the production lock. Lock the production
            first.
          </span>
        )}
      </div>

      {segments && (
        <>
          <div className="vi-script-fanout">
            <div className="vi-script-parent">
              <h4>Locked artifacts</h4>
              <div className="vi-script-digest">
                production_lock {short(segments.lock_digest)} · revision{" "}
                {segments.revision} · {segments.parent_digests.length} parent
                digests
              </div>
              <p className="vi-script-note">{segments.sibling_shape.note}</p>
            </div>
            <div className="vi-script-stem" />
            <div className="vi-script-children">
              {segments.specs.map((row) => (
                <SegmentCard
                  key={row.segment_id}
                  row={row}
                  attempts={run.attempts?.[row.segment_id] ?? []}
                  busy={segmentBusy === row.segment_id}
                  onRegenerate={(kind) => onRegenerate(row.segment_id, kind)}
                  onPromote={() => onPromote(row.segment_id)}
                />
              ))}
            </div>
          </div>

          <h3 style={{ marginTop: "1rem" }}>
            Plan graph
            <span className="vi-script-sub">
              {segments.graph.nodes.length} nodes ·{" "}
              {segments.validation.ok === true
                ? "validator: clean"
                : segments.validation.ok === false
                  ? "validator: reported findings"
                  : "validator: could not run"}
            </span>
          </h3>
          {(segments.validation.errors ?? []).length > 0 && (
            <ul className="vi-script-limits">
              {(segments.validation.errors ?? []).map((e, i) => (
                <li key={i}>
                  <code>{String(asRecord(e).code ?? "")}</code>{" "}
                  {String(asRecord(e).message ?? "")}{" "}
                  {asRecord(e).node_id ? (
                    <em>({String(asRecord(e).node_id)})</em>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
          {segments.validation.note && (
            <p className="vi-script-note">{segments.validation.note}</p>
          )}
          <p className="vi-script-note">
            sequential batches: {segments.execution_order.sequential.length} ·
            parallel batches: {segments.execution_order.parallel.length} — the
            same graph, batched differently.
          </p>
        </>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// The station
// ---------------------------------------------------------------------------

export function ScriptFirstStation({ spec }: { spec: StationSpec }) {
  const api = useScriptFirstRun();
  const { run } = api;

  const [deliverable, setDeliverable] = useState("");
  const [requirements, setRequirements] = useState("");
  const [sourceText, setSourceText] = useState("");
  const [reason, setReason] = useState("");
  const [promoting, setPromoting] = useState<string | null>(null);
  const [promoteNote, setPromoteNote] = useState("");
  const [audioDraft, setAudioDraft] = useState("");
  const [showAudio, setShowAudio] = useState(false);

  const lastPromotion = useMemo(
    () =>
      run && run.promotions.length > 0
        ? run.promotions[run.promotions.length - 1]
        : null,
    [run],
  );

  const journalled = run?.last_refusal ?? null;
  const refusal = api.refusal;

  return (
    <section className="station-card station-wide vi-script" aria-label={spec.title}>
      <p className="station-phase">{spec.phase}</p>
      <p className="station-blurb">{spec.blurb}</p>

      {/* ---- run picker + create ---- */}
      <section className="vi-script-panel">
        <h3>
          Runs
          <span className="vi-script-sub">
            a run is immutable once started — a new source means a new run
          </span>
        </h3>
        <div className="vi-script-runs">
          {api.runs.map((r) => (
            <button
              key={r.run_id}
              type="button"
              className={`vi-btn vi-btn-sm vi-script-runbtn${
                api.runId === r.run_id ? " vi-btn-on" : ""
              }`}
              onClick={() => api.select(r.run_id)}
            >
              <code>{r.run_id}</code> · {r.locked ? "locked" : "open"} ·{" "}
              {r.segments} seg · {r.attempts} attempts
            </button>
          ))}
          {api.runs.length === 0 && (
            <span className="vi-script-note">No runs on this box yet.</span>
          )}
        </div>
        <div className="vi-script-row">
          <input
            className="vi-knob-input vi-script-grow"
            placeholder="Deliverable (what is being asked for)"
            value={deliverable}
            onChange={(e) => setDeliverable(e.target.value)}
          />
          <input
            className="vi-knob-input vi-script-grow"
            placeholder="Requirements"
            value={requirements}
            onChange={(e) => setRequirements(e.target.value)}
          />
        </div>
        <div className="vi-script-row">
          <input
            className="vi-knob-input vi-script-grow"
            placeholder="An existing prompt to capture (optional)"
            value={sourceText}
            onChange={(e) => setSourceText(e.target.value)}
          />
          <button
            type="button"
            className="vi-btn vi-btn-accent"
            disabled={!deliverable.trim() || api.busy != null}
            onClick={() => {
              void api
                .createRun({
                  deliverable,
                  requirements,
                  sources: sourceText.trim()
                    ? [{ prompt_id: "operator-1", text: sourceText }]
                    : [],
                })
                .then((id) => {
                  if (id) api.select(id);
                });
            }}
          >
            {api.busy === "runs.create" ? "Starting…" : "Start a run"}
          </button>
        </div>
        {api.sources.length > 0 && (
          <p className="vi-script-note">
            {api.sources.length} promoted source(s) are available to a NEW run:{" "}
            {api.sources.map((s) => s.source_id).join(", ")}
          </p>
        )}
      </section>

      {/* ---- registry template: the ordered steps + per-step model choice ---- */}
      <TemplateStepsPanel templateId="video-script-first" />

      {(refusal || journalled) && (
        <section className="vi-script-panel" role="alert">
          <h3>
            Refused
            <span className="vi-script-sub">
              {(refusal ?? journalled)?.code}
            </span>
          </h3>
          <p className="vi-error">{(refusal ?? journalled)?.message}</p>
          <pre className="vi-script-pre">
            {((refusal ?? journalled)?.errors ?? []).join("\n")}
          </pre>
          {asRecord(journalled?.detail).requirement && (
            <p className="vi-script-note">
              {String(asRecord(journalled?.detail).requirement)}
            </p>
          )}
          <button type="button" className="vi-btn vi-btn-sm" onClick={api.clearRefusal}>
            Dismiss
          </button>
        </section>
      )}

      {api.loading && <p className="vi-script-note">Loading run…</p>}

      {run && (
        <>
          <RunHeader run={run} />
          <SnapshotPanel run={run} />

          <ArtifactEditor
            stage="plot"
            title="Plot"
            run={run}
            busy={api.busy}
            canAuthor
            onAuthor={() => api.authorPlot(requirements)}
            onSave={api.putPlot}
          />
          <ArtifactEditor
            stage="screenplay"
            title="Screenplay"
            run={run}
            busy={api.busy}
            canAuthor={Boolean(run.artifacts?.plot?.digest)}
            onAuthor={api.authorScreenplay}
            onSave={api.putScreenplay}
          />

          <section className="vi-script-panel">
            <h3>
              Lock
              <span className="vi-script-sub">
                doc Stage 11 — version and lock the screenplay, the continuity
                bible, the audio master and the shot plan together
              </span>
            </h3>
            <div className="vi-script-row">
              <button
                type="button"
                className="vi-btn"
                disabled={run.locked || api.busy != null}
                onClick={api.buildPreproduction}
              >
                {api.busy === "preproduction"
                  ? "Deriving…"
                  : "Derive continuity + shot plan"}
              </button>
              <button
                type="button"
                className="vi-btn vi-btn-accent"
                disabled={run.locked || api.busy != null}
                onClick={() => api.lockRun()}
              >
                {api.busy === "lock" ? "Locking…" : "Lock production"}
              </button>
              <button
                type="button"
                className="vi-btn vi-btn-sm"
                onClick={() => setShowAudio((v) => !v)}
              >
                {showAudio ? "Hide" : "Supply"} audio master JSON
              </button>
            </div>
            {showAudio && (
              <>
                <textarea
                  className="vi-script-json"
                  value={audioDraft}
                  spellCheck={false}
                  placeholder='{"timeline_digest": "…", "line_timings": [], "tracks": [], "total_seconds": 0, "locked": true}'
                  onChange={(e) => setAudioDraft(e.target.value)}
                  aria-label="Audio master JSON"
                />
                <div className="vi-script-row">
                  <button
                    type="button"
                    className="vi-btn"
                    disabled={api.busy != null || !audioDraft.trim()}
                    onClick={() => {
                      try {
                        api.putAudioMaster(JSON.parse(audioDraft));
                      } catch {
                        /* the textarea shows its own parse error below */
                      }
                    }}
                  >
                    Save audio master
                  </button>
                  <span className="vi-script-note">
                    There is no synthesize button. audio.tts is ineligible on
                    this fleet, and a master with fabricated track refs would
                    make every downstream shot window a lie inspection cannot
                    catch.
                  </span>
                </div>
              </>
            )}
            {run.locked && (
              <>
                <table className="vi-script-table">
                  <thead>
                    <tr>
                      <th>revision</th>
                      <th>reason</th>
                      <th>digest</th>
                      <th>at</th>
                    </tr>
                  </thead>
                  <tbody>
                    {run.lock_history.map((h) => (
                      <tr key={h.digest}>
                        <td>{h.revision}</td>
                        <td>{h.reason}</td>
                        <td className="vi-script-digest">{short(h.digest)}</td>
                        <td>{h.at}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="vi-script-row">
                  <input
                    className="vi-knob-input vi-script-grow"
                    placeholder="Reason for this revision (mandatory)"
                    value={reason}
                    onChange={(e) => setReason(e.target.value)}
                  />
                  <button
                    type="button"
                    className="vi-btn"
                    disabled={!reason.trim() || api.busy != null}
                    onClick={() => {
                      api.revise(reason);
                      setReason("");
                    }}
                  >
                    {api.busy === "revise" ? "Revising…" : "Revise"}
                  </button>
                </div>
                <p className="vi-script-note">
                  A revision drops the previously compiled segments: they belong
                  to the previous revision and are never re-pointed at this one.
                </p>
              </>
            )}
          </section>

          <ContinuityViewer run={run} />
          <ShotPlanViewer run={run} />

          <SegmentsPanel
            run={run}
            busy={api.busy}
            segmentBusy={api.segmentBusy}
            onCompile={api.compileSegments}
            onRegenerate={api.regenerate}
            onPromote={(id) => {
              setPromoting(id);
              setPromoteNote("");
            }}
          />

          {promoting && (
            <section className="vi-script-confirm">
              <strong>
                Promote <code>{promoting}</code> as a persisted source?
              </strong>
              <p>
                This writes the segment's prompt as a source a NEW run can
                snapshot. It will be recorded in THIS run's ledger, which makes
                it structurally inadmissible here: feeding it back into this run
                is refused by digest. That is the doc's rule — an accepted
                output influences later work only through a new generation run.
              </p>
              <input
                className="vi-knob-input"
                placeholder="Why is this accepted? (recorded on the source)"
                value={promoteNote}
                onChange={(e) => setPromoteNote(e.target.value)}
              />
              <div className="vi-script-row">
                <button
                  type="button"
                  className="vi-btn vi-btn-accent"
                  disabled={api.busy != null}
                  onClick={() => {
                    api.promote(promoting, "", promoteNote);
                    setPromoting(null);
                  }}
                >
                  Promote for a new run
                </button>
                <button
                  type="button"
                  className="vi-btn"
                  onClick={() => setPromoting(null)}
                >
                  Cancel
                </button>
              </div>
            </section>
          )}

          {lastPromotion && (
            <section className="vi-script-panel">
              <h3>
                Promoted
                <span className="vi-script-sub">{lastPromotion.source_id}</span>
              </h3>
              <p className="vi-script-note">{lastPromotion.usable_in}</p>
              <pre className="vi-script-raw">{lastPromotion.refused_here}</pre>
            </section>
          )}
        </>
      )}
    </section>
  );
}

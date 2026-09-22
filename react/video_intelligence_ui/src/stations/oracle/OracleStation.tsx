/**
 * or-k18 / k114 — the Oracle station: the in-between of generation, made
 * visible.
 *
 *  1. Steward — the selector's self-audit (findings by severity, matrix_stale
 *     banner, last run time, re-run → POST).
 *  2. Run picker — the performance jobs on the media bus (the same
 *     GET /video/jobs feed Active Processes reads), and the picked run's
 *     manifest from GET /video/jobs/<id> → result.movie.
 *  3. Run header — run id, ok/gap, digests, registry version, receipts.
 *  4. Tabs: DAG (node states + per-node drawer) · Spatial overlays (honest
 *     placeholder) · Ledger (central reliability ledger summary + catalog).
 *
 * Every object on this screen is shown with where it came from and a raw JSON
 * fold-out; every absence is named (route missing, selector disabled, no dag
 * block, no spatial conditioning) rather than left blank.
 */
import { useState } from "react";

import type { StationSpec } from "../types";
import { DagPanel } from "./DagPanel";
import { SpatialTab } from "./SpatialTab";
import { StewardPanel } from "./StewardPanel";
import { GapNote, RawJson, StateChip, Tag, short, when } from "./oracleShared";
import {
  useCapabilities,
  useLedgerSummary,
  usePerformanceJobs,
  useRunManifest,
  useSteward,
  type RunManifestApi,
} from "./useOracle";
import "./oracle.css";

type TabId = "dag" | "spatial" | "ledger";

function RunPicker({ runs }: { runs: RunManifestApi }) {
  const jobs = usePerformanceJobs();
  const [manual, setManual] = useState("");
  return (
    <section className="vi-oracle-panel" aria-label="Runs">
      <h3>
        Runs
        <span className="vi-oracle-sub">performance jobs on the media bus (GET /video/jobs?all=1) — pick one to read its manifest</span>
      </h3>
      <div className="vi-oracle-row">
        <button type="button" className="vi-btn vi-btn-sm" onClick={jobs.refresh}>
          Refresh list
        </button>
        <input
          className="vi-oracle-grow"
          placeholder="…or paste a job id"
          value={manual}
          onChange={(e) => setManual(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && manual.trim()) runs.select(manual.trim());
          }}
        />
        <button type="button" className="vi-btn vi-btn-sm" disabled={!manual.trim()} onClick={() => runs.select(manual.trim())}>
          Open
        </button>
        {runs.jobId ? (
          <button type="button" className="vi-btn vi-btn-sm vi-btn-ghost" onClick={() => runs.select(null)}>
            Clear
          </button>
        ) : null}
      </div>
      <GapNote gap={jobs.gap} />
      <div className="vi-oracle-runs" style={{ marginTop: "0.5rem" }}>
        {jobs.jobs.map((j) => (
          <button
            key={j.id}
            type="button"
            className={`vi-btn vi-btn-sm${runs.jobId === j.id ? " vi-btn-on" : ""}`}
            onClick={() => runs.select(j.id)}
            title={`${j.name} · ${j.status}${j.runId ? ` · run ${j.runId}` : ""}`}
          >
            {short(j.id, 10)} · {j.status}
            {j.runId ? ` · ${short(j.runId, 10)}` : ""}
          </button>
        ))}
        {!jobs.jobs.length && !jobs.gap ? (
          <span className="vi-knob-hint">
            No performance jobs in the bus feed (live or recent terminal). Enqueue one via the oracle runner, or paste an
            older job id.
          </span>
        ) : null}
      </div>
    </section>
  );
}

function RunHeader({ runs }: { runs: RunManifestApi }) {
  const m = runs.manifest;
  const job = runs.job;
  if (!runs.jobId) return null;
  return (
    <section className="vi-oracle-panel" aria-label="Run">
      <h3>
        Run
        <span className="vi-oracle-sub">GET /video/jobs/{runs.jobId} → result.movie</span>
        <button type="button" className="vi-btn vi-btn-sm" style={{ marginLeft: "auto" }} onClick={runs.refresh} disabled={runs.loading}>
          {runs.loading ? "Reading…" : "Re-read"}
        </button>
      </h3>
      <GapNote gap={runs.gap} />
      {job && !m ? (
        <p className="vi-oracle-gap" role="status">
          Job <code>{job.job_id}</code> is <StateChip state={job.status ?? "unknown"} />
          {job.result ? (
            <> and its result carries no manifest (<code>result.movie</code> is empty){job.result.error ? <> — {job.result.error.code}: {job.result.error.message}</> : null}.</>
          ) : (
            <> with no result yet — the manifest is written at the terminal transition.</>
          )}
          {job.progress ? <RawJson label="live progress" value={job.progress} /> : null}
        </p>
      ) : null}
      {m ? (
        <>
          <div className="vi-oracle-row">
            <Tag kind={m.ok ? "ok" : "alarm"}>{m.ok ? "ok" : "not ok"}</Tag>
            {m.stopped_after ? <Tag kind="warn">stopped after {m.stopped_after}</Tag> : null}
            {job?.status ? <StateChip state={job.status} /> : null}
            {m.dag ? <Tag kind="accent">ran on the DAG</Tag> : <Tag>linear recipe</Tag>}
            {job?.result?.error ? (
              <Tag kind="alarm">
                {job.result.error.code}: {job.result.error.message}
              </Tag>
            ) : null}
          </div>
          <dl className="vi-oracle-kv" style={{ marginTop: "0.5rem" }}>
            <dt>run id</dt>
            <dd>
              <code>{m.run_id}</code>
            </dd>
            <dt>registry version</dt>
            <dd>
              <code>{m.registry_version ?? "not recorded"}</code>
            </dd>
            <dt>goal / snapshot / lock</dt>
            <dd>
              <code>{short(m.goal_digest)}</code> · <code>{short(m.snapshot_digest)}</code> · <code>{short(m.lock_digest)}</code>
            </dd>
            <dt>segments</dt>
            <dd>{m.segment_digests?.length ?? 0}</dd>
            <dt>video</dt>
            <dd>{m.video_ref ? <code title={m.video_ref}>{short(m.video_ref, 56)}</code> : "none accepted"}</dd>
            <dt>receipts</dt>
            <dd>
              {(m.receipts ?? []).length ? (
                <ul className="vi-oracle-reasons" style={{ marginTop: 0 }}>
                  {(m.receipts ?? []).map((r, i) => (
                    <li key={i}>
                      <code>{r.capability}</code> → <code>{r.model_id}</code>
                      {r.worker ? <> on {r.worker}</> : null} · {when(r.started_at)}
                      {typeof r.duration_s === "number" ? <> · {r.duration_s.toFixed(1)}s</> : null}
                      {r.failure ? <Tag kind="alarm">{r.failure}</Tag> : null}
                    </li>
                  ))}
                </ul>
              ) : (
                <span className="vi-knob-hint">none in the manifest</span>
              )}
            </dd>
            <dt>state path</dt>
            <dd>
              <code>{m.state_path ?? "—"}</code>
            </dd>
          </dl>
          {m.gap ? (
            <p className="vi-oracle-gap">
              gap at stage <code>{String(m.gap.stage)}</code>: {String(m.gap.diagnosis ?? "")}
              {m.gap.requirement ? <> — required: {String(m.gap.requirement)}</> : null}
            </p>
          ) : null}
          {m.limitations?.length ? (
            <ul className="vi-oracle-limits">
              {m.limitations.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          ) : null}
          {m.warnings?.length ? <RawJson label={`warnings (${m.warnings.length})`} value={m.warnings} /> : null}
          <RawJson label="manifest json" value={m} />
        </>
      ) : null}
    </section>
  );
}

function LedgerTab() {
  const ledger = useLedgerSummary();
  const caps = useCapabilities();
  return (
    <>
      <section className="vi-oracle-panel" aria-label="Ledger">
        <h3>
          Reliability ledger
          <span className="vi-oracle-sub">GET /api/oracle/ledger/summary — the ONE record workers share</span>
          <button type="button" className="vi-btn vi-btn-sm" style={{ marginLeft: "auto" }} onClick={ledger.refresh}>
            Refresh
          </button>
        </h3>
        <GapNote gap={ledger.gap} />
        {ledger.summary ? (
          <>
            <div className="vi-oracle-row">
              {ledger.summary.remote ? <Tag kind="accent">remote: {String(ledger.summary.remote)}</Tag> : <Tag>local ledger</Tag>}
              {Object.entries(ledger.summary)
                .filter(([k, v]) => typeof v === "number" && k !== "ok")
                .map(([k, v]) => (
                  <span key={k}>
                    <Tag>
                      {k}: {String(v)}
                    </Tag>
                  </span>
                ))}
            </div>
            <RawJson label="ledger summary json" value={ledger.summary} open />
          </>
        ) : !ledger.gap ? (
          <p className="vi-oracle-note">Reading the ledger…</p>
        ) : null}
      </section>
      <section className="vi-oracle-panel" aria-label="Capabilities">
        <h3>
          Capability catalog
          <span className="vi-oracle-sub">GET /api/oracle/capabilities — registry {caps.caps?.registry_version ? <code>{short(caps.caps.registry_version)}</code> : "unknown"}</span>
        </h3>
        <GapNote gap={caps.gap} />
        {caps.caps?.capabilities?.length ? (
          <table className="vi-oracle-table">
            <thead>
              <tr>
                <th>capability</th>
                <th>eligible models</th>
              </tr>
            </thead>
            <tbody>
              {caps.caps.capabilities.map((c) => (
                <tr key={c.name}>
                  <td>
                    <code>{c.name}</code>
                  </td>
                  <td>{c.model_ids?.length ? c.model_ids.map((m) => <code key={m} style={{ marginRight: "0.5rem" }}>{m}</code>) : <span className="vi-knob-hint">none eligible</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : caps.caps && !caps.gap ? (
          <p className="vi-oracle-note">The catalog answered with zero capabilities.</p>
        ) : null}
        {caps.caps ? <RawJson label="capabilities json" value={caps.caps} /> : null}
      </section>
    </>
  );
}

export function OracleStation({ spec }: { spec: StationSpec }) {
  const steward = useSteward();
  const runs = useRunManifest();
  const [tab, setTab] = useState<TabId>("dag");

  const tabs: Array<{ id: TabId; label: string }> = [
    { id: "dag", label: "DAG" },
    { id: "spatial", label: "Spatial overlays" },
    { id: "ledger", label: "Ledger + catalog" },
  ];

  return (
    <section className="station-card station-wide vi-oracle" aria-label={spec.title}>
      <p className="station-phase">{spec.phase}</p>
      <p className="station-blurb">{spec.blurb}</p>

      <StewardPanel api={steward} embedded={runs.manifest?.dag?.steward ?? null} />
      <RunPicker runs={runs} />
      <RunHeader runs={runs} />

      <div>
        <div className="vi-oracle-tabs" role="tablist" aria-label="Oracle views">
          {tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              id={`vi-oracle-tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`vi-oracle-panel-${t.id}`}
              className="vi-oracle-tab"
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div role="tabpanel" id={`vi-oracle-panel-${tab}`} aria-labelledby={`vi-oracle-tab-${tab}`}>
          {tab === "dag" ? <DagPanel runs={runs} /> : null}
          {tab === "spatial" ? <SpatialTab manifest={runs.manifest} /> : null}
          {tab === "ledger" ? <LedgerTab /> : null}
        </div>
      </div>
    </section>
  );
}

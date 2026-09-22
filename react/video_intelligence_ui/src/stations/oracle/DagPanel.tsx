/**
 * or-k18 — the DAG state list + the per-node drawer.
 *
 * The list is derived ONLY from the run manifest (GET /video/jobs/<id> →
 * result.movie.dag, the VisualResult the performance relay wrote). Per row:
 * node id, state, the artifact it produced (from shots[]), the producer model
 * (from the manifest's receipts when they list the artifact; otherwise the
 * central ledger via GET /api/oracle/producers?ref=), and the lease — which
 * the manifest does NOT export (it lives in NodeRecord in the journal), so
 * that column says so instead of showing a blank that could be mistaken for
 * "no lease".
 *
 * The drawer for a node:
 *   • selection decision — POST /api/oracle/selection for the node's
 *     capability, candidates by model with every verdict and reason, the
 *     ordered step log. This explains what the selector would pick NOW; it is
 *     not a replay of the journal's decision, and the drawer says so.
 *   • scorecard — confidence, checks, judges, disagreements (from the manifest).
 *   • repair plan — the RepairPlan recorded for this node, when one was.
 */
import { useEffect, useMemo, useState } from "react";

import type { DagNodeRow, RunManifest } from "./oracleTypes";
import {
  capabilityForNode,
  dagRows,
  useProducers,
  useSelectionExplain,
  type ProducerState,
  type RunManifestApi,
} from "./useOracle";
import { GapNote, RawJson, ScorecardView, StateChip, Tag, short, when } from "./oracleShared";

function ProducerCell({ row, state }: { row: DagNodeRow; state: ProducerState | null }) {
  if (row.producerModel) {
    return (
      <>
        <code>{row.producerModel}</code> <span className="vi-knob-hint">(receipt)</span>
      </>
    );
  }
  if (!row.producedRef) {
    return <span className="vi-knob-hint">{capabilityForNode(row.kind) ? "no artifact recorded" : "model-free node"}</span>;
  }
  if (!state || state.status === "loading") return <span className="vi-knob-hint">asking ledger…</span>;
  if (state.status === "hit") {
    return (
      <>
        <code>{state.producer.model_id}</code>{" "}
        <span className="vi-knob-hint">(ledger{state.producer.worker ? ` · ${state.producer.worker}` : ""})</span>
      </>
    );
  }
  if (state.status === "miss") return <span className="vi-knob-hint">ledger has no producer row for this ref</span>;
  return (
    <span className="vi-knob-hint" title={state.gap.message}>
      unknown — {state.gap.code === "not_found" ? "producers route absent" : state.gap.code}
    </span>
  );
}

function NodeDrawer({
  row,
  manifest,
  producer,
  onClose,
}: {
  row: DagNodeRow;
  manifest: RunManifest;
  producer: ProducerState | null;
  onClose: () => void;
}) {
  const capability = capabilityForNode(row.kind);
  const sel = useSelectionExplain();
  const [quality, setQuality] = useState<"preview" | "balanced" | "best">("balanced");

  useEffect(() => {
    if (capability) sel.explain({ capability, quality, candidates: 3 });
    else sel.clear();
    // sel.explain/clear are stable (useCallback with [] deps)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capability, quality, row.nodeId]);

  const decision = sel.decision;
  const dagRun = manifest.dag?.run;
  const producerModel =
    row.producerModel ?? (producer && producer.status === "hit" ? producer.producer.model_id : null);

  return (
    <aside className="vi-oracle-drawer" aria-label={`Node ${row.nodeId}`}>
      <h3>
        <code>{row.nodeId}</code>
        <StateChip state={row.state} />
        <button type="button" className="vi-btn vi-btn-sm vi-oracle-close" onClick={onClose}>
          Close
        </button>
      </h3>
      <dl className="vi-oracle-kv">
        <dt>kind</dt>
        <dd>{row.kind}</dd>
        <dt>segment</dt>
        <dd>{row.segmentId ?? "—"}</dd>
        <dt>capability</dt>
        <dd>{capability ?? <span className="vi-knob-hint">model-free (no selector call)</span>}</dd>
        <dt>produced</dt>
        <dd>{row.producedRef ? <code title={row.producedRef}>{short(row.producedRef, 48)}</code> : "—"}</dd>
        <dt>producer</dt>
        <dd>
          <ProducerCell row={row} state={producer} />
        </dd>
        <dt>lease</dt>
        <dd>
          <span className="vi-knob-hint">not in manifest (journal-only: NodeRecord.lease_owner / lease_expires_at)</span>
        </dd>
        <dt>run revision</dt>
        <dd>
          {dagRun ? (
            <>
              {dagRun.revision} · repairs used {dagRun.revisions_used}/{dagRun.repair_budget}
            </>
          ) : (
            "—"
          )}
        </dd>
      </dl>

      <h4>selection decision</h4>
      {capability ? (
        <>
          <div className="vi-oracle-row">
            <label className="vi-knob-hint">
              quality{" "}
              <select value={quality} onChange={(e) => setQuality(e.target.value as typeof quality)} disabled={sel.busy}>
                <option value="preview">preview</option>
                <option value="balanced">balanced</option>
                <option value="best">best</option>
              </select>
            </label>
            {sel.busy ? <span className="vi-knob-hint">asking the selector…</span> : null}
          </div>
          <p className="vi-oracle-note">
            What the selector would choose for <code>{capability}</code> NOW (POST /api/oracle/selection). It is
            not a replay of this node's journalled decision; the model that actually produced the artifact is the
            producer above{producerModel ? <> (<code>{producerModel}</code>)</> : null}.
          </p>
          <GapNote gap={sel.gap} />
          {decision ? (
            <>
              <div className="vi-oracle-row">
                <Tag kind={decision.gap ? "alarm" : "ok"}>{decision.gap ? "gap — no eligible model" : "decided"}</Tag>
                <Tag>op: {decision.operation}</Tag>
                {decision.explored ? <Tag kind="warn">explored (not the top-ranked)</Tag> : null}
                {decision.spread ? <Tag>spread across candidates</Tag> : null}
                {producerModel && decision.model_id && producerModel !== decision.model_id ? (
                  <Tag kind="warn">differs from producer</Tag>
                ) : null}
              </div>
              <p className="vi-oracle-note">
                pick: <code>{decision.model_id ?? "none"}</code>
                {typeof decision.score === "number" ? <> · score {decision.score.toFixed(3)}</> : null}
                {decision.fallback ? <> · fallback <code>{decision.fallback}</code></> : null}
              </p>
              <p className="vi-oracle-note">{decision.rationale}</p>
              <ul className="vi-oracle-candidates">
                {decision.ranked.map((c) => (
                  <li key={`r-${c.model_id}`} className={`vi-oracle-candidate${c.selected ? " is-selected" : ""}`}>
                    <div className="vi-oracle-candidate-head">
                      <code>{c.model_id}</code>
                      <Tag kind={c.selected ? "accent" : undefined}>{c.selected ? "selected" : "eligible"}</Tag>
                      <code>{c.score.toFixed(3)}</code>
                    </div>
                    {c.reasons.length ? (
                      <ul className="vi-oracle-reasons">
                        {c.reasons.map((r, i) => (
                          <li key={i}>{r}</li>
                        ))}
                      </ul>
                    ) : null}
                    {Object.keys(c.evidence ?? {}).length ? <RawJson label="evidence" value={c.evidence} /> : null}
                  </li>
                ))}
                {decision.rejected.map((c) => (
                  <li key={`x-${c.model_id}`} className="vi-oracle-candidate is-rejected">
                    <div className="vi-oracle-candidate-head">
                      <code>{c.model_id}</code>
                      <Tag kind="alarm">rejected{c.rejected_at ? ` at ${c.rejected_at}` : ""}</Tag>
                    </div>
                    {c.reasons.length ? (
                      <ul className="vi-oracle-reasons">
                        {c.reasons.map((r, i) => (
                          <li key={i}>{r}</li>
                        ))}
                      </ul>
                    ) : null}
                  </li>
                ))}
              </ul>
              {decision.steps.length ? (
                <ol className="vi-oracle-steps">
                  {decision.steps.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ol>
              ) : null}
              <RawJson label="selection json" value={decision} />
            </>
          ) : null}
        </>
      ) : (
        <p className="vi-oracle-note">This node kind runs without a model; there is no selection to explain.</p>
      )}

      <h4>scorecard</h4>
      <ScorecardView card={row.scorecard} />

      <h4>repair plan</h4>
      {row.repair ? (
        <>
          <div className="vi-oracle-row">
            <Tag kind={row.repair.repairable ? "warn" : "alarm"}>{row.repair.repairable ? "repairable" : "not repairable"}</Tag>
            <Tag>code: {row.repair.code}</Tag>
            <Tag>budget left: {row.repair.budget_left}</Tag>
          </div>
          <dl className="vi-oracle-kv" style={{ marginTop: "0.4rem" }}>
            <dt>strategy</dt>
            <dd>{row.repair.strategy}</dd>
            <dt>root</dt>
            <dd>{row.repair.root ?? "—"}</dd>
            <dt>path</dt>
            <dd>
              <code>{row.repair.path.join(" → ") || "—"}</code>
            </dd>
            <dt>param changes</dt>
            <dd>
              <code>{Object.keys(row.repair.param_changes ?? {}).length ? JSON.stringify(row.repair.param_changes) : "none"}</code>
            </dd>
          </dl>
          <p className="vi-oracle-note">{row.repair.rationale}</p>
          <RawJson label="repair plan json" value={row.repair} />
        </>
      ) : (
        <p className="vi-oracle-note">
          {row.failed ? "Node failed but no repair plan was recorded for it." : "No repair was diagnosed for this node."}
        </p>
      )}
    </aside>
  );
}

export function DagPanel({ runs }: { runs: RunManifestApi }) {
  const manifest = runs.manifest;
  const rows = useMemo(() => dagRows(manifest), [manifest]);
  const [selected, setSelected] = useState<string | null>(null);
  const producers = useProducers();

  useEffect(() => {
    setSelected(null);
  }, [runs.jobId]);

  const dag = manifest?.dag ?? null;
  const selectedRow = rows.find((r) => r.nodeId === selected) ?? null;

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const r of rows) c[r.state] = (c[r.state] ?? 0) + 1;
    return c;
  }, [rows]);

  return (
    <div className={`vi-oracle-split${selectedRow ? " has-drawer" : ""}`}>
      <section className="vi-oracle-panel" aria-label="DAG state">
        <h3>
          DAG
          <span className="vi-oracle-sub">node states from the run manifest (result.movie.dag) — journal truth, not a live poll</span>
        </h3>
        {!manifest ? (
          <p className="vi-oracle-note">Pick a performance run above to read its DAG.</p>
        ) : !dag ? (
          <>
            <p className="vi-oracle-note">
              This run's manifest carries no <code>dag</code> block: the visual stages ran on the linear recipe
              (DAG recipe disabled, or the run stopped at <code>{manifest.stopped_after ?? "a pre-visual stage"}</code>),
              so there are no node states to list. The linear journal's stages are below.
            </p>
            <RawJson label="manifest.stages" value={manifest.stages ?? []} />
          </>
        ) : (
          <>
            <div className="vi-oracle-row">
              <StateChip state={dag.run.state} />
              <Tag kind={dag.ok ? "ok" : "alarm"}>{dag.ok ? "deliverable accepted" : "no accepted deliverable"}</Tag>
              <Tag>
                graph <code>{dag.run.graph_id}</code>
              </Tag>
              <Tag>
                rev {dag.run.revision} · repairs {dag.run.revisions_used}/{dag.run.repair_budget}
              </Tag>
              <Tag>updated {when(dag.run.updated_at)}</Tag>
              {Object.entries(counts).map(([s, n]) => (
                <span key={s}>
                  <Tag>
                    {n} {s}
                  </Tag>
                </span>
              ))}
            </div>
            {dag.run.note ? <p className="vi-oracle-note">note: {dag.run.note}</p> : null}
            <table className="vi-oracle-table" style={{ marginTop: "0.6rem" }}>
              <thead>
                <tr>
                  <th>node</th>
                  <th>state</th>
                  <th>producer model</th>
                  <th>lease</th>
                  <th>card</th>
                  <th>repair</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r.nodeId}
                    className={`is-clickable${selected === r.nodeId ? " is-selected" : ""}`}
                    onClick={() => setSelected(selected === r.nodeId ? null : r.nodeId)}
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setSelected(selected === r.nodeId ? null : r.nodeId);
                      }
                    }}
                    aria-selected={selected === r.nodeId}
                  >
                    <td>
                      <code>{r.nodeId}</code>
                    </td>
                    <td>
                      <StateChip state={r.state} />
                    </td>
                    <td>
                      <ProducerCell row={r} state={producers.lookup(r.producedRef)} />
                    </td>
                    <td>
                      <span className="vi-knob-hint" title="NodeRecord.lease_owner is journal-only; the manifest does not export it">
                        not exported
                      </span>
                    </td>
                    <td>
                      {r.scorecard ? (
                        <Tag kind={r.scorecard.hard_pass ? "ok" : "alarm"}>
                          {r.scorecard.hard_pass ? "pass" : "fail"}
                          {typeof r.scorecard.confidence === "number" ? ` · ${r.scorecard.confidence.toFixed(2)}` : ""}
                        </Tag>
                      ) : (
                        <span className="vi-knob-hint">—</span>
                      )}
                    </td>
                    <td>
                      {r.repair ? (
                        <Tag kind={r.repair.repairable ? "warn" : "alarm"}>{r.repair.code}</Tag>
                      ) : (
                        <span className="vi-knob-hint">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {dag.limitations?.length ? (
              <ul className="vi-oracle-limits">
                {dag.limitations.map((l, i) => (
                  <li key={i}>{l}</li>
                ))}
              </ul>
            ) : null}
            {dag.validation ? <RawJson label="graph validation" value={dag.validation} /> : null}
            <RawJson label="manifest.dag json" value={dag} />
          </>
        )}
      </section>
      {selectedRow && manifest ? (
        <NodeDrawer
          row={selectedRow}
          manifest={manifest}
          producer={producers.lookup(selectedRow.producedRef)}
          onClose={() => setSelected(null)}
        />
      ) : null}
    </div>
  );
}

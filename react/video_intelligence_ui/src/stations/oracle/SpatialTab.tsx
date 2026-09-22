/**
 * or-k18 — "spatial overlays", the honest placeholder.
 *
 * The spatial layer (oracle/spatial.py SpatialSceneManifest, oracle/spatial_eval.py
 * SpatialEvalReport) exists, and the DAG grows a `spatial:<seg>` node when a
 * segment carries a manifest — but the evaluator is NOT yet folded into the run
 * manifest, and most runs carry `SegmentSpec.spatial_ref = None`. So this tab
 * does not draw overlays it cannot source. It walks the manifest for anything
 * shaped like a spatial manifest / eval report / spatial_manifest ref and prints
 * what it finds with the JSON path it found it at; otherwise it says "no spatial
 * conditioning" and quotes the run's own limitation line when there is one.
 */
import { useMemo } from "react";

import type { RunManifest } from "./oracleTypes";
import { findSpatial } from "./useOracle";
import { RawJson, Tag } from "./oracleShared";

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function SpatialTab({ manifest }: { manifest: RunManifest | null }) {
  const finds = useMemo(() => findSpatial(manifest), [manifest]);
  const spatialNodes = useMemo(
    () => Object.entries(manifest?.dag?.nodes ?? {}).filter(([id]) => id.startsWith("spatial:")),
    [manifest],
  );
  const limitation = (manifest?.limitations ?? []).find((l) => /spatial/i.test(l)) ?? null;

  if (!manifest) {
    return (
      <section className="vi-oracle-panel" aria-label="Spatial overlays">
        <h3>
          Spatial overlays
          <span className="vi-oracle-sub">placeholder — renders spatial manifests + evaluator metrics when a run carries them</span>
        </h3>
        <p className="vi-oracle-note">Pick a performance run to look for spatial conditioning.</p>
      </section>
    );
  }

  const manifests = finds.filter((f) => f.kind === "manifest");
  const evals = finds.filter((f) => f.kind === "eval");
  const refs = finds.filter((f) => f.kind === "ref");

  return (
    <section className="vi-oracle-panel" aria-label="Spatial overlays">
      <h3>
        Spatial overlays
        <span className="vi-oracle-sub">placeholder — no geometry is drawn here yet; the data is shown as it is</span>
      </h3>

      <div className="vi-oracle-row">
        <Tag kind={manifests.length ? "accent" : undefined}>{manifests.length} spatial manifest(s)</Tag>
        <Tag kind={evals.length ? "accent" : undefined}>{evals.length} evaluator report(s)</Tag>
        <Tag>{refs.length} spatial_manifest ref(s) on segments</Tag>
        <Tag>{spatialNodes.length} spatial:* DAG node(s)</Tag>
      </div>

      {!manifests.length && !evals.length && !refs.length && !spatialNodes.length ? (
        <>
          <p className="vi-oracle-gap" role="status">
            No spatial conditioning on this run. {limitation ? <>The run says so itself: “{limitation}”</> : <>No SegmentSpec carried a <code>spatial_ref</code>, no <code>spatial:&lt;seg&gt;</code> node was built, and no SpatialSceneManifest or SpatialEvalReport appears anywhere in the manifest.</>}
          </p>
          <p className="vi-oracle-note">
            When a run is conditioned, this tab will list each segment's manifest (coordinate system, camera, entities,
            timebase, provenance) and the evaluator's per-rubric metrics with their codes and thresholds. Overlay
            rendering on frames is not built; nothing here will pretend it is.
          </p>
        </>
      ) : null}

      {spatialNodes.length ? (
        <>
          <h4>spatial DAG nodes</h4>
          <table className="vi-oracle-table">
            <thead>
              <tr>
                <th>node</th>
                <th>state</th>
              </tr>
            </thead>
            <tbody>
              {spatialNodes.map(([id, st]) => (
                <tr key={id}>
                  <td>
                    <code>{id}</code>
                  </td>
                  <td>{String(st)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="vi-oracle-note">
            A spatial node validates the segment's manifest against the lock and emits the conditioning request; its
            outputs (conditioning, manifest_digest, validation) live in the journal and are not exported in the manifest.
          </p>
        </>
      ) : null}

      {refs.length ? (
        <>
          <h4>spatial_manifest refs</h4>
          <ul className="vi-oracle-reasons">
            {refs.map((f) => (
              <li key={f.path}>
                <code>{f.path}</code> → <code>{String(f.value)}</code>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      {manifests.map((f) => {
        const m = isRecord(f.value) ? f.value : {};
        return (
          <div key={f.path}>
            <h4>
              spatial manifest <span className="vi-knob-hint">at {f.path || "(root)"}</span>
            </h4>
            <dl className="vi-oracle-kv">
              <dt>segment</dt>
              <dd>{String(m.segment_id ?? "—")}</dd>
              <dt>schema</dt>
              <dd>{String(m.schema_version ?? "—")}</dd>
              <dt>revision</dt>
              <dd>{String(m.artifact_revision ?? "—")}</dd>
              <dt>entities</dt>
              <dd>{Array.isArray(m.entities) ? m.entities.length : "—"}</dd>
              <dt>fallbacks</dt>
              <dd>{Array.isArray(m.fallbacks) ? m.fallbacks.length : "—"}</dd>
            </dl>
            <RawJson label="manifest json" value={m} open />
          </div>
        );
      })}

      {evals.map((f) => {
        const r = isRecord(f.value) ? f.value : {};
        const metrics = Array.isArray(r.metrics) ? (r.metrics as Array<Record<string, unknown>>) : [];
        return (
          <div key={f.path}>
            <h4>
              evaluator report <span className="vi-knob-hint">at {f.path || "(root)"}</span>
            </h4>
            <div className="vi-oracle-row">
              <Tag kind={r.ok ? "ok" : "alarm"}>{r.ok ? "ok" : "drift"}</Tag>
              <Tag>segment {String(r.segment_id ?? "—")}</Tag>
              <Tag>{Array.isArray(r.measured) ? r.measured.length : 0} measured</Tag>
              <Tag>{Array.isArray(r.skipped) ? r.skipped.length : 0} skipped</Tag>
            </div>
            {metrics.length ? (
              <table className="vi-oracle-table" style={{ marginTop: "0.5rem" }}>
                <thead>
                  <tr>
                    <th>metric</th>
                    <th>value</th>
                    <th>threshold</th>
                    <th>code</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m, i) => (
                    <tr key={i}>
                      <td>{String(m.name ?? m.rubric ?? i)}</td>
                      <td>
                        <code>{String(m.value ?? m.measured ?? "—")}</code>
                      </td>
                      <td>
                        <code>{String(m.threshold ?? "—")}</code>
                      </td>
                      <td>{m.code ? <Tag kind="warn">{String(m.code)}</Tag> : <span className="vi-knob-hint">—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : null}
            <RawJson label="evaluator json" value={r} />
          </div>
        );
      })}
    </section>
  );
}

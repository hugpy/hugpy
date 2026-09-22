// CharacterGroupsPanel — the char360 REVIEW + CURATE surface (dev/CHARACTER-GROUPS-PLAN.md
// slice S2). A sibling of ExtractFromVideoPanel inside the Frames station's
// "Character tools" group: where ExtractFromVideoPanel commits char→profile mapping
// SERVER-SIDE at submit time (target:"create"/"<slug>"), THIS panel runs the S1
// "review" extraction (target:"review", enqueue-only) which returns the per-character
// grouped manifest WITHOUT writing anything, then lets the user curate the groups
// (remove / move / merge / rename / select-to-proceed) and only THEN commits the
// chosen groups to identity profiles via the S3 from-groups route.
//
// WHY A SIBLING, NOT A SECOND useIdentityProfiles() MOUNT — the same reason
// GenIdentityBar is a pure prop-driven sibling: useIdentityProfiles() has no
// cross-instance cache, so mounting it twice fires duplicate GET
// /video/identity-profiles and holds two independently-stale lists. This panel is
// PURE — it takes the ONE station hook's `extractFromVideo` + the loaded `source`
// as props (FrameExtractCore owns the single hook instance).
//
// CURATION STATE IS CLIENT-HELD — keyed off the S1 manifest, mutated locally
// (remove/move/merge). Nothing is written until "Create identities from selected
// groups" POSTs the from-groups body. Reversible by design (re-run review to reset).
import { useEffect, useRef, useState } from "react";
import {
  mediaBytesUrl,
  videoJobUrl,
  hugpyConfig,
  identityGenerateUrl,
  identityMeshStatusUrl,
} from "../../config";
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import type { MediaRef } from "../../video/contract";
import type { IdentityProfilesState } from "../studio/useIdentityProfiles";
import { MeshViewer } from "../studio/IdentitiesStation";

// ── S1 review manifest shapes (tolerant reads — the poll result is cast, exactly
// as ExtractFromVideoPanel casts its job result; strictNullChecks is OFF in this
// scoped tsconfig so these are flat, nullable-friendly shapes). ─────────────────
interface ReviewView {
  url: string; // media HANDLE (jailed absolute path) — render via mediaBytesUrl(url)
  yaw: number | null;
  bin: number | null;
  score: number | null;
}
interface ReviewGroup {
  char: string;
  face_centroid: number[] | null;
  views: ReviewView[];
}
interface ReviewManifest {
  n_characters: number;
  groups: ReviewGroup[];
}

// ── client-held (editable) curation model ───────────────────────────────────────
interface GroupState {
  /** Stable local id (survives rename; merges keep the target id). */
  id: string;
  /** Editable display name — defaults to the manifest `char` id. */
  name: string;
  /** The original char360 char id (immutable provenance, shown as a subtitle). */
  char: string;
  views: ReviewView[];
  /** Whether this group advances to the commit step. */
  proceed: boolean;
  /** Per-group commit outcome (set after a from-groups POST), or null. */
  slug: string | null;
  error: string | null;
}

// Response of the S3 from-groups route: { results: [{ name, ok, slug?, error? }] }.
interface CommitResult {
  name?: string;
  ok?: boolean;
  slug?: string;
  error?: string;
}

// ── S4 per-group 3D job state (dev/CHARACTER-GROUPS-PLAN.md) — keyed by the
// committed profile `slug`. Owned INTERNALLY by this panel so "Generate 3D"
// works whether or not a caller passes `onGenerate3D` (that prop stays a pure
// fire-and-forget notify hook, called once when a job starts, never driving
// state here). Mirrors IdentitiesStation's `generateFullIdentity` one-click
// flow verbatim: POST identityGenerateUrl(slug) with a bare {} body (the
// happy path — mesh + 360 turntable + auto-promote), then poll
// identityMeshStatusUrl(slug, reconId) every ~5s (bounded ~20min) for the
// SAME status vocabulary ("none"|"queued"|"running"|"done"|"error"|
// "cancelled") and the SAME terminal field, `glb_path` (snake_case) —
// wrapped with mediaBytesUrl for MeshViewer, exactly as Studio does.
type Mesh3DPhase = "idle" | "running" | "done" | "error";
interface Mesh3DState {
  phase: Mesh3DPhase;
  status: string | null; // last raw poll status / stage text for display
  glbUrl: string | null;
  error: string | null;
}
const IDLE_3D: Mesh3DState = { phase: "idle", status: null, glbUrl: null, error: null };

/** Composite selection key — a view lives in exactly one group at a time, but a
 *  handle could (in principle) repeat across groups in the raw manifest, so the
 *  group id is part of the key. U+241F is a control-picture separator that can't
 *  appear in a group id or a path. */
function selKey(groupId: string, url: string): string {
  return `${groupId}␟${url}`;
}

export interface CharacterGroupsPanelProps {
  /** The loaded source video (kind==="video") — review runs against THIS ref. */
  source: MediaRef | null;
  /** The ONE station hook's enqueue verb (shared — never a second hook mount). */
  extractFromVideo: IdentityProfilesState["extractFromVideo"];
  /** S4 (dev/CHARACTER-GROUPS-PLAN.md): the panel now owns the actual mesh-gen
   *  flow internally (POST identityGenerateUrl(slug) → poll
   *  identityMeshStatusUrl → MeshViewer), so this button works with or without
   *  a caller-supplied hook. This prop is now an OPTIONAL fire-and-forget
   *  notify — called once when a group's "Generate 3D" job starts, for a
   *  caller that wants to observe it (e.g. cross-station touch tracking). It
   *  never drives this panel's own state. */
  onGenerate3D?: (slug: string) => void;
}

export function CharacterGroupsPanel({
  source,
  extractFromVideo,
  onGenerate3D,
}: CharacterGroupsPanelProps) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [groups, setGroups] = useState<GroupState[]>([]);
  const [reviewed, setReviewed] = useState(false);
  // Multi-select ACROSS groups — composite selKey(groupId, url) entries.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [committing, setCommitting] = useState(false);
  const [commitError, setCommitError] = useState<string | null>(null);
  // S4 — per-slug 3D job state + its own active/timer refs (one poll loop per
  // slug can run concurrently, mirrors the group-level state being a map too).
  const [mesh3D, setMesh3D] = useState<Record<string, Mesh3DState>>({});
  const meshActive = useRef<Record<string, boolean>>({});
  const meshTimer = useRef<Record<string, number>>({});

  const active = useRef(false);
  const timer = useRef<number | null>(null);
  const idSeq = useRef(0);

  useEffect(() => {
    return () => {
      active.current = false;
      if (timer.current != null) window.clearTimeout(timer.current);
      for (const slug of Object.keys(meshActive.current)) meshActive.current[slug] = false;
      for (const slug of Object.keys(meshTimer.current)) window.clearTimeout(meshTimer.current[slug]);
    };
  }, []);

  const sourceIsVideo = source != null && source.kind === "video";
  const canReview = sourceIsVideo && !busy;

  function nextId(): string {
    idSeq.current += 1;
    return `cg${idSeq.current}`;
  }

  // Build the editable model from the S1 manifest. Fresh ids each review so a
  // re-run cleanly replaces the prior curation (reversible reset).
  function loadManifest(manifest: ReviewManifest) {
    const built: GroupState[] = (manifest.groups || []).map((g) => ({
      id: nextId(),
      name: g.char,
      char: g.char,
      views: (g.views || []).filter((v) => v && v.url),
      proceed: false,
      slug: null,
      error: null,
    }));
    setGroups(built);
    setSelected(new Set());
    setReviewed(true);
  }

  async function runReview() {
    if (!source || busy) return;
    setBusy(true);
    setError(null);
    setCommitError(null);
    setStatus("Reviewing… (queued)");
    active.current = true;

    // target:"review" — the S1 enqueue-only, non-committing branch.
    const enq = await extractFromVideo(source, "review");
    if (!active.current) return;
    if (!enq.ok || !enq.jobId) {
      active.current = false;
      setBusy(false);
      setStatus(null);
      setError(enq.message || "Could not start the character review.");
      return;
    }

    const jobId = enq.jobId;
    const deadline = Date.now() + 20 * 60 * 1000; // ~20 min bound, mirrors ExtractFromVideoPanel
    const pollOnce = async () => {
      if (!active.current) return;
      if (Date.now() > deadline) {
        active.current = false;
        setBusy(false);
        setStatus(null);
        setError("Timed out waiting for the character review.");
        return;
      }
      const res = await request<unknown>(videoJobUrl(jobId), {
        meta: { specKey: "studio", operation: "identity.groups.review.status" },
      });
      if (!active.current) return;
      if (res.ok) {
        const job = okValue(res) as {
          progress?: number | null;
          result?: {
            ok?: boolean;
            groups?: ReviewManifest;
            error?: { code?: string; message?: string; retryable?: boolean };
          } | null;
          stage?: string | null;
          message?: string | null;
        };
        const result = job.result;
        if (result && typeof result.ok === "boolean") {
          active.current = false;
          setBusy(false);
          if (result.ok) {
            setStatus("done");
            if (result.groups && Array.isArray(result.groups.groups)) {
              loadManifest(result.groups);
            } else {
              setError("Review finished but returned no character groups.");
            }
          } else {
            setStatus(null);
            setError((result.error && result.error.message) || "Character review failed.");
          }
          return;
        }
        const pct = typeof job.progress === "number" ? ` (${Math.round(job.progress * 100)}%)` : "";
        const stage = job.stage || job.message;
        setStatus(`Reviewing…${pct}${stage ? ` — ${stage}` : ""}`);
      }
      // queued/running/transient-poll-failure → keep polling.
      timer.current = window.setTimeout(() => void pollOnce(), 5000);
    };
    timer.current = window.setTimeout(() => void pollOnce(), 5000);
  }

  // ── selection helpers ─────────────────────────────────────────────────────
  function toggleView(groupId: string, url: string) {
    const key = selKey(groupId, url);
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }
  function clearSelection() {
    setSelected(new Set());
  }
  const selectedCount = selected.size;

  // ── curation ops (all client-held) ────────────────────────────────────────
  // Remove every selected view from its group.
  function removeSelected() {
    if (selectedCount === 0) return;
    setGroups((prev) =>
      prev.map((g) => ({
        ...g,
        views: g.views.filter((v) => !selected.has(selKey(g.id, v.url))),
      })),
    );
    clearSelection();
  }

  // Move every selected view into `destId` (append; skip views already there).
  function moveSelected(destId: string) {
    if (selectedCount === 0 || !destId) return;
    setGroups((prev) => {
      // Collect the moving views (from any source group), preserving order.
      const moving: ReviewView[] = [];
      for (const g of prev) {
        for (const v of g.views) {
          if (g.id !== destId && selected.has(selKey(g.id, v.url))) moving.push(v);
        }
      }
      if (moving.length === 0) return prev;
      return prev.map((g) => {
        if (g.id === destId) {
          const have = new Set(g.views.map((v) => v.url));
          const added = moving.filter((v) => !have.has(v.url));
          return { ...g, views: [...g.views, ...added] };
        }
        // Drop the moved views from their source group.
        return {
          ...g,
          views: g.views.filter((v) => !selected.has(selKey(g.id, v.url))),
        };
      });
    });
    clearSelection();
  }

  // Merge `srcId` INTO `destId` — dest keeps its name/id and gains src's views
  // (deduped by handle); src is dropped.
  function mergeInto(srcId: string, destId: string) {
    if (!srcId || !destId || srcId === destId) return;
    setGroups((prev) => {
      const src = prev.find((g) => g.id === srcId);
      if (!src) return prev;
      return prev
        .map((g) => {
          if (g.id !== destId) return g;
          const have = new Set(g.views.map((v) => v.url));
          const added = src.views.filter((v) => !have.has(v.url));
          return { ...g, views: [...g.views, ...added] };
        })
        .filter((g) => g.id !== srcId);
    });
    // Drop any selection that pointed at the vanished source group.
    setSelected((prev) => {
      const next = new Set<string>();
      for (const k of prev) if (!k.startsWith(`${srcId}␟`)) next.add(k);
      return next;
    });
  }

  function renameGroup(id: string, name: string) {
    setGroups((prev) => prev.map((g) => (g.id === id ? { ...g, name } : g)));
  }
  function toggleProceed(id: string) {
    setGroups((prev) => prev.map((g) => (g.id === id ? { ...g, proceed: !g.proceed } : g)));
  }

  // ── commit (S3 from-groups) ────────────────────────────────────────────────
  const proceedGroups = groups.filter((g) => g.proceed && g.views.length > 0);
  const canCommit = proceedGroups.length > 0 && !committing;

  async function commit() {
    if (!canCommit) return;
    setCommitting(true);
    setCommitError(null);
    // Ordered payload — results come back positionally aligned to THIS order.
    const ordered = proceedGroups;
    const body = {
      groups: ordered.map((g) => ({
        name: g.name.trim() || g.char,
        reference_images: g.views.map((v) => v.url),
      })),
    };
    const res = await request<unknown>(hugpyConfig.identityProfilesFromGroupsUrl, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "studio", operation: "identity.groups.commit" },
    });
    setCommitting(false);
    if (!res.ok) {
      setCommitError(describeAppError(errorOf(res)));
      return;
    }
    const payload = okValue(res) as { results?: CommitResult[] };
    const results = Array.isArray(payload.results) ? payload.results : [];
    // Zip results back onto the submitted groups by index (positional contract).
    setGroups((prev) =>
      prev.map((g) => {
        const idx = ordered.findIndex((o) => o.id === g.id);
        if (idx < 0) return g;
        const r = results[idx];
        if (!r) return { ...g, error: "No result returned for this group." };
        if (r.ok && r.slug) return { ...g, slug: r.slug, error: null };
        return { ...g, slug: null, error: r.error || "Commit failed for this group." };
      }),
    );
  }

  // ── S4: per-group 3D generation (dev/CHARACTER-GROUPS-PLAN.md) ────────────
  // Reuses the ALREADY-EXISTING mesh-from-profile backend — POST
  // identityGenerateUrl(slug) with a bare {} body (the documented happy path:
  // Hunyuan3D mesh + 360° turntable + auto-promoted canonical, off the
  // profile's own reference images, no prior reconstruction needed) →
  // {job_id, recon_id}. Then poll identityMeshStatusUrl(slug, recon_id) — the
  // SAME retrieval Studio's generateFullIdentity uses — until the mesh state
  // block reports status:"done" with a glb_path (or a terminal error).
  function setMesh(slug: string, patch: Partial<Mesh3DState>) {
    setMesh3D((prev) => ({ ...prev, [slug]: { ...(prev[slug] || IDLE_3D), ...patch } }));
  }

  async function generate3D(slug: string) {
    if (!slug) return;
    if (meshActive.current[slug]) return; // already running for this slug
    if (onGenerate3D) onGenerate3D(slug); // optional fire-and-forget notify hook
    meshActive.current[slug] = true;
    setMesh(slug, { phase: "running", status: "Generating… (queued)", glbUrl: null, error: null });

    const res = await request<unknown>(identityGenerateUrl(slug), {
      method: "POST",
      body: JSON.stringify({}),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "studio", operation: "identity.groups.generate3d" },
    });
    if (!meshActive.current[slug]) return;
    if (!res.ok) {
      meshActive.current[slug] = false;
      setMesh(slug, { phase: "error", status: null, error: describeAppError(errorOf(res)) });
      return;
    }
    const enq = okValue(res) as { job_id?: string; recon_id?: string };
    const reconId = enq.recon_id;
    if (!reconId) {
      meshActive.current[slug] = false;
      setMesh(slug, { phase: "error", status: null, error: "Malformed generate response." });
      return;
    }

    const deadline = Date.now() + 20 * 60 * 1000; // ~20 min bound, mirrors Studio's generate poll
    const pollOnce = async () => {
      if (!meshActive.current[slug]) return;
      if (Date.now() > deadline) {
        meshActive.current[slug] = false;
        setMesh(slug, { phase: "error", status: null, error: "Timed out waiting for the 3D build." });
        return;
      }
      const sres = await request<unknown>(identityMeshStatusUrl(slug, reconId), {
        meta: { specKey: "studio", operation: "identity.groups.mesh.status" },
      });
      if (!meshActive.current[slug]) return;
      if (sres.ok) {
        const ms = okValue(sres) as { status?: string; error?: string | null; glb_path?: string | null };
        const st = ms.status ?? "queued";
        if (st === "done") {
          meshActive.current[slug] = false;
          if (ms.glb_path) {
            setMesh(slug, { phase: "done", status: "done", glbUrl: mediaBytesUrl(ms.glb_path), error: null });
          } else {
            setMesh(slug, { phase: "error", status: null, error: "Build finished but returned no model." });
          }
          return;
        }
        if (st === "error" || st === "cancelled") {
          meshActive.current[slug] = false;
          setMesh(slug, { phase: "error", status: null, error: ms.error || `Build ${st}.` });
          return;
        }
        setMesh(slug, { status: `Generating… (${st})` });
      }
      // queued/running/transient-poll-failure → keep polling.
      meshTimer.current[slug] = window.setTimeout(() => void pollOnce(), 5000);
    };
    meshTimer.current[slug] = window.setTimeout(() => void pollOnce(), 5000);
  }

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <section
      className="vi-comfy-bar"
      aria-label="Review and curate character groups"
      style={{ marginTop: "0.8rem", flexDirection: "column", alignItems: "stretch" }}
    >
      <div>
        <span className="vi-comfy-label">Character groups — review &amp; curate</span>
        <p className="vi-comfy-hint" style={{ margin: "0.15rem 0 0.4rem" }}>
          Run a NON-committing char360 review over the loaded video: detected characters
          come back as editable groups. Remove, move, or merge images, rename groups, mark
          which ones to proceed, then create identity profiles from the ones you keep.
        </p>
      </div>

      <div className="vi-comfy-bar-actions vi-run-bar" style={{ flexWrap: "wrap" }}>
        <button
          type="button"
          className="vi-btn vi-btn-accent"
          disabled={!canReview}
          onClick={() => void runReview()}
        >
          {busy ? "Reviewing…" : reviewed ? "Re-run review" : "Review characters"}
        </button>
        {!sourceIsVideo && (
          <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
            Load a video above to review its characters.
          </span>
        )}
        {status && (
          <span className="vi-comfy-hint" role="status">
            {status}
          </span>
        )}
      </div>
      {error && (
        <p className="vi-error" role="alert" style={{ margin: 0 }}>
          {error}
        </p>
      )}

      {reviewed && groups.length === 0 && !busy && (
        <p className="vi-comfy-hint">No characters were detected in this video.</p>
      )}

      {groups.length > 0 && (
        <>
          {/* Cross-group selection toolbar. */}
          <div className="vi-frames-toolbar" style={{ flexWrap: "wrap" }}>
            <span className="vi-frames-summary">
              {groups.length} group{groups.length === 1 ? "" : "s"} · {selectedCount} image
              {selectedCount === 1 ? "" : "s"} selected
            </span>
            <div className="vi-frames-actions" style={{ flexWrap: "wrap" }}>
              <button
                type="button"
                className="vi-btn vi-btn-sm vi-btn-ghost"
                onClick={clearSelection}
                disabled={selectedCount === 0}
              >
                Clear selection
              </button>
              <button
                type="button"
                className="vi-btn vi-btn-sm vi-btn-ghost"
                onClick={removeSelected}
                disabled={selectedCount === 0}
                title="Remove the selected images from their groups"
              >
                Remove selected
              </button>
              <label className="vi-comfy-field" style={{ flexBasis: "auto" }}>
                <span className="vi-comfy-label">Move selected to</span>
                <select
                  className="vi-knob-input"
                  value=""
                  disabled={selectedCount === 0}
                  onChange={(e) => {
                    if (e.target.value) moveSelected(e.target.value);
                    e.target.value = "";
                  }}
                >
                  <option value="">choose group…</option>
                  {groups.map((g) => (
                    <option key={g.id} value={g.id}>
                      {g.name || g.char}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          {/* One card per group. */}
          {groups.map((g) => {
            const mergeTargets = groups.filter((o) => o.id !== g.id);
            return (
              <div key={g.id} className="vi-character-tools" style={{ marginTop: "0.5rem" }}>
                <div
                  className="vi-character-tools-head"
                  style={{ flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: "0.6rem" }}
                >
                  <label className="vi-comfy-field" style={{ flexBasis: "14rem" }}>
                    <span className="vi-comfy-label">Group name</span>
                    <input
                      type="text"
                      className="vi-knob-input"
                      value={g.name}
                      placeholder={g.char}
                      onChange={(e) => renameGroup(g.id, e.target.value)}
                    />
                  </label>
                  <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
                    char360 id: {g.char} · {g.views.length} image{g.views.length === 1 ? "" : "s"}
                  </span>
                  <label className="vi-knob-check" style={{ marginLeft: "auto" }}>
                    <input
                      type="checkbox"
                      checked={g.proceed}
                      onChange={() => toggleProceed(g.id)}
                    />
                    proceed to commit
                  </label>
                  {mergeTargets.length > 0 && (
                    <label className="vi-comfy-field" style={{ flexBasis: "auto" }}>
                      <span className="vi-comfy-label">Merge into</span>
                      <select
                        className="vi-knob-input"
                        value=""
                        onChange={(e) => {
                          if (e.target.value) mergeInto(g.id, e.target.value);
                          e.target.value = "";
                        }}
                      >
                        <option value="">choose group…</option>
                        {mergeTargets.map((o) => (
                          <option key={o.id} value={o.id}>
                            {o.name || o.char}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                </div>

                {g.views.length === 0 ? (
                  <p className="vi-comfy-hint">No images left in this group.</p>
                ) : (
                  <div className="vi-frames-grid">
                    {g.views.map((v) => {
                      const picked = selected.has(selKey(g.id, v.url));
                      const parts: string[] = [];
                      if (v.yaw != null) parts.push(`yaw ${Math.round(v.yaw)}°`);
                      if (v.bin != null) parts.push(`bin ${v.bin}`);
                      if (v.score != null) parts.push(`score ${v.score.toFixed(2)}`);
                      return (
                        <button
                          type="button"
                          key={v.url}
                          className={picked ? "vi-frame-cell vi-frame-cell-selected" : "vi-frame-cell"}
                          onClick={() => toggleView(g.id, v.url)}
                          aria-pressed={picked}
                          title={`${v.url}${parts.length ? ` — ${parts.join(" · ")}` : ""}`}
                        >
                          <img
                            className="vi-frame-img"
                            src={mediaBytesUrl(v.url)}
                            alt={parts.join(" ") || "character view"}
                            loading="lazy"
                          />
                          {parts.length > 0 && <span className="vi-frame-tag">{parts[0]}</span>}
                          {picked && <span className="vi-frame-check">✓</span>}
                        </button>
                      );
                    })}
                  </div>
                )}

                {/* Per-group commit outcome + the S4 "Generate 3D" flow (self-contained —
                    see generate3D() above; onGenerate3D, if passed, is fired as an
                    optional notify hook only). */}
                {g.slug && (() => {
                  const m3 = mesh3D[g.slug] || IDLE_3D;
                  const running = m3.phase === "running";
                  return (
                    <>
                      <div className="vi-frames-actions" style={{ flexWrap: "wrap", alignItems: "center" }}>
                        <span className="vi-comfy-hint" style={{ color: "var(--vi-accent)" }}>
                          ✓ created identity “{g.name || g.char}” ({g.slug})
                        </span>
                        <button
                          type="button"
                          className="vi-btn vi-btn-sm vi-btn-accent"
                          disabled={running}
                          onClick={() => void generate3D(g.slug as string)}
                          title={`Generate a 3D model for “${g.name || g.char}” (Hunyuan3D mesh + 360° turntable — takes several minutes)`}
                        >
                          {running
                            ? "Generating 3D…"
                            : m3.phase === "done"
                              ? "Regenerate 3D"
                              : "Generate 3D"}
                        </button>
                        {m3.status && (m3.phase === "running") && (
                          <span className="vi-comfy-hint" role="status">
                            {m3.status}
                          </span>
                        )}
                      </div>
                      {m3.error && (
                        <p className="vi-error" role="alert" style={{ margin: 0 }}>
                          {m3.error}
                        </p>
                      )}
                      {m3.phase === "done" && m3.glbUrl && (
                        <div style={{ marginTop: "0.4rem" }}>
                          <MeshViewer url={m3.glbUrl} height={320} />
                        </div>
                      )}
                    </>
                  );
                })()}
                {g.error && (
                  <p className="vi-error" role="alert" style={{ margin: 0 }}>
                    {g.error}
                  </p>
                )}
              </div>
            );
          })}

          {/* Commit bar. */}
          <div className="vi-comfy-bar-actions vi-run-bar" style={{ flexWrap: "wrap" }}>
            <button
              type="button"
              className="vi-btn vi-btn-accent"
              disabled={!canCommit}
              onClick={() => void commit()}
              title="Create one identity profile per selected group from its kept images"
            >
              {committing
                ? "Creating…"
                : `Create identities from selected groups${
                    proceedGroups.length ? ` (${proceedGroups.length})` : ""
                  }`}
            </button>
            {proceedGroups.length === 0 && (
              <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
                Mark at least one non-empty group “proceed to commit”.
              </span>
            )}
          </div>
          {commitError && (
            <p className="vi-error" role="alert" style={{ margin: 0 }}>
              {commitError}
            </p>
          )}
        </>
      )}
    </section>
  );
}

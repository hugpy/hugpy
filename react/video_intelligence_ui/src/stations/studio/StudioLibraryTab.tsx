// The inline LIBRARY SECTION of the single Studio plane (StudioPlane) — the bottom
// half of the one plane: the durable clip catalog as a list-left / player-right view
// with the per-row Details expander and per-clip actions, directly below the generate
// controls (no tabs, no navigation). (The file keeps its historical name; the export
// is `StudioLibrarySection`.)
//
// STATUS (confirmed 2026-07-12, still true): NOT MOUNTED anywhere — StudioPlane.tsx
// does not import this file. Retained per house rule (never delete, archive instead).
// Its row `onClick={() => c.playable && setSelected(c.job_id)}` below already IS a
// working select-to-viewer — but since nothing renders this component, that click
// handler was unreachable in BOTH plane mounts, which is part of why the viewer read as
// a stuck, unchangeable default. The reachable equivalent (a minimal, non-tabbed poster/
// id picker, same idiom) now lives on the viewer itself: StudioViewer.tsx's "▤ Play from
// library" panel, mounted inside the one shared StudioPlane. Re-mounting this whole
// section instead would reintroduce the dual-library redundancy Round 9 deliberately
// removed (see StudioPlane.tsx's header note) — its filters / Details / cancel / use-as-
// source affordances stay deferred here, for history only.
//
// Reuses, does not reinvent:
//   • the clip list + 6s poll come from the plane (useStudioClips) — this section
//     only RENDERS them, so the polling cadence is unchanged;
//   • ClipDetailsPanel + the lazy per-row detail fetch are the original station's;
//   • cancel goes through the SAME media-bus job cancel the rest of the arm uses
//     (POST /video/jobs/<id>/cancel via config.jobCancelUrl) — a studio clip IS a
//     media-bus job, so this is its cancel route (see report note on the brief's
//     /llm/jobs/<id>/cancel, which is the comms store, not the media bus);
//   • "use as source" hands the clip back to the plane, which stages it into the
//     generate controls above (same plane) through applyStage / the studioBridge seam.
//
// Filters live in a COMPACT COLLAPSIBLE strip (not a separate tab): STATUS is fully
// backed by the list; CAPABILITY / MODEL need the detail endpoint (the list projection
// omits them — flagged in the report), so those two filter over loaded details;
// failed/cancelled rows auto-load their detail so their error shows inline without
// expanding, and "Load all details" populates the rest. The Details expander is per-row.
import { useCallback, useEffect, useRef, useState } from "react";
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import { studioClipDetailUrl, jobCancelUrl } from "../../config";
import {
  ClipDetailsPanel,
  clipDetailSchema,
  shortId,
  dims,
  whenText,
  type Clip,
  type ClipDetail,
} from "./studioShared";

export interface StudioLibrarySectionProps {
  clips: Clip[];
  loading: boolean;
  error: string | null;
  reload: () => void;
  refresh: () => void;
  /** Stage this clip as the generate controls' source (extend) — same plane. */
  onUseAsSource: (c: Clip) => void;
  /** Stage this clip as a v2v scene-continuity source (restyle, holds identity). */
  onContinueScene: (c: Clip) => void;
  /** Selected clip id + setter — LIFTED to the plane so the player-forward viewer (in the
   *  generator column) follows the row picked here across the two-column workbench split. */
  selected: string | null;
  setSelected: (id: string | null) => void;
}

const CANCELLABLE = new Set(["queued", "claimed", "running"]);

function capOf(d: ClipDetail | undefined): string | undefined {
  if (!d) return undefined;
  const fromManifest = d.manifest?.capability;
  if (typeof fromManifest === "string") return fromManifest;
  const fromSpec = d.spec?.capability;
  return typeof fromSpec === "string" ? fromSpec : undefined;
}

function modelOf(d: ClipDetail | undefined): string | undefined {
  if (!d) return undefined;
  const fromManifest = d.manifest?.model_id;
  if (typeof fromManifest === "string") return fromManifest;
  const fromSpec = d.spec?.model_id;
  return typeof fromSpec === "string" ? fromSpec : undefined;
}

export function StudioLibrarySection({
  clips,
  loading,
  error,
  reload,
  refresh,
  onUseAsSource,
  onContinueScene,
  selected,
  setSelected,
}: StudioLibrarySectionProps) {
  const [openRows, setOpenRows] = useState<Set<string>>(new Set());
  const [details, setDetails] = useState<Record<string, ClipDetail>>({});
  const [detailErr, setDetailErr] = useState<Record<string, string>>({});
  const [detailLoading, setDetailLoading] = useState<Set<string>>(new Set());

  const [showFilters, setShowFilters] = useState(false);
  const [statusFilter, setStatusFilter] = useState("all");
  const [capFilter, setCapFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("all");
  const [loadingAll, setLoadingAll] = useState(false);

  const [cancelBusy, setCancelBusy] = useState<Set<string>>(new Set());
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Lazy per-job detail fetch (cached after — never rides the 6s list poll).
  const fetchDetail = useCallback(
    async (jobId: string) => {
      if (details[jobId] || detailLoading.has(jobId)) return;
      setDetailLoading((prev) => new Set(prev).add(jobId));
      try {
        const res = await request<unknown>(studioClipDetailUrl(jobId), {
          meta: { specKey: "studio", operation: "studio.clip.detail" },
        });
        if (!mounted.current) return;
        if (!res.ok) {
          setDetailErr((p) => ({ ...p, [jobId]: describeAppError(errorOf(res)) }));
          return;
        }
        const parsed = clipDetailSchema.safeParse(okValue(res));
        if (!parsed.success) {
          setDetailErr((p) => ({ ...p, [jobId]: "Malformed clip detail response." }));
          return;
        }
        setDetails((p) => ({ ...p, [jobId]: parsed.data }));
        setDetailErr((p) => {
          const n = { ...p };
          delete n[jobId];
          return n;
        });
      } finally {
        if (mounted.current)
          setDetailLoading((prev) => {
            const n = new Set(prev);
            n.delete(jobId);
            return n;
          });
      }
    },
    [details, detailLoading],
  );

  // (Auto-select of the newest playable clip is LIFTED to the plane, which owns the
  //  shared `selected` so the viewer in the generator column can follow it.)

  // Failed/cancelled rows are the ones that most need explaining, so auto-load their
  // detail ONCE (targeted + cheap) — the error then shows inline without expanding.
  const autoFetched = useRef<Set<string>>(new Set());
  useEffect(() => {
    for (const c of clips) {
      const s = c.status ?? "";
      if (
        (s === "failed" || s === "cancelled") &&
        !autoFetched.current.has(c.job_id) &&
        !details[c.job_id] &&
        !detailErr[c.job_id]
      ) {
        autoFetched.current.add(c.job_id);
        void fetchDetail(c.job_id);
      }
    }
  }, [clips, details, detailErr, fetchDetail]);

  const toggleDetails = useCallback(
    (jobId: string) => {
      let willOpen = false;
      setOpenRows((prev) => {
        const next = new Set(prev);
        if (next.has(jobId)) next.delete(jobId);
        else {
          next.add(jobId);
          willOpen = true;
        }
        return next;
      });
      if (willOpen) void fetchDetail(jobId);
    },
    [fetchDetail],
  );

  const loadAllDetails = useCallback(async () => {
    setLoadingAll(true);
    try {
      for (const c of clips) {
        if (!details[c.job_id] && !detailErr[c.job_id]) {
          // eslint-disable-next-line no-await-in-loop
          await fetchDetail(c.job_id);
        }
      }
    } finally {
      if (mounted.current) setLoadingAll(false);
    }
  }, [clips, details, detailErr, fetchDetail]);

  const onCancel = useCallback(
    async (jobId: string) => {
      setCancelBusy((prev) => new Set(prev).add(jobId));
      try {
        await request<unknown>(jobCancelUrl(jobId), {
          method: "POST",
          meta: { specKey: "studio", operation: "studio.clip.cancel" },
        });
        if (!mounted.current) return;
        refresh();
      } finally {
        if (mounted.current)
          setCancelBusy((prev) => {
            const n = new Set(prev);
            n.delete(jobId);
            return n;
          });
      }
    },
    [refresh],
  );

  const onCopyPath = useCallback((jobId: string, uri: string | null | undefined) => {
    if (!uri) return;
    try {
      void navigator.clipboard?.writeText(uri);
      setCopiedId(jobId);
      window.setTimeout(() => {
        if (mounted.current) setCopiedId((c) => (c === jobId ? null : c));
      }, 1500);
    } catch {
      // clipboard unavailable — silently ignore (the path is still shown on expand)
    }
  }, []);

  // Filter option sets. Status is from the list; capability/model from loaded
  // details (the list projection omits them).
  const statuses = Array.from(
    new Set(clips.map((c) => c.status ?? "").filter(Boolean)),
  ).sort();
  const capabilities = Array.from(
    new Set(clips.map((c) => capOf(details[c.job_id])).filter(Boolean) as string[]),
  ).sort();
  const models = Array.from(
    new Set(clips.map((c) => modelOf(details[c.job_id])).filter(Boolean) as string[]),
  ).sort();

  const filtered = clips.filter((c) => {
    if (statusFilter !== "all" && (c.status ?? "") !== statusFilter) return false;
    if (capFilter !== "all" && capOf(details[c.job_id]) !== capFilter) return false;
    if (modelFilter !== "all" && modelOf(details[c.job_id]) !== modelFilter) return false;
    return true;
  });

  const filtersActive = capFilter !== "all" || modelFilter !== "all";
  const activeFilterCount =
    (statusFilter !== "all" ? 1 : 0) +
    (capFilter !== "all" ? 1 : 0) +
    (modelFilter !== "all" ? 1 : 0);

  return (
    <div className="vi-studio-library">
      <p className="vi-comfy-label" style={{ marginBottom: "0.4rem" }}>
        Clips
      </p>

      {/* ── compact, collapsible filters strip (not a separate tab) ── */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" }}>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          aria-expanded={showFilters}
          onClick={() => setShowFilters((v) => !v)}
        >
          {showFilters ? "▾ Filters" : "▸ Filters"}
          {activeFilterCount > 0 ? ` (${activeFilterCount})` : ""}
        </button>
        <button type="button" className="vi-btn vi-btn-sm vi-btn-ghost" onClick={reload}>
          Refresh
        </button>
        <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
          {filtered.length}/{clips.length} clips
        </span>
      </div>

      {showFilters && (
        <>
          <div
            className="vi-comfy-bar"
            style={{ flexWrap: "wrap", gap: "0.75rem", alignItems: "flex-end", marginTop: "0.5rem" }}
          >
            <label className="vi-comfy-field" style={{ maxWidth: "10rem" }}>
              <span className="vi-comfy-label">Status</span>
              <select
                className="vi-knob-input"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="all">all</option>
                {statuses.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <label className="vi-comfy-field" style={{ maxWidth: "10rem" }}>
              <span className="vi-comfy-label">Capability</span>
              <select
                className="vi-knob-input"
                value={capFilter}
                onChange={(e) => setCapFilter(e.target.value)}
              >
                <option value="all">all</option>
                {capabilities.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            <label className="vi-comfy-field" style={{ maxWidth: "14rem" }}>
              <span className="vi-comfy-label">Model</span>
              <select
                className="vi-knob-input"
                value={modelFilter}
                onChange={(e) => setModelFilter(e.target.value)}
              >
                <option value="all">all</option>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>
            <div className="vi-comfy-bar-actions">
              <button
                type="button"
                className="vi-btn vi-btn-ghost"
                onClick={loadAllDetails}
                disabled={loadingAll}
                title="Fetch every clip's detail so the Capability / Model filters cover all rows."
              >
                {loadingAll ? "Loading details…" : "Load all details"}
              </button>
            </div>
          </div>
          {filtersActive && (
            <p className="vi-comfy-hint" role="note" style={{ marginTop: "0.3rem" }}>
              Capability / Model filter over loaded details — use <strong>Load all details</strong>{" "}
              so every clip is covered.
            </p>
          )}
        </>
      )}

      {/* Clip list only — the player-forward viewer now lives in the GENERATOR column
          (StudioViewer), driven by the lifted `selected`. This is the Library TAB body. */}
      <ul
        className="vi-studio-clips-list"
        style={{
          listStyle: "none",
          margin: "1rem 0 0",
          padding: 0,
          maxHeight: "34rem",
          overflowY: "auto",
        }}
      >
          {loading && <li className="vi-comfy-hint">Loading clips…</li>}
          {!loading && error && (
            <li className="vi-error" role="alert">
              {error}
            </li>
          )}
          {!loading && !error && clips.length === 0 && (
            <li className="vi-comfy-hint">No studio clips yet — generate one above.</li>
          )}
          {!loading && !error && clips.length > 0 && filtered.length === 0 && (
            <li className="vi-comfy-hint">No clips match the current filters.</li>
          )}
          {filtered.map((c) => {
            const isSel = c.job_id === selected;
            const isOpen = openRows.has(c.job_id);
            const status = c.status ?? "";
            const isBad = status === "failed" || status === "cancelled";
            const d = details[c.job_id];
            const inlineErr = isBad && d?.error ? d.error : null;
            const cancellable = CANCELLABLE.has(status);
            return (
              <li key={c.job_id} style={{ marginBottom: "0.5rem" }}>
                <button
                  type="button"
                  className={isSel ? "vi-btn vi-btn-accent" : "vi-btn vi-btn-ghost"}
                  onClick={() => c.playable && setSelected(c.job_id)}
                  disabled={!c.playable}
                  title={c.job_id}
                  style={{
                    width: "100%",
                    justifyContent: "space-between",
                    display: "flex",
                    gap: "0.5rem",
                    opacity: c.playable ? 1 : 0.6,
                  }}
                >
                  <span>{shortId(c.job_id)}</span>
                  <span style={{ fontVariantNumeric: "tabular-nums", opacity: 0.8 }}>
                    {status}
                    {dims(c) ? ` · ${dims(c)}` : ""}
                    {whenText(c) ? ` · ${whenText(c)}` : ""}
                  </span>
                </button>

                {/* Prominent inline error for a failed/cancelled row (auto-loaded). */}
                {inlineErr && (
                  <div
                    className="vi-error"
                    role="alert"
                    style={{ margin: "0.25rem 0", fontSize: "0.82rem" }}
                  >
                    <strong>
                      {status === "cancelled" ? "Cancelled" : "Failed"} [
                      {inlineErr.code ?? "error"}]
                    </strong>
                    {inlineErr.message ? ` — ${inlineErr.message}` : ""}
                  </div>
                )}

                {/* per-clip actions */}
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem", marginTop: "0.2rem" }}>
                  {c.playable && (
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      onClick={() => setSelected(c.job_id)}
                    >
                      Play
                    </button>
                  )}
                  {c.output?.uri && (
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      onClick={() => onCopyPath(c.job_id, c.output?.uri)}
                    >
                      {copiedId === c.job_id ? "Copied" : "Copy path"}
                    </button>
                  )}
                  {c.playable && (
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      onClick={() => onUseAsSource(c)}
                      title="Prefill the generator above with this clip as the source (extend from its last frame). If you already have reference images, the identity is carried across."
                    >
                      Use as source
                    </button>
                  )}
                  {c.playable && (
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      onClick={() => onContinueScene(c)}
                      title="Continue this scene: stage the clip as a v2v scene-continuity source (restyle / extend the prior scene while HOLDING an identity). Add or keep 1–4 reference images to lock the subject across the cut."
                    >
                      Continue scene ↦
                    </button>
                  )}
                  {cancellable && (
                    <button
                      type="button"
                      className="vi-btn vi-btn-sm vi-btn-ghost"
                      onClick={() => void onCancel(c.job_id)}
                      disabled={cancelBusy.has(c.job_id)}
                    >
                      {cancelBusy.has(c.job_id) ? "Cancelling…" : "Cancel"}
                    </button>
                  )}
                  <button
                    type="button"
                    className="vi-btn vi-btn-sm vi-btn-ghost"
                    aria-expanded={isOpen}
                    onClick={() => toggleDetails(c.job_id)}
                    style={{ opacity: 0.85 }}
                  >
                    {isOpen ? "▾ Details" : "▸ Details"}
                  </button>
                </div>

                {isOpen && (
                  <ClipDetailsPanel
                    detail={details[c.job_id]}
                    loading={detailLoading.has(c.job_id)}
                    error={detailErr[c.job_id]}
                  />
                )}
              </li>
            );
          })}
      </ul>
    </div>
  );
}

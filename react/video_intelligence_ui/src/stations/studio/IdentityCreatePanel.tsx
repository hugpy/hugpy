// k94 — the Identities tab's ONE explicit path (the clownworld MO).
//
//   Create an identity
//   name: [__________]
//   [+ video]  [+ images]        ← inline expansion (upload · library pick) in the same row
//   ( Create identity )          ← ONE button; disabled until name + a source exist
//   progress: stage · % · log tail
//   > advanced                   ← everything else, collapsed, default-hidden
//
// Input: a video (or a few photos) of a character. Output: an identity profile with
// reference views + a 3D GLB, ready to bind from "+ identity". No knobs on the happy
// path — the pipeline answers every question it can answer itself (front view, mesh
// params, turntable, canonical promotion).
//
//   + video  → POST /video/identity-profiles/from-video  {source, name}
//              ONE chained char360 + Hunyuan3D job on the render service; on done ONE
//              profile per detected character (slug, slug-2, …).
//   + images → POST /video/identity-profiles/from-images {sources, name}
//              create profile + the full 3D build, chained server-side.
// Both hand back a job id watched by useIdentityCreateJob (GET /video/jobs/<id>).
import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { mediaBytesUrl } from "../../config";
import type { LibraryItem } from "../../video/mediaLibrary";
import type { MediaRef } from "../../video/contract";
import type { IdentityProfilesState } from "./useIdentityProfiles";
import { LibraryImageGrid } from "./studioShared";
import { uploadAndIngestImage, uploadAndIngestVideo } from "./identityUpload";
import { useIdentityCreateJob, describeCreateJob } from "./useIdentityCreateJob";
import type { CreatedIdentity } from "./useIdentityCreateJob";

// Mirrors the catalogue's MAX_SOURCE_IMAGES (identity_profiles.py).
const MAX_SOURCE_IMAGES = 12;

type SourceMode = "video" | "images";

/** The ONE-LINE explanation the tab keeps (k94 removed the option-explaining prose). */
export const ONE_LINE =
  "A video or a few photos in — a bindable character with reference views and a 3D model out.";

/**
 * The `> advanced` expander — the PromptCardSettings idiom (a native <details>,
 * the open flag lives in the DOM, never in state) with an `open` default so the
 * sidebar "settings" tab can lead with it expanded. Same CSS hooks as
 * PromptCardSettings (vi-prompt-card-settings*), so it renders identically.
 */
export function AdvancedExpander({
  label = "advanced",
  open = false,
  children,
}: {
  label?: string;
  open?: boolean;
  children: ReactNode;
}) {
  return (
    <details className="vi-prompt-card-settings" open={open}>
      <summary className="vi-prompt-card-settings-summary">{label}</summary>
      <div className="vi-prompt-card-settings-body">{children}</div>
    </details>
  );
}

function fileName(ref: MediaRef): string {
  return ref.uri.split("/").pop() || ref.asset_id;
}

/** A small library-VIDEO list — the video sibling of LibraryImageGrid (which
 *  renders <img> thumbnails, wrong for a clip). Each row is a plain <button>. */
function LibraryVideoList({
  videos,
  onPick,
}: {
  videos: LibraryItem[];
  onPick: (ref: MediaRef) => void;
}) {
  if (videos.length === 0) {
    return (
      <p className="vi-comfy-hint" style={{ margin: "0.4rem 0 0" }}>
        No clips in the session library yet — upload a video, or produce one in Scene/Movie.
      </p>
    );
  }
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", marginTop: "0.4rem", maxHeight: "9rem", overflowY: "auto" }}>
      {videos.map((it) => (
        <button
          key={it.ref.uri}
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          onClick={() => onPick(it.ref)}
          title={it.ref.uri}
          style={{ maxWidth: "16rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
        >
          ▶ {it.label ?? fileName(it.ref)}
          {it.ref.duration_s ? ` · ${it.ref.duration_s.toFixed(1)}s` : ""}
        </button>
      ))}
    </div>
  );
}

export function IdentityCreatePanel({
  createFromVideo,
  createFromImages,
  library,
  reload,
  onCreated,
  advanced,
  advancedOpen = false,
}: {
  createFromVideo: IdentityProfilesState["createFromVideo"];
  createFromImages: IdentityProfilesState["createFromImages"];
  library: LibraryItem[];
  reload: () => void;
  /** Fires when the job lands — `slugs` are the profiles now bindable. */
  onCreated: (slugs: string[]) => void;
  /** The `> advanced` body (legacy create panel, char360 knobs, …). */
  advanced?: ReactNode;
  advancedOpen?: boolean;
}) {
  const [name, setName] = useState("");
  const [mode, setMode] = useState<SourceMode | null>(null);
  const [video, setVideo] = useState<MediaRef | null>(null);
  const [images, setImages] = useState<MediaRef[]>([]);
  const [showLib, setShowLib] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [pickError, setPickError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const job = useIdentityCreateJob();
  const jobSlug = useRef<string | null>(null);

  const libraryVideos = library.filter((it) => it.ref.kind === "video");
  const libraryImages = library.filter((it) => it.ref.kind === "image");

  // A reload after the job lands so the new profile(s) + mesh show; done once.
  useEffect(() => {
    if (job.view.phase === "done" || job.view.phase === "error") {
      setBusy(false);
      reload();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.view.phase]);

  const toggleMode = (m: SourceMode) => {
    setPickError(null);
    setShowLib(false);
    setMode((cur) => (cur === m ? null : m));
  };

  const addImage = useCallback((ref: MediaRef) => {
    setImages((prev) =>
      prev.length >= MAX_SOURCE_IMAGES || prev.some((r) => r.uri === ref.uri) ? prev : [...prev, ref],
    );
  }, []);

  async function onPickVideoFile(file: File) {
    setPickError(null);
    setUploading(true);
    const { ref, error } = await uploadAndIngestVideo(file);
    setUploading(false);
    if (error) setPickError(error);
    else if (ref) setVideo(ref);
  }

  async function onPickImageFiles(files: FileList) {
    setPickError(null);
    setUploading(true);
    const list = Array.from(files as ArrayLike<File>).slice(0, Math.max(0, MAX_SOURCE_IMAGES - images.length));
    const results = await Promise.all(list.map((f) => uploadAndIngestImage(f)));
    setUploading(false);
    for (const r of results) {
      if (r.ref) addImage(r.ref);
      else if (r.error) setPickError(r.error);
    }
  }

  const hasSource = mode === "video" ? video != null : mode === "images" ? images.length > 0 : false;
  const canCreate = name.trim().length > 0 && hasSource && !busy && !uploading;

  async function create() {
    if (!canCreate || !mode) return;
    const trimmed = name.trim();
    setBusy(true);
    job.reset();
    const res =
      mode === "video" && video
        ? await createFromVideo(video, trimmed)
        : await createFromImages(images, trimmed);
    if (!res.ok || !res.jobId) {
      setBusy(false);
      job.fail(
        res.duplicate
          ? `An identity named "${trimmed}" already exists — choose another name.`
          : res.message || "Could not start the identity build.",
      );
      return;
    }
    jobSlug.current = res.slug ?? null;
    // The from-images profile exists already — make it selectable right away.
    if (mode === "images" && res.slug) onCreated([res.slug]);
    job.watch(res.jobId, (ids: CreatedIdentity[]) => {
      const slugs = ids.map((i) => i.slug);
      if (slugs.length === 0 && jobSlug.current) slugs.push(jobSlug.current);
      onCreated(slugs);
      // Clear the form for the next character; the progress line stays ("done").
      setName("");
      setVideo(null);
      setImages([]);
      setMode(null);
    });
  }

  const progress = describeCreateJob(job.view);
  const createdCount = job.view.identities.length;

  return (
    <section className="vi-comfy-bar vi-id-create" aria-label="Create an identity"
      style={{ display: "flex", flexDirection: "column", gap: "0.5rem", alignItems: "stretch" }}>
      <div>
        <span className="vi-comfy-label" style={{ fontSize: "0.95rem" }}>Create an identity</span>
        <p className="vi-comfy-hint" style={{ margin: "0.15rem 0 0", opacity: 0.8 }}>{ONE_LINE}</p>
      </div>

      {/* name + [+ video] [+ images] — one row */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" }}>
        <label className="vi-comfy-field" style={{ flex: "1 1 14rem", maxWidth: "22rem" }}>
          <span className="vi-comfy-label">name</span>
          <input
            type="text"
            className="vi-knob-input"
            value={name}
            placeholder="e.g. Mira"
            disabled={busy}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <button
          type="button"
          className={`vi-btn vi-btn-sm ${mode === "video" ? "vi-btn-accent" : "vi-btn-ghost"}`}
          aria-expanded={mode === "video"}
          disabled={busy}
          onClick={() => toggleMode("video")}
          title="A video of the character — every person in it becomes an identity"
        >
          + video{video && mode !== "video" ? " ✓" : ""}
        </button>
        <button
          type="button"
          className={`vi-btn vi-btn-sm ${mode === "images" ? "vi-btn-accent" : "vi-btn-ghost"}`}
          aria-expanded={mode === "images"}
          disabled={busy}
          onClick={() => toggleMode("images")}
          title="A few photos of the character (up to 12)"
        >
          + images{images.length && mode !== "images" ? ` (${images.length})` : ""}
        </button>
      </div>

      {/* inline expansion — the SAME row, under the buttons (k93 idiom) */}
      {mode === "video" && (
        <div style={{ border: "1px solid var(--vi-border)", borderRadius: "0.4rem", padding: "0.5rem" }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" }}>
            <label className="vi-btn vi-btn-sm vi-btn-accent vi-file-label">
              {uploading ? "Uploading…" : video ? "Change video" : "Upload a video"}
              <input
                type="file"
                accept="video/*"
                className="vi-file-input"
                disabled={busy || uploading}
                onChange={(e) => {
                  const file = e.target.files && e.target.files[0];
                  if (file) void onPickVideoFile(file);
                  e.target.value = "";
                }}
              />
            </label>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              disabled={busy}
              aria-expanded={showLib}
              onClick={() => setShowLib((v) => !v)}
            >
              {showLib ? "Hide library" : `Pick from library (${libraryVideos.length})`}
            </button>
            {video && (
              <span className="vi-comfy-hint" style={{ opacity: 0.9, display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
                ▶ {fileName(video)}
                {video.duration_s ? ` · ${video.duration_s.toFixed(1)}s` : ""}
                <button type="button" className="vi-btn vi-btn-sm vi-btn-ghost" disabled={busy}
                  onClick={() => setVideo(null)} title="Remove this video" style={{ padding: "0 0.3rem" }}>✕</button>
              </span>
            )}
          </div>
          {showLib && <LibraryVideoList videos={libraryVideos} onPick={(ref) => { setVideo(ref); setShowLib(false); }} />}
        </div>
      )}

      {mode === "images" && (
        <div style={{ border: "1px solid var(--vi-border)", borderRadius: "0.4rem", padding: "0.5rem" }}>
          {images.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "0.4rem" }}>
              {images.map((ref, idx) => (
                <div key={`${ref.uri}_${idx}`} style={{ position: "relative" }}>
                  <img src={mediaBytesUrl(ref.uri)} alt={`photo ${idx + 1}`}
                    style={{ width: "3.6rem", height: "3.6rem", objectFit: "cover", borderRadius: "0.35rem", background: "#000", border: "1px solid var(--vi-border)" }} />
                  <button type="button" className="vi-btn vi-btn-sm vi-btn-ghost" disabled={busy}
                    onClick={() => setImages((prev) => prev.filter((_, i) => i !== idx))}
                    title="Remove this photo"
                    style={{ position: "absolute", top: -6, right: -6, padding: "0 0.3rem", fontSize: "0.7rem", lineHeight: 1.4, background: "var(--vi-border)", borderRadius: "0.3rem" }}>
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" }}>
            <label className="vi-btn vi-btn-sm vi-btn-accent vi-file-label">
              {uploading ? "Uploading…" : images.length ? "Add photos" : "Upload photos"}
              <input
                type="file"
                accept="image/*"
                multiple
                className="vi-file-input"
                disabled={busy || uploading || images.length >= MAX_SOURCE_IMAGES}
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) void onPickImageFiles(e.target.files);
                  e.target.value = "";
                }}
              />
            </label>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              disabled={busy || images.length >= MAX_SOURCE_IMAGES}
              aria-expanded={showLib}
              onClick={() => setShowLib((v) => !v)}
            >
              {showLib ? "Hide library" : `Pick from library (${libraryImages.length})`}
            </button>
            <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>{images.length}/{MAX_SOURCE_IMAGES}</span>
          </div>
          {showLib && images.length < MAX_SOURCE_IMAGES && (
            <LibraryImageGrid images={libraryImages} onPick={addImage} />
          )}
        </div>
      )}

      {pickError && (
        <p className="vi-error" role="alert" style={{ margin: 0 }}>{pickError}</p>
      )}

      {/* ONE button + the progress line */}
      <div className="vi-comfy-bar-actions vi-run-bar" style={{ flexWrap: "wrap", alignItems: "center" }}>
        <button
          type="button"
          className="vi-btn vi-btn-accent"
          disabled={!canCreate}
          onClick={() => void create()}
          title={
            !name.trim()
              ? "Give the identity a name"
              : !hasSource
                ? "Add a video or a few photos"
                : "Build the identity: reference views + 3D model"
          }
        >
          {busy ? "Creating…" : "Create identity"}
        </button>
        {progress && (
          <span
            className={job.view.phase === "error" ? "vi-error" : "vi-comfy-hint"}
            role="status"
            style={{ opacity: 0.9, fontVariantNumeric: "tabular-nums", overflow: "hidden", textOverflow: "ellipsis" }}
            title={job.view.logLine ?? undefined}
          >
            {job.view.phase === "done" && createdCount > 0
              ? `done · ${createdCount} identit${createdCount === 1 ? "y" : "ies"}: ${job.view.identities
                  .map((i) => `${i.slug}${i.glb ? " ✓" : i.error ? " (views only)" : ""}`)
                  .join(", ")}`
              : progress}
          </span>
        )}
      </div>

      {advanced && <AdvancedExpander open={advancedOpen}>{advanced}</AdvancedExpander>}
    </section>
  );
}

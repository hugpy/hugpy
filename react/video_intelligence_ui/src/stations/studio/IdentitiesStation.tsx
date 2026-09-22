// IDENTITIES — the dedicated station for PROCURING and EDITING identity profiles
// (studio stage (a)'s durable library item), separate from the compact "👤 From
// profile" picker the Studio Generate/Movie surfaces embed. Those surfaces only
// ATTACH a saved identity to a render; this station is where one is actually
// built and maintained.
//
// k94 — ONE PATH (the clownworld MO): the top of the tab is IdentityCreatePanel —
// name + [+ video] [+ images] + ONE "Create identity" button + a progress line.
// A video (or a few photos) in; a bindable character (reference views + 3D GLB)
// out. Everything else this tab used to expose as separate steps (the manual
// create panel, char360 knobs, pose / vision model / frames / fps / size,
// turnaround prompt, seed, regeneration prompt, the per-step generate / approve /
// new-version buttons) lives under ONE collapsed `> advanced` expander, still
// functional, default-hidden. The profile list + viewer (canonical views,
// turntable, GLB) stay, with a ✓ when an identity is ready to bind.
//
// Backed entirely by useIdentityProfiles (list/create/update/remove) — this file
// adds no new wire contract, just the surface. The reference-image upload path
// is the EXACT upload→ingest→guard-kind flow StudioGenerateTab's onPickImageFile
// / StudioMovieComposer's run (POST /uploads → POST /video/ingest → MediaRef);
// the library pick reuses studioShared's LibraryImageGrid + useMediaLibrary,
// unmodified — both imported, neither file touched.
//
// USE-IN-STUDIO (operator ask: "a tab for identities, dedicated to procuring
// them, editing them, etc"): a per-card "Use in Studio" jump is SKIPPED here — a
// clean navigation/state handoff into the Studio station's bound-profile state
// would mean reaching into the Studio plane/bridge files a concurrent change was
// touching at build time. The generate surfaces already offer "👤 From profile"
// (IdentityProfileControls) as the attach point, so nothing is unreachable in
// the meantime — just not a one-click deep link from here yet. Follow-up, not a
// gap in capability.
import { useCallback, useState, useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import { mediaBytesUrl, identityGenerateUrl, identityMeshStatusUrl, videoJobUrl } from "../../config";
import { useMediaLibrary } from "../../video/mediaLibrary";
import type { LibraryItem } from "../../video/mediaLibrary";
import type { MediaRef, IdentityReconstruction, IdentityAngleView } from "../../video/contract";
import { LibraryImageGrid } from "./studioShared";
import { uploadAndIngestImage, uploadAndIngestVideo } from "./identityUpload";
import { IdentityCreatePanel, AdvancedExpander } from "./IdentityCreatePanel";
import { useIdentityProfiles } from "./useIdentityProfiles";
import type {
  IdentityProfile,
  IdentityProfilesState,
  GenSettings,
  IdentityVersion,
  Char360Params,
} from "./useIdentityProfiles";
import { useModelsByTask } from "../../video/useModels";
import { useReconstruction, DEFAULT_VIEWS, VIEW_LABELS } from "./useReconstruction";
import type { ReconstructionApi } from "./useReconstruction";
import { TurntableViewer } from "./TurntableViewer";
import { viewLabel } from "../GenIdentityBar";
import type { StationSpec } from "../types";

// The VL capability the per-identity vision-model picker filters the registry by — the
// SAME task the Movie director's judge-VLM select uses (see GenerateStation). The
// 3D-imaging front-select step is an image->text ask, so only image-text-to-text
// models are offered.
const VISION_TASK = "image-text-to-text";

// Increased from 4 to 12. Canonical anchors stay at 4 (handled via promote limits)
const MAX_SOURCE_IMAGES = 12;

type UploadPhase = "idle" | "uploading" | "ingesting";

// uploadAndIngestImage / uploadAndIngestVideo moved to ./identityUpload (k94) so the
// one-path create panel shares them without a circular import — same behavior.

/** "Ready to bind" (k94): views (a canonical set, or any reconstruction with views)
 *  AND a finished mesh (some reconstruction's mesh.status === "done" + glb_path).
 *  Reads the same snake_case wire truth the detail's viewers gate on. */
export function identityReadiness(p: IdentityProfile): { views: boolean; mesh: boolean; ready: boolean } {
  const recons = p.reconstructions ?? [];
  const views =
    (p.canonical ?? []).length > 0 ||
    recons.some((r) => Array.isArray(r.views) && (r.views as unknown[]).length > 0);
  const mesh = recons.some((r) => {
    const m = (r as { mesh?: { status?: string; glb_path?: string | null } | null }).mesh;
    return !!m && m.status === "done" && !!m.glb_path;
  });
  return { views, mesh, ready: views && mesh };
}

function fmtCreated(t: number | null | undefined): { short: string; full: string } {
  if (t == null) return { short: "", full: "" };
  try {
    const d = new Date(t * 1000);
    return { short: d.toLocaleDateString(), full: d.toLocaleString() };
  } catch {
    return { short: "", full: "" };
  }
}

// ── GLB Mesh Viewer (Phase 4) ────────────────────────────────────────────────
export function MeshViewer({ url, height = 420 }: { url: string; height?: number }) {
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(35, host.clientWidth / height, 0.01, 1000);
    camera.position.set(0, 1.2, 3);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(host.clientWidth, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;

    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 1, 0);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3);
    keyLight.position.set(3, 5, 4);
    scene.add(keyLight);

    const loader = new GLTFLoader();
    let loadedRoot: THREE.Object3D | null = null;
    let animationFrame = 0;

    loader.load(
      url,
      (gltf) => {
        loadedRoot = gltf.scene;
        scene.add(gltf.scene);

        const box = new THREE.Box3().setFromObject(gltf.scene);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        gltf.scene.position.sub(center);

        const largest = Math.max(size.x, size.y, size.z, 0.001);
        camera.position.set(0, size.y * 0.15, largest * 2.2);
        camera.near = largest / 100;
        camera.far = largest * 100;
        camera.updateProjectionMatrix();

        controls.target.set(0, 0, 0);
        controls.update();
      },
      undefined,
      (error) => console.error("Could not load identity GLB", error)
    );

    const resize = () => {
      const width = host.clientWidth;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(host);

    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(render);
    };
    render();

    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();

      if (loadedRoot) {
        loadedRoot.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return;
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          for (const material of materials) {
            for (const value of Object.values(material)) {
              if (value instanceof THREE.Texture) value.dispose();
            }
            material.dispose();
          }
        });
      }
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [url, height]);

  return (
    <div
      ref={hostRef}
      style={{
        width: "100%",
        height,
        overflow: "hidden",
        borderRadius: "0.5rem",
        border: "1px solid var(--vi-border)",
      }}
    />
  );
}

// ── Angle Ring Editor (Phase 3) ─────────────────────────────────────────────
function AngleRingTile({
  view,
  reconId,
  api,
  promptOverride,
}: {
  view: IdentityAngleView;
  reconId: string;
  api: ReconstructionApi;
  promptOverride: string;
}) {
  const isGenerating = view.status === "pending" || view.status === "generating";
  const hasImage = view.image_uri != null; // <--- snake_case

  return (
    <div
      className={`vi-scene-frame-cell vi-angle-tile vi-angle-tile-${view.status}`}
      style={{ display: "flex", flexDirection: "column", gap: "0.2rem", alignItems: "center", width: "5rem" }}
    >
      <div
        style={{
          position: "relative",
          width: "5rem",
          height: "5rem",
          borderRadius: "0.35rem",
          overflow: "hidden",
          background: "#000",
          border: view.status === "approved" ? "2px solid var(--vi-accent, #6aa8ff)" : "1px solid var(--vi-border)",
        }}
      >
        {hasImage ? (
          <img
            className="vi-scene-frame-img"
            src={mediaBytesUrl(view.image_uri!)} // <--- snake_case
            alt={`${view.azimuth_deg}° view`} // <--- snake_case
            loading="lazy"
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <span
            className="vi-comfy-hint"
            style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.6rem", opacity: 0.6 }}
          >
            {isGenerating ? "Generating…" : "No Image"}
          </span>
        )}
        <span
          className="vi-frame-tag"
          style={{ position: "absolute", bottom: 2, left: 3, fontSize: "0.75rem", textShadow: "0 0 3px #000" }}
        >
          {view.azimuth_deg}°
        </span>
      </div>

      <span className="vi-comfy-hint" style={{ fontSize: "0.6rem", opacity: 0.85, lineHeight: 1.1 }}>
        {view.status || "pending"} {view.version ? `(v${view.version})` : ""}
      </span>

      <div style={{ display: "flex", gap: "0.2rem", flexWrap: "wrap", justifyContent: "center" }}>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          disabled={!hasImage || view.status === "approved" || isGenerating}
          onClick={() => void api.setViewStatus(reconId, view.view_id, "approved")} // <--- snake_case
          title="Approve this angle"
          style={{ fontSize: "0.65rem", padding: "0 0.35rem" }}
        >
          ✓
        </button>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          disabled={!hasImage || view.status === "rejected" || isGenerating}
          onClick={() => void api.setViewStatus(reconId, view.view_id, "rejected")} // <--- snake_case
          title="Reject this angle"
          style={{ fontSize: "0.65rem", padding: "0 0.35rem" }}
        >
          ✕
        </button>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          disabled={isGenerating}
          onClick={() => void api.regenerateView(reconId, view.view_id, promptOverride, undefined, true)} // <--- snake_case
          title="Regenerate using nearest approved neighbors"
          style={{ fontSize: "0.65rem", padding: "0 0.35rem", flexBasis: "100%" }}
        >
          ↻ Regen
        </button>
      </div>
    </div>
  );
}

function AngleRingEditor({ reconstruction, api }: { reconstruction: IdentityReconstruction; api: ReconstructionApi }) {
  const [regeneratePrompt, setRegeneratePrompt] = useState("");

  const views = reconstruction.views ?? [];
  const approvedCount = views.filter((v: any) => v.status === "approved").length;
  const allViewsApproved = views.length > 0 && approvedCount === views.length;
  const meshRunning = reconstruction.mesh?.status === "queued" || reconstruction.mesh?.status === "running";

  const getMeshAnchorIds = () => {
    const anchorDegrees = [0, 45, 90, 135, 180, 225, 270, 315];
    return views.filter((v: any) => anchorDegrees.includes(v.azimuth_deg)).map((v: any) => v.view_id); // <--- snake_case
  };

  return (
    <div style={{ marginTop: "0.4rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.4rem" }}>
        <p className="vi-comfy-label" style={{ margin: 0 }}>
          Angle Ring Approval ({approvedCount}/{views.length})
        </p>
        <button
          className="vi-btn vi-btn-sm vi-btn-accent"
          disabled={!allViewsApproved || meshRunning || api.running}
          onClick={() => api.buildMesh(reconstruction.recon_id, getMeshAnchorIds())} // <--- snake_case
        >
          {meshRunning ? "Building 3D Mesh…" : "Build Approved 3D Model"}
        </button>
      </div>

      {reconstruction.mesh?.error && (
        <p className="vi-error" style={{ marginBottom: "0.4rem" }}>Mesh Error: {reconstruction.mesh.error}</p>
      )}

      <label className="vi-comfy-field" style={{ marginBottom: "0.6rem" }}>
        <span className="vi-comfy-label">Global Regeneration Prompt</span>
        <input
          type="text"
          className="vi-knob-input"
          placeholder="e.g. Preserve exact identity, clothing, and proportions..."
          value={regeneratePrompt}
          onChange={(e) => setRegeneratePrompt(e.target.value)}
        />
      </label>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem", maxHeight: "28rem", overflowY: "auto", paddingRight: "0.5rem" }}>
        {views.map((view: any) => (
          <div key={view.view_id} style={{ display: "contents" }}>
            <AngleRingTile
              view={view}
              reconId={reconstruction.recon_id} // <--- snake_case
              api={api}
              promptOverride={regeneratePrompt}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

// ── a small reusable "N/12 thumbnails + add (upload | library)" block ───────
function ReferenceEditor({
  refs,
  onAdd,
  onRemove,
  libraryImages,
  busy,
}: {
  refs: string[];
  onAdd: (ref: MediaRef) => void;
  onRemove: (idx: number) => void;
  libraryImages: LibraryItem[];
  busy: boolean;
}) {
  const [showLib, setShowLib] = useState(false);
  const canAdd = refs.length < MAX_SOURCE_IMAGES;

  return (
    <div style={{ marginTop: "0.4rem" }}>
      {refs.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "0.4rem" }}>
          {refs.map((uri, idx) => (
            <div key={`${uri}_${idx}`} style={{ position: "relative" }}>
              <img
                src={mediaBytesUrl(uri)}
                alt={`reference ${idx + 1}`}
                style={{
                  width: "3.6rem",
                  height: "3.6rem",
                  objectFit: "cover",
                  borderRadius: "0.35rem",
                  background: "#000",
                  border: "1px solid var(--vi-border)",
                }}
              />
              <button
                type="button"
                className="vi-btn vi-btn-sm vi-btn-ghost"
                disabled={busy}
                onClick={() => onRemove(idx)}
                title="Remove this reference"
                style={{
                  position: "absolute",
                  top: -6,
                  right: -6,
                  padding: "0 0.3rem",
                  fontSize: "0.7rem",
                  lineHeight: 1.4,
                  background: "var(--vi-border)",
                  borderRadius: "0.3rem",
                }}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
      {canAdd ? (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
          <label className="vi-btn vi-btn-sm vi-btn-accent vi-file-label">
            {busy ? "Working…" : refs.length ? "Add references" : "Upload references"}
            <input
              type="file"
              accept="image/*"
              multiple // 1. Allow selecting multiple files at once
              className="vi-file-input"
              disabled={busy}
              onChange={(e) => {
                const files = e.target.files;
                if (files && files.length > 0) {
                  void (async () => {
                    // Convert FileList to Array and enforce space limits
                    const fileArray = Array.from(files as ArrayLike<File>);
                    const remainingSlots = MAX_SOURCE_IMAGES - refs.length;
                    
                    if (fileArray.length > remainingSlots) {
                      window.alert(`You can only add ${remainingSlots} more image(s). Only the first ${remainingSlots} will be uploaded.`);
                    }
                    
                    const filesToUpload = fileArray.slice(0, remainingSlots);
                    
                    // 2. Upload and ingest them concurrently
                    await Promise.all(
                      filesToUpload.map(async (f) => {
                        const { ref, error } = await uploadAndIngestImage(f);
                        if (ref) onAdd(ref);
                        if (error) console.error(error);
                      })
                    );
                  })();
                }
                e.target.value = "";
              }}
            />
          </label>
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={busy}
            onClick={() => setShowLib((v) => !v)}
            aria-expanded={showLib}
          >
            {showLib ? "Hide library" : `Pick from library (${libraryImages.length})`}
          </button>
          <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
            {refs.length}/{MAX_SOURCE_IMAGES}
          </span>
        </div>
      ) : (
        <p className="vi-comfy-hint" role="note">
          Maximum {MAX_SOURCE_IMAGES} source reference images.
        </p>
      )}
      {showLib && canAdd && (
        <LibraryImageGrid
          images={libraryImages}
          onPick={(ref) => {
            onAdd(ref);
          }}
        />
      )}
    </div>
  );
}

// ── char360 S4 — "Extract from video" panel ─────────────────────────────────
// Mints (target:"create") or adds views to (target:"<slug>") an identity
// profile from a source video's char360 view-set extraction. A header-area
// peer of CreatePanel, not a per-card control — a "create" extract mints NEW
// profiles, so it doesn't belong to any one existing card.
//
// t25: `presetVideo` lets the Frames & Models station reuse this WHOLESALE over
// its ALREADY-LOADED source video — when provided, the built-in uploader is
// hidden and the extraction runs against that video. Omitted (the identities
// door) keeps the self-contained pick-a-video behavior exactly as before.
export function ExtractFromVideoPanel({
  profiles,
  extractFromVideo,
  reload,
  presetVideo,
}: {
  profiles: IdentityProfile[];
  extractFromVideo: IdentityProfilesState["extractFromVideo"];
  reload: () => void;
  presetVideo?: MediaRef | null;
}) {
  const [pickedVideo, setPickedVideo] = useState<MediaRef | null>(null);
  // When a preset video is supplied (Frames station), it IS the source — the
  // panel never runs its own uploader. Otherwise the locally-picked video is used.
  const usingPreset = presetVideo != null;
  const video = usingPreset ? presetVideo : pickedVideo;
  const setVideo = setPickedVideo;
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [target, setTarget] = useState<string>("create");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [stride, setStride] = useState("");
  const [yoloModel, setYoloModel] = useState("");
  const [minHFrac, setMinHFrac] = useState("");
  const [clusterDist, setClusterDist] = useState("");
  const [minFaces, setMinFaces] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [pickError, setPickError] = useState<string | null>(null);

  const active = useRef(false);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      active.current = false;
      if (timer.current != null) window.clearTimeout(timer.current);
    };
  }, []);

  async function onPickFile(file: File) {
    setPickError(null);
    setUploadPhase("uploading");
    const { ref, error } = await uploadAndIngestVideo(file);
    setUploadPhase("ingesting");
    if (error) {
      setPickError(error);
      setVideo(null);
    } else if (ref) {
      setVideo(ref);
    }
    setUploadPhase("idle");
  }

  async function submit() {
    if (!video || busy) return;
    setBusy(true);
    setStatus("Extracting… (queued)");
    active.current = true;

    const params: Char360Params = {};
    if (stride.trim()) params.stride = Number(stride);
    if (yoloModel.trim()) params.yolo_model = yoloModel.trim();
    if (minHFrac.trim()) params.min_h_frac = Number(minHFrac);
    if (clusterDist.trim()) params.cluster_dist = Number(clusterDist);
    if (minFaces.trim()) params.min_faces = Number(minFaces);

    const enq = await extractFromVideo(video, target, Object.keys(params).length ? params : undefined);
    if (!active.current) return;
    if (!enq.ok || !enq.jobId) {
      active.current = false;
      setBusy(false);
      setStatus(enq.message || "Could not start the video extraction.");
      return;
    }

    const jobId = enq.jobId;
    const deadline = Date.now() + 20 * 60 * 1000; // ~20 min bound, mirrors the /generate poll
    const pollOnce = async () => {
      if (!active.current) return;
      if (Date.now() > deadline) {
        active.current = false;
        setBusy(false);
        setStatus("Timed out waiting for the video extraction.");
        return;
      }
      const res = await request<unknown>(videoJobUrl(jobId), {
        meta: { specKey: "studio", operation: "identity.video_extract.status" },
      });
      if (!active.current) return;
      if (res.ok) {
        const job = okValue(res) as {
          progress?: number | null;
          result?: { ok?: boolean; error?: { code?: string; message?: string; retryable?: boolean } } | null;
          stage?: string | null;
          message?: string | null;
        };
        const result = job.result;
        if (result && typeof result.ok === "boolean") {
          active.current = false;
          setBusy(false);
          if (result.ok) {
            setStatus("done");
            setVideo(null);
            reload();
          } else {
            setStatus((result.error && result.error.message) || "Extraction failed.");
          }
          return;
        }
        const pct = typeof job.progress === "number" ? ` (${Math.round(job.progress * 100)}%)` : "";
        const stage = job.stage || job.message;
        setStatus(`Extracting…${pct}${stage ? ` — ${stage}` : ""}`);
      }
      // queued/running/transient-poll-failure → keep polling.
      timer.current = window.setTimeout(() => void pollOnce(), 5000);
    };
    timer.current = window.setTimeout(() => void pollOnce(), 5000);
  }

  const canSubmit = video != null && !busy && uploadPhase === "idle";

  return (
    <section className="vi-comfy-bar" aria-label="Extract character views from a video" style={{ marginTop: "0.8rem" }}>
      <div style={{ flexBasis: "100%" }}>
        <span className="vi-comfy-label">Extract character views (char360)</span>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem", alignItems: "center" }}>
        {!usingPreset && (
          <label className="vi-btn vi-btn-sm vi-btn-accent vi-file-label">
            {uploadPhase !== "idle" ? "Uploading…" : video ? "Change video" : "Pick a video"}
            <input
              type="file"
              accept="video/*"
              className="vi-file-input"
              disabled={busy || uploadPhase !== "idle"}
              onChange={(e) => {
                const file = e.target.files && e.target.files[0];
                if (file) void onPickFile(file);
                e.target.value = "";
              }}
            />
          </label>
        )}
        {video && (
          <span className="vi-comfy-hint" style={{ opacity: 0.85 }}>
            {usingPreset ? "loaded video · " : ""}
            {video.uri.split("/").pop() || video.asset_id}
            {video.duration_s ? ` · ${video.duration_s.toFixed(1)}s` : ""}
          </span>
        )}
        {usingPreset && !video && (
          <span className="vi-comfy-hint" style={{ opacity: 0.7 }}>
            Load a video above to extract a character view-set from it.
          </span>
        )}
      </div>
      {pickError && (
        <p className="vi-error" role="alert" style={{ flexBasis: "100%", margin: 0 }}>
          {pickError}
        </p>
      )}

      <label className="vi-comfy-field">
        <span className="vi-comfy-label">Target</span>
        <select
          className="vi-knob-input"
          value={target}
          disabled={busy}
          onChange={(e) => setTarget(e.target.value)}
        >
          <option value="create">Create new profile(s)</option>
          {profiles.map((p) => (
            <option key={p.slug} value={p.slug}>
              Add to: {p.name}
            </option>
          ))}
        </select>
      </label>

      <div className="vi-comfy-bar-actions vi-run-bar" style={{ flexWrap: "wrap" }}>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          onClick={() => setAdvancedOpen((v) => !v)}
          aria-expanded={advancedOpen}
          title="char360 extraction knobs — blank means the service default"
        >
          {advancedOpen ? "Advanced ▾" : "Advanced ▸"}
        </button>
        <button type="button" className="vi-btn vi-btn-accent" disabled={!canSubmit} onClick={() => void submit()}>
          {busy ? "Extracting…" : "Extract from video"}
        </button>
        {status && (
          <p className="vi-comfy-hint" role="status" style={{ flexBasis: "100%", margin: 0 }}>
            {status}
          </p>
        )}
      </div>

      {advancedOpen && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem", flexBasis: "100%" }}>
          <label className="vi-comfy-field" style={{ flexBasis: "7rem" }}>
            <span className="vi-comfy-label">Stride</span>
            <input
              type="number"
              className="vi-knob-input"
              min={1}
              placeholder="8"
              value={stride}
              disabled={busy}
              onChange={(e) => setStride(e.target.value)}
            />
          </label>
          <label className="vi-comfy-field" style={{ flexBasis: "10rem" }}>
            <span className="vi-comfy-label">YOLO model</span>
            <input
              type="text"
              className="vi-knob-input"
              placeholder="service default"
              value={yoloModel}
              disabled={busy}
              onChange={(e) => setYoloModel(e.target.value)}
            />
          </label>
          <label className="vi-comfy-field" style={{ flexBasis: "8rem" }}>
            <span className="vi-comfy-label">Min height frac</span>
            <input
              type="number"
              className="vi-knob-input"
              step="0.01"
              min={0}
              max={1}
              placeholder="default"
              value={minHFrac}
              disabled={busy}
              onChange={(e) => setMinHFrac(e.target.value)}
            />
          </label>
          <label className="vi-comfy-field" style={{ flexBasis: "8rem" }}>
            <span className="vi-comfy-label">Cluster dist</span>
            <input
              type="number"
              className="vi-knob-input"
              step="0.01"
              min={0}
              placeholder="default"
              value={clusterDist}
              disabled={busy}
              onChange={(e) => setClusterDist(e.target.value)}
            />
          </label>
          <label className="vi-comfy-field" style={{ flexBasis: "7rem" }}>
            <span className="vi-comfy-label">Min faces</span>
            <input
              type="number"
              className="vi-knob-input"
              min={0}
              placeholder="default"
              value={minFaces}
              disabled={busy}
              onChange={(e) => setMinFaces(e.target.value)}
            />
          </label>
        </div>
      )}
    </section>
  );
}

// ── CREATE panel ──────────────────────────────────────────────────────────────
function CreatePanel({
  create,
  libraryImages,
}: {
  create: IdentityProfilesState["create"];
  libraryImages: LibraryItem[];
}) {
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");
  const [refs, setRefs] = useState<MediaRef[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const addRef = useCallback((ref: MediaRef) => {
    setRefs((prev) =>
      prev.length >= MAX_SOURCE_IMAGES || prev.some((r) => r.uri === ref.uri) ? prev : [...prev, ref],
    );
  }, []);
  const removeRef = useCallback((idx: number) => {
    setRefs((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  async function submit() {
    const trimmed = name.trim();
    if (!trimmed) {
      setMsg("A name is required.");
      return;
    }
    if (refs.length === 0) {
      setMsg("At least one reference image is required — an identity keeps ≥1.");
      return;
    }
    setBusy(true);
    setMsg(null);
    const res = await create(
      trimmed,
      refs.map((r) => r.uri),
      notes.trim(),
    );
    setBusy(false);
    if (res.ok) {
      setName("");
      setNotes("");
      setRefs([]);
      setMsg(`Created identity profile "${res.profile ? res.profile.name : trimmed}".`);
    } else if (res.duplicate) {
      setMsg(`An identity profile named "${trimmed}" already exists — choose another name.`);
    } else {
      setMsg(res.message || "Could not create the identity profile.");
    }
  }

  const canSubmit = name.trim().length > 0 && refs.length > 0 && !busy;

  return (
    <section className="vi-comfy-bar" aria-label="Create an identity profile" style={{ marginTop: "0.8rem" }}>
      <label className="vi-comfy-field">
        <span className="vi-comfy-label">Name</span>
        <input
          type="text"
          className="vi-knob-input"
          value={name}
          placeholder="e.g. Mira"
          disabled={busy}
          onChange={(e) => setName(e.target.value)}
        />
      </label>
      <label className="vi-comfy-field" style={{ flexBasis: "24rem" }}>
        <span className="vi-comfy-label">Notes (optional)</span>
        <textarea
          className="vi-knob-input"
          rows={1}
          value={notes}
          placeholder="what makes this identity, continuity notes, etc."
          disabled={busy}
          onChange={(e) => setNotes(e.target.value)}
        />
      </label>
      <div style={{ flex: "1 1 100%" }}>
        <span className="vi-comfy-label">Reference images</span>
        <ReferenceEditor refs={refs.map((r) => r.uri)} onAdd={addRef} onRemove={removeRef} libraryImages={libraryImages} busy={busy} />
      </div>
      <div className="vi-comfy-bar-actions vi-run-bar" style={{ flexWrap: "wrap" }}>
        <button type="button" className="vi-btn vi-btn-accent" disabled={!canSubmit} onClick={() => void submit()}>
          {busy ? "Creating…" : "Create identity profile"}
        </button>
        {msg && (
          <p className="vi-comfy-hint" role="status" style={{ flexBasis: "100%", margin: 0 }}>
            {msg}
          </p>
        )}
      </div>
    </section>
  );
}

// ── one generated turnaround VIEW tile (Legacy mode) ────────────────────────
const RECON_TILE = "4.6rem";
function ReconViewTile({
  relPath,
  label,
  canonical,
  selected,
  busy,
  onToggle,
  onRegenerate,
}: {
  relPath: string;
  label: string;
  canonical: boolean;
  selected: boolean;
  busy: boolean;
  onToggle: () => void;
  onRegenerate: () => void;
}) {
  const [state, setState] = useState<"loading" | "ok" | "error">("loading");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.2rem", alignItems: "center", width: RECON_TILE }}>
      <div
        style={{
          position: "relative",
          width: RECON_TILE,
          height: RECON_TILE,
          borderRadius: "0.35rem",
          overflow: "hidden",
          background: "#000",
          border: selected ? "2px solid var(--vi-accent, #6aa8ff)" : "1px solid var(--vi-border)",
        }}
      >
        {state !== "error" && (
          <img
            src={mediaBytesUrl(relPath)}
            alt={label}
            onLoad={() => setState("ok")}
            onError={() => setState("error")}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              opacity: state === "ok" ? 1 : 0,
              transition: "opacity 0.2s ease",
            }}
          />
        )}
        {state === "loading" && (
          <span
            className="vi-comfy-hint"
            style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.7rem", opacity: 0.6 }}
          >
            …
          </span>
        )}
        {state === "error" && (
          <span
            className="vi-comfy-hint"
            title="This view failed to load."
            style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.8rem", opacity: 0.85 }}
          >
            ⚠
          </span>
        )}
        {canonical && (
          <span title="in the canonical set" style={{ position: "absolute", top: 2, right: 3, fontSize: "0.75rem", lineHeight: 1, textShadow: "0 0 3px #000" }}>
            ★
          </span>
        )}
      </div>
      <span className="vi-comfy-hint" style={{ fontSize: "0.65rem", opacity: 0.85, lineHeight: 1.1 }}>
        {label}
      </span>
      <label style={{ display: "flex", alignItems: "center", gap: "0.2rem", fontSize: "0.65rem", cursor: busy ? "default" : "pointer" }}>
        <input type="checkbox" checked={selected} disabled={busy} onChange={onToggle} /> pick
      </label>
      <button
        type="button"
        className="vi-btn vi-btn-sm vi-btn-ghost"
        disabled={busy}
        onClick={onRegenerate}
        title={`Regenerate the ${label} view`}
        style={{ fontSize: "0.65rem", padding: "0 0.35rem" }}
      >
        ↻ redo
      </button>
    </div>
  );
}

// ── SETTINGS panel (IDENTITY-VERSIONS-SLICE.md §2) — per-identity generation
// defaults: texture / pose / turntable geometry / auto-promote / background
// removal / front-reference pick. Local DRAFT + explicit Save (the same idiom
// the card's own name/notes/refs edit already uses — house-consistent, and
// simpler than a debounce timer), so a mid-edit save-mistake is never a silent
// partial-PATCH surprise. Opening the panel resets the draft to the profile's
// current gen_settings, discarding any unsaved edits from a prior open. ────────
function SettingsPanel({
  profile,
  saveSettings,
}: {
  profile: IdentityProfile;
  saveSettings: IdentityProfilesState["saveSettings"];
}) {
  const [draft, setDraft] = useState<GenSettings>(profile.gen_settings);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // The image-text-to-text models the fleet actually advertises, for the vision-model
  // picker — same registry source + stale-while-revalidate posture as the Movie
  // director's judge-VLM select. "" (Default) maps to gen_settings.vision_model = null,
  // i.e. the fleet-default VL model (the 3B). The list shows what's SERVABLE (never a
  // typed string), and the server re-validates the key against the live registry.
  const {
    models: visionModels,
    loading: visionLoading,
    error: visionError,
    refresh: refreshVision,
  } = useModelsByTask(VISION_TASK);

  async function save() {
    setBusy(true);
    setMsg(null);
    const res = await saveSettings(profile.slug, draft);
    setBusy(false);
    setMsg(res.ok ? "Saved." : res.message || "Could not save settings.");
  }

  return (
    <section
      className="vi-comfy-bar vi-id-settings"
      aria-label={`Generation settings for ${profile.name}`}
      style={{
        marginTop: "0.3rem",
        padding: "0.5rem",
        borderRadius: "0.4rem",
        border: "1px solid var(--vi-border)",
        display: "flex",
        flexDirection: "column",
        gap: "0.4rem",
      }}
    >
      <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.85rem" }}>
          <input
            type="checkbox"
            checked={draft.texture}
            disabled={busy}
            onChange={(e) => setDraft((d) => ({ ...d, texture: e.target.checked }))}
          />
          Texture
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.85rem" }}>
          <input
            type="checkbox"
            checked={draft.auto_promote}
            disabled={busy}
            onChange={(e) => setDraft((d) => ({ ...d, auto_promote: e.target.checked }))}
          />
          Auto-promote canonical
        </label>
        <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.85rem" }}>
          <input
            type="checkbox"
            checked={draft.remove_background}
            disabled={busy}
            onChange={(e) => setDraft((d) => ({ ...d, remove_background: e.target.checked }))}
          />
          Remove background
        </label>
      </div>

      <label className="vi-comfy-field" style={{ flexBasis: "12rem" }}>
        <span className="vi-comfy-label">Pose</span>
        <select
          className="vi-knob-select"
          value={draft.pose}
          disabled={busy}
          onChange={(e) => setDraft((d) => ({ ...d, pose: e.target.value === "t-pose" ? "t-pose" : "none" }))}
        >
          <option value="none">None</option>
          <option value="t-pose">T-pose</option>
        </select>
      </label>

      <label className="vi-comfy-field" style={{ flexBasis: "20rem" }}>
        <span className="vi-comfy-label">Vision model</span>
        <span style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          <select
            className="vi-knob-select"
            style={{ flex: 1 }}
            value={draft.vision_model ?? ""}
            disabled={busy || visionLoading}
            onChange={(e) =>
              setDraft((d) => ({ ...d, vision_model: e.target.value || null }))
            }
          >
            {/* Default (null/empty) is always the first, always-selectable option.
                Server-side AUTO: prefers a 7B VL when the fleet has one, else the
                fleet default (3B). */}
            <option value="">Auto (7B if available, else fleet default)</option>
            {/* Keep a saved value selectable even if it isn't in the current list
                (registry still loading, or the model went off-disk). */}
            {draft.vision_model &&
              !visionModels.some((m) => m.id === draft.vision_model) && (
                <option value={draft.vision_model}>{draft.vision_model}</option>
              )}
            {visionModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            onClick={refreshVision}
            disabled={busy || visionLoading}
            title="Re-scan the registry for image-text-to-text vision models"
          >
            ↻
          </button>
        </span>
        {visionError && <span className="vi-error">{visionError}</span>}
      </label>

      {/* CLEANUP-PROMPT slice (C4 — operator 2026-07-15): "possibly these are wired but
          not set... id like pregen prompting... to be facilitated" — makes the C1-C3
          plumbed cleanup_prompt/negative_prompt fields reachable. cleanup_prompt is a
          POSITIVE-worded avoid instruction woven into the T-pose front render (the mesh
          path); negative_prompt is a TRUE negative forwarded to the studio Wan-VACE
          render (reconstruction + the mesh route's id_lock front render). Both persist
          via the same gen_settings PATCH every other Advanced field above uses; both
          default "" — byte-identical to today until set (defaults-are-promises). */}
      <label className="vi-comfy-field" style={{ flexBasis: "20rem" }}>
        <span className="vi-comfy-label">Avoid / cleanup</span>
        <input
          type="text"
          className="vi-knob-input"
          style={{ width: "100%" }}
          value={draft.cleanup_prompt}
          disabled={busy}
          placeholder="e.g. no object on back, no logos, clean bare back"
          onChange={(e) => setDraft((d) => ({ ...d, cleanup_prompt: e.target.value }))}
        />
      </label>

      <label className="vi-comfy-field" style={{ flexBasis: "20rem" }}>
        <span className="vi-comfy-label">Negative prompt</span>
        <input
          type="text"
          className="vi-knob-input"
          style={{ width: "100%" }}
          value={draft.negative_prompt}
          disabled={busy}
          placeholder="e.g. backpack, symbols, logo"
          onChange={(e) => setDraft((d) => ({ ...d, negative_prompt: e.target.value }))}
        />
      </label>

      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem" }}>
        <label className="vi-comfy-field" style={{ flexBasis: "6rem" }}>
          <span className="vi-comfy-label">Frames</span>
          <input
            type="number"
            className="vi-knob-input"
            min={1}
            value={draft.frame_count}
            disabled={busy}
            onChange={(e) => setDraft((d) => ({ ...d, frame_count: Number(e.target.value) || d.frame_count }))}
          />
        </label>
        <label className="vi-comfy-field" style={{ flexBasis: "6rem" }}>
          <span className="vi-comfy-label">FPS</span>
          <input
            type="number"
            className="vi-knob-input"
            min={1}
            value={draft.fps}
            disabled={busy}
            onChange={(e) => setDraft((d) => ({ ...d, fps: Number(e.target.value) || d.fps }))}
          />
        </label>
        <label className="vi-comfy-field" style={{ flexBasis: "6rem" }}>
          <span className="vi-comfy-label">Width</span>
          <input
            type="number"
            className="vi-knob-input"
            min={64}
            step={64}
            value={draft.width}
            disabled={busy}
            onChange={(e) => setDraft((d) => ({ ...d, width: Number(e.target.value) || d.width }))}
          />
        </label>
        <label className="vi-comfy-field" style={{ flexBasis: "6rem" }}>
          <span className="vi-comfy-label">Height</span>
          <input
            type="number"
            className="vi-knob-input"
            min={64}
            step={64}
            value={draft.height}
            disabled={busy}
            onChange={(e) => setDraft((d) => ({ ...d, height: Number(e.target.value) || d.height }))}
          />
        </label>
      </div>

      <div>
        <span className="vi-comfy-label">Front reference</span>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", marginTop: "0.3rem", alignItems: "center" }}>
          <button
            type="button"
            className={`vi-btn vi-btn-sm ${draft.front_ref == null ? "vi-btn-accent" : "vi-btn-ghost"}`}
            disabled={busy}
            onClick={() => setDraft((d) => ({ ...d, front_ref: null }))}
            title="Let the pipeline pick the best source photo as the mesh front"
          >
            Auto
          </button>
          {profile.reference_images.map((uri, idx) => (
            <button
              type="button"
              key={`${uri}_${idx}`}
              disabled={busy}
              onClick={() => setDraft((d) => ({ ...d, front_ref: uri }))}
              title={`Use reference ${idx + 1} as the mesh front`}
              style={{
                padding: 0,
                border: draft.front_ref === uri ? "2px solid var(--vi-accent, #6aa8ff)" : "1px solid var(--vi-border)",
                borderRadius: "0.35rem",
                overflow: "hidden",
                background: "none",
                cursor: busy ? "default" : "pointer",
                lineHeight: 0,
              }}
            >
              <img
                src={mediaBytesUrl(uri)}
                alt={`reference ${idx + 1}`}
                style={{ width: "2.6rem", height: "2.6rem", objectFit: "cover", display: "block" }}
              />
            </button>
          ))}
        </div>
      </div>

      <div className="vi-comfy-bar-actions" style={{ alignItems: "center" }}>
        <button type="button" className="vi-btn vi-btn-sm vi-btn-accent" disabled={busy} onClick={() => void save()}>
          {busy ? "Saving…" : "Save settings"}
        </button>
        {msg && (
          <span className="vi-comfy-hint" role="status" style={{ opacity: 0.85 }}>
            {msg}
          </span>
        )}
      </div>
    </section>
  );
}

// ── one VERSION row (IDENTITY-VERSIONS-SLICE.md §1/§4) — name, kind badge,
// created date, ACTIVE marker; Activate / Rename (inline) / Archive (confirm;
// disabled for the base clay version + the active version, tooltip says why).
// Never-delete lives at this layer now — Archive reads "archive", not "delete",
// and the confirm says so. `isBase` is inferred from `kind === "clay"` (the
// wire contract carries no explicit is-base flag; the design doc says the clay
// base is "always kept, never replaced silently" and is minted once per
// identity, so kind alone is the stable signal today). ─────────────────────
function VersionRow({
  version,
  isActive,
  isBase,
  onActivate,
  onRename,
  onArchive,
}: {
  version: IdentityVersion;
  isActive: boolean;
  isBase: boolean;
  onActivate: () => Promise<{ ok: boolean; message?: string }>;
  onRename: (name: string) => Promise<{ ok: boolean; message?: string }>;
  onArchive: () => Promise<{ ok: boolean; message?: string }>;
}) {
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [draftName, setDraftName] = useState(version.name);
  const [msg, setMsg] = useState<string | null>(null);

  const archiveDisabled = isActive || isBase;
  const archiveTitle = isActive
    ? "Can't archive — this is the active version."
    : isBase
      ? "Can't archive — this is the base (clay) version."
      : "Archive this version (recoverable, never erased).";

  async function doActivate() {
    setBusy(true);
    setMsg(null);
    const res = await onActivate();
    setBusy(false);
    if (!res.ok) setMsg(res.message || "Could not activate this version.");
  }
  async function doRename() {
    const trimmed = draftName.trim();
    if (!trimmed) {
      setMsg("A name is required.");
      return;
    }
    setBusy(true);
    setMsg(null);
    const res = await onRename(trimmed);
    setBusy(false);
    if (res.ok) {
      setRenaming(false);
    } else {
      setMsg(res.message || "Could not rename this version.");
    }
  }
  async function doArchive() {
    if (!window.confirm(`Archive version "${version.name || version.version_id}"? Recoverable, never erased.`)) return;
    setBusy(true);
    setMsg(null);
    const res = await onArchive();
    setBusy(false);
    if (!res.ok) setMsg(res.message || "Could not archive this version.");
  }

  const { short: createdShort, full: createdFull } = fmtCreated(version.created_at);

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: "0.5rem",
        padding: "0.35rem 0.5rem",
        borderRadius: "0.35rem",
        border: "1px solid var(--vi-border)",
        background: isActive ? "rgba(106,168,255,0.08)" : "transparent",
      }}
    >
      {renaming ? (
        <>
          <input
            type="text"
            className="vi-knob-input"
            style={{ flex: "1 1 8rem" }}
            value={draftName}
            disabled={busy}
            onChange={(e) => setDraftName(e.target.value)}
          />
          <button type="button" className="vi-btn vi-btn-sm vi-btn-accent" disabled={busy} onClick={() => void doRename()}>
            {busy ? "Saving…" : "Save"}
          </button>
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={busy}
            onClick={() => {
              setRenaming(false);
              setDraftName(version.name);
              setMsg(null);
            }}
          >
            Cancel
          </button>
        </>
      ) : (
        <>
          <span style={{ flex: "1 1 8rem", fontSize: "0.85rem" }}>
            {version.name || version.version_id}
            {isBase && (
              <span
                title="the clay base — the identity's geometric ground truth, always kept and never archivable"
                style={{ marginLeft: "0.4rem", fontSize: "0.7rem", opacity: 0.85 }}
              >
                📌 base
              </span>
            )}
            {isActive && (
              <span title="the active version" style={{ marginLeft: "0.4rem", fontSize: "0.7rem", opacity: 0.85 }}>
                ● ACTIVE
              </span>
            )}
          </span>
          <span
            className="vi-comfy-hint"
            title={`kind: ${version.kind}`}
            style={{ padding: "0.05rem 0.4rem", borderRadius: "0.3rem", border: "1px solid var(--vi-border)", fontSize: "0.7rem" }}
          >
            {version.kind}
          </span>
          {createdShort && (
            <span className="vi-comfy-hint" title={createdFull} style={{ fontSize: "0.7rem", opacity: 0.7 }}>
              {createdShort}
            </span>
          )}
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={busy || isActive}
            onClick={() => void doActivate()}
            title={isActive ? "Already the active version" : "Make this the active version"}
          >
            {isActive ? "Active" : busy ? "Activating…" : "Activate"}
          </button>
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={busy}
            onClick={() => {
              setDraftName(version.name);
              setMsg(null);
              setRenaming(true);
            }}
          >
            Rename
          </button>
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={busy || archiveDisabled}
            title={archiveTitle}
            onClick={() => void doArchive()}
          >
            Archive
          </button>
        </>
      )}
      {msg && (
        <span className="vi-error" role="alert" style={{ flexBasis: "100%", fontSize: "0.75rem" }}>
          {msg}
        </span>
      )}
    </div>
  );
}

// ── VERSIONS list (IDENTITY-VERSIONS-SLICE.md §1/§4) — the identity's clay base
// + N accrued render-sets. Reads straight off the profile's `versions[]` /
// `active_version` (both live off GET /video/identity-profiles). The clay base is
// pinned + visually marked and is never archivable (VersionRow disables Archive for
// it and the active version). Hidden entirely until a first Generate mints the base
// version, so a never-generated identity shows no empty section. ────────────────
function VersionsPanel({
  profile,
  activateVersion,
  renameVersion,
  archiveVersion,
}: {
  profile: IdentityProfile;
  activateVersion: IdentityProfilesState["activateVersion"];
  renameVersion: IdentityProfilesState["renameVersion"];
  archiveVersion: IdentityProfilesState["archiveVersion"];
}) {
  const versions = profile.versions ?? [];
  if (versions.length === 0) return null;
  return (
    <div style={{ marginTop: "0.6rem" }}>
      <p className="vi-comfy-label" style={{ margin: "0 0 0.3rem" }}>
        Versions ({versions.length})
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
        {versions.map((v) => {
          // The clay base is identifiable on the wire by name "base" + kind "clay"
          // (the backend strips the internal is-base flag from _public_version). A
          // renamed base still can't be archived — the server refuses it regardless
          // (honest 400 surfaced in the row) — so name+kind is the display signal only.
          const isBase = v.kind === "clay" && v.name === "base";
          const isActive = profile.active_version != null && profile.active_version === v.version_id;
          return (
            <div key={v.version_id} style={{ display: "contents" }}>
            <VersionRow
              version={v}
              isActive={isActive}
              isBase={isBase}
              onActivate={() => activateVersion(profile.slug, v.version_id)}
              onRename={(name) => renameVersion(profile.slug, v.version_id, name)}
              onArchive={() => archiveVersion(profile.slug, v.version_id)}
            />
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── one profile CARD — display + inline edit + archive + character-sheet ────────
// ── IdentityDetail — the ONE selected identity's full detail (was ProfileCard) ─
// Body logic (state, the desync-safe active-version/active-recon resolution, the
// generate→poll flow, promote/approve, save/archive) is UNCHANGED from the old
// per-card version; only the RETURN JSX is restructured into the operator's
// wireframe (header version picker · turntable + approved 3D · uploaded refs +
// canonical 8 · generate-360). `tab` decides whether the settings surface leads
// (the "settings" sidebar tab); `onTouch` marks this identity touched-this-session
// when it's generated/edited so the "session" list stays honest.
function IdentityDetail({
  profile,
  tab,
  update,
  remove,
  reload,
  libraryImages,
  saveSettings,
  activateVersion,
  renameVersion,
  archiveVersion,
  onTouch,
}: {
  profile: IdentityProfile;
  tab: IdTab;
  update: IdentityProfilesState["update"];
  remove: IdentityProfilesState["remove"];
  reload: IdentityProfilesState["reload"];
  libraryImages: LibraryItem[];
  saveSettings: IdentityProfilesState["saveSettings"];
  activateVersion: IdentityProfilesState["activateVersion"];
  renameVersion: IdentityProfilesState["renameVersion"];
  archiveVersion: IdentityProfilesState["archiveVersion"];
  onTouch: (slug: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(profile.name);
  const [notes, setNotes] = useState(profile.notes ?? "");
  const [refs, setRefs] = useState<string[]>(profile.reference_images);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // ── SETTINGS (IDENTITY-VERSIONS-SLICE.md §2). In the t17 master-detail layout
  // the settings surface is driven by the sidebar "settings" tab (leads the
  // detail) and by the "⚙ Settings & advanced" disclosure on the other tabs —
  // so there is no longer a per-card settingsOpen toggle to track here.
  const recon = useReconstruction(profile.slug, reload);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [sheetMode, setSheetMode] = useState<"sheet" | "turntable" | "angle-ring">("sheet");
  const [sheetPrompt, setSheetPrompt] = useState(profile.notes ?? "");
  const [sheetSeed, setSheetSeed] = useState("");
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [promoting, setPromoting] = useState(false);
  const [promoteMsg, setPromoteMsg] = useState<string | null>(null);

  // ── ONE-CLICK "Generate 3D identity" (mesh -> 360° turntable -> canonical) ──
  const [genBusy, setGenBusy] = useState(false);
  const [genStatus, setGenStatus] = useState<string | null>(null);
  // Honest degrade surface for pose="t-pose" when the render stage isn't capable on
  // this deployment — the /generate response carries a structured `pose` notice, which
  // we hold here (separate from the poll-churned genStatus) so it stays visible.
  const [poseNotice, setPoseNotice] = useState<string | null>(null);
  // Legacy 2D render actions (angle ring / character sheet / orbit-clip 360) are
  // SUPERSEDED by the one-click mesh path for the identity gen flow — kept behind
  // the "⚙ Settings & advanced" disclosure (see the render) so the tab stays to
  // the point.
  const genTimer = useRef<number | null>(null);
  const genActive = useRef(false);
  // Stop the poll loop + free the timer when this card unmounts mid-build.
  useEffect(() => {
    return () => {
      genActive.current = false;
      if (genTimer.current != null) window.clearTimeout(genTimer.current);
    };
  }, []);

  const recons = profile.reconstructions ?? [];
  const latestRecon = recons.length
    ? recons.reduce((a, b) => ((b.created_at ?? 0) >= (a.created_at ?? 0) ? b : a))
    : null;
  // The reconstruction the card should DISPLAY: the one bound to the ACTIVE version
  // (version.recon_id -> reconstructions[]), NOT "newest by date". Prior code used
  // latestRecon everywhere, so activating base/clay-03/clay-04 never repointed the
  // views panel or the 3D mesh viewer — they stuck to whatever was generated last.
  // Fall back to latestRecon when there is no active version, or its recon_id names
  // nothing on the list (a versionless/legacy profile, or a version whose recon was
  // pruned) — so those profiles behave exactly as before (defaults-are-promises).
  const activeVersion = (profile.versions ?? []).find(
    (v) => profile.active_version != null && v.version_id === profile.active_version,
  );
  const activeRecon =
    (activeVersion?.recon_id
      ? recons.find((r) => r.recon_id === activeVersion.recon_id)
      : null) ?? latestRecon;
  const canonicalPaths = profile.canonical ?? [];
  const canonicalSet = new Set(canonicalPaths);

  function parsedSeed(): number | undefined {
    const t = sheetSeed.trim();
    if (t === "") return undefined;
    const n = Number(t);
    return Number.isFinite(n) ? n : undefined;
  }
  function generateSheet() {
    setSelected(new Set());
    setPromoteMsg(null);
    recon.generate({ mode: "sheet", prompt: sheetPrompt.trim() || undefined, views: [...DEFAULT_VIEWS], seed: parsedSeed() });
  }
  function generateTurntable() {
    setSelected(new Set());
    setPromoteMsg(null);
    recon.generate({ mode: "turntable", prompt: sheetPrompt.trim() || undefined, seed: parsedSeed() });
  }
  function generateAngleRing() {
    setSelected(new Set());
    setPromoteMsg(null);
    recon.generate({ 
      mode: "angle-ring", 
      prompt: sheetPrompt.trim() || undefined, 
      seed: parsedSeed(), 
      angle_step_deg: 10, 
      elevations_deg: [0] 
    });
  }
  function openPanel(mode: "sheet" | "turntable" | "angle-ring") {
    if (sheetOpen && sheetMode === mode) {
      setSheetOpen(false);
      return;
    }
    setSheetMode(mode);
    setSheetOpen(true);
  }
  function regenerateView(idx: number) {
    setPromoteMsg(null);
    const view = DEFAULT_VIEWS[idx];
    recon.generate({ prompt: sheetPrompt.trim() || undefined, views: view ? [view] : undefined, seed: parsedSeed() });
  }
  function toggleSelected(idx: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }
  async function approveCanonical() {
    if (!activeRecon) return;
    const idxs = [...selected].sort((a, b) => a - b);
    if (idxs.length === 0) {
      setPromoteMsg("Pick at least one view to approve.");
      return;
    }
    setPromoting(true);
    setPromoteMsg(null);
    const res = await recon.promote(activeRecon.recon_id, idxs);
    setPromoting(false);
    if (res.ok) {
      setPromoteMsg("Promoted to canonical.");
      setSelected(new Set());
      reload();
    } else {
      setPromoteMsg(res.message || "Could not promote to canonical.");
    }
  }
  async function approveTurntable() {
    if (!activeRecon || activeRecon.mode !== "turntable") return;
    const n = activeRecon.views.length;
    if (n === 0) return;
    const idxs = Array.from(new Set([0, 1, 2, 3].map((k) => Math.min(n - 1, Math.round((k * n) / 4)))));
    setPromoting(true);
    setPromoteMsg(null);
    const res = await recon.promote(activeRecon.recon_id, idxs);
    setPromoting(false);
    if (res.ok) {
      setPromoteMsg(`Approved ${idxs.length} anchor view${idxs.length === 1 ? "" : "s"} to canonical.`);
      reload();
    } else {
      setPromoteMsg(res.message || "Could not approve to canonical.");
    }
  }

  function startEdit() {
    setName(profile.name);
    setNotes(profile.notes ?? "");
    setRefs(profile.reference_images);
    setMsg(null);
    setEditing(true);
  }
  function cancelEdit() {
    setEditing(false);
    setMsg(null);
  }
  const addRef = useCallback((ref: MediaRef) => {
    setRefs((prev) => (prev.length >= MAX_SOURCE_IMAGES || prev.includes(ref.uri) ? prev : [...prev, ref.uri]));
  }, []);
  const removeRef = useCallback((idx: number) => {
    setRefs((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  async function save() {
    const trimmed = name.trim();
    if (!trimmed) {
      setMsg("A name is required.");
      return;
    }
    if (refs.length === 0) {
      setMsg("At least one reference image is required — an identity keeps ≥1.");
      return;
    }
    const fields: { name?: string; notes?: string; referenceImages?: string[] } = {};
    if (trimmed !== profile.name) fields.name = trimmed;
    if (notes.trim() !== (profile.notes ?? "")) fields.notes = notes.trim();
    const refsChanged = refs.length !== profile.reference_images.length || refs.some((r, i) => r !== profile.reference_images[i]);
    if (refsChanged) fields.referenceImages = refs;
    if (Object.keys(fields).length === 0) {
      setEditing(false);
      return;
    }
    setBusy(true);
    setMsg(null);
    const res = await update(profile.slug, fields);
    setBusy(false);
    if (res.ok) {
      setEditing(false);
    } else {
      setMsg(res.message || "Could not save changes.");
    }
  }

  async function archive() {
    // Label says "Delete" (the verb users reach for — an "Archive"-only label read as
    // "no delete option exists"); the confirm keeps the never-delete truth visible.
    if (!window.confirm(`Delete identity profile "${profile.name}"? It is archived, not erased — recoverable.`)) return;
    setBusy(true);
    await remove(profile.slug);
    setBusy(false);
  }

  // ONE ACTION → a complete 3D identity: POST /generate (bare body = the happy path:
  // chain the 360° turntable + auto-promote canonical), then poll the minted recon's
  // mesh state every ~5s (bounded ~20min) until done/error, and reload the list so the
  // new reconstruction + canonical show. Reuses the shared `request` transport helper.
  async function generateFullIdentity() {
    if (genBusy) return;
    onTouch(profile.slug);   // generating counts as touching it this session
    setGenBusy(true);
    setGenStatus("Generating… (queued)");
    setPoseNotice(null);
    genActive.current = true;

    // PREFILL from the identity's persisted gen_settings (IDENTITY-VERSIONS-SLICE.md
    // §2 — "settings … prefill the Generate click; the bare click still equals the
    // happy-path defaults"). DEFAULT_GEN_SETTINGS mirror the /generate route defaults,
    // so an unedited identity sends exactly what a bare {} would. remove_background is
    // persisted on the identity but NOT yet consumed by the /generate route, so it is
    // deliberately omitted from the body here (it takes effect once the route reads it).
    const gs = profile.gen_settings;
    const genBody: Record<string, unknown> = {
      texture: gs.texture,
      pose: gs.pose,
      auto_promote: gs.auto_promote,
      turntable: {
        frame_count: gs.frame_count,
        fps: gs.fps,
        width: gs.width,
        height: gs.height,
      },
    };
    // Explicit front-view pick → the route's `views.front` override (jailed server-side
    // to this profile's own images). Omitted when Auto, so the pipeline picks the front.
    if (gs.front_ref) genBody.views = { front: gs.front_ref };
    // Per-identity VISION MODEL → the route resolves body > gen_settings > null. Sent
    // only when set (mirrors front_ref); omitted == the fleet-default VL model. The route
    // ALSO falls back to the persisted gen_settings.vision_model, so a bare click honors
    // the identity's saved choice regardless.
    if (gs.vision_model) genBody.vision_model = gs.vision_model;

    const res = await request<unknown>(identityGenerateUrl(profile.slug), {
      method: "POST",
      body: JSON.stringify(genBody),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "studio", operation: "identity.generate" },
    });
    if (!genActive.current) return;
    if (!res.ok) {
      genActive.current = false;
      setGenBusy(false);
      setGenStatus(describeAppError(errorOf(res)));
      return;
    }
    const enq = okValue(res) as {
      job_id?: string;
      recon_id?: string;
      pose?: { requested?: string; applied?: boolean; capable?: boolean; message?: string };
    };
    // Surface the honest T-pose degrade: when the render stage isn't capable, the build
    // still proceeds off the normal front and the response says so — show that note.
    if (enq.pose && enq.pose.applied === false) {
      setPoseNotice(
        enq.pose.message ||
          "T-pose normalization is not available on this deployment yet — generated from the source pose instead.",
      );
    } else if (enq.pose && enq.pose.applied === true) {
      setPoseNotice("T-pose normalization applied before meshing.");
    }
    const reconId = enq.recon_id;
    if (!reconId) {
      genActive.current = false;
      setGenBusy(false);
      setGenStatus("Malformed generate response.");
      return;
    }

    const deadline = Date.now() + 20 * 60 * 1000; // ~20 min bound
    const pollOnce = async () => {
      if (!genActive.current) return;
      if (Date.now() > deadline) {
        genActive.current = false;
        setGenBusy(false);
        setGenStatus("Timed out waiting for the 3D identity build.");
        reload();
        return;
      }
      const sres = await request<unknown>(identityMeshStatusUrl(profile.slug, reconId), {
        meta: { specKey: "studio", operation: "identity.mesh.status" },
      });
      if (!genActive.current) return;
      if (sres.ok) {
        const ms = okValue(sres) as { status?: string; error?: string | null };
        const st = ms.status ?? "queued";
        if (st === "done") {
          genActive.current = false;
          setGenBusy(false);
          setGenStatus("done");
          reload();
          return;
        }
        if (st === "error" || st === "cancelled") {
          genActive.current = false;
          setGenBusy(false);
          setGenStatus(ms.error || `Build ${st}.`);
          reload();
          return;
        }
        setGenStatus(`Generating… (${st})`);
      }
      // queued/running/transient-poll-failure → keep polling.
      genTimer.current = window.setTimeout(() => void pollOnce(), 5000);
    };
    genTimer.current = window.setTimeout(() => void pollOnce(), 5000);
  }

  const { short: createdShort, full: createdFull } = fmtCreated(profile.created_at);

  // Wire-truth for the CENTREPIECE panels (constraint #1, reused verbatim from
  // the fixed ProfileCard logic above): the turntable + the "approved 3D id"
  // both drive off `activeRecon`, which resolves from the ACTIVE VERSION's
  // recon_id — never `latestRecon` unless there is no active version at all —
  // and the mesh gate checks the backend's real `mesh.status === "done"` +
  // `mesh.glb_path` (snake_case), never a wrong "completed"/glbUri flag.
  const turntableRecon =
    activeRecon && activeRecon.mode === "turntable" && (activeRecon.views as string[]).length > 0
      ? activeRecon
      : null;
  const meshDone =
    (activeRecon as any)?.mesh?.status === "done" && (activeRecon as any).mesh.glb_path
      ? ((activeRecon as any).mesh.glb_path as string)
      : null;
  const canApproveActive = activeRecon != null && !promoting && !recon.running;

  // Approve the ACTIVE version: turntable → 4 anchor views; sheet/angle-ring →
  // the selected views (or all when none picked). One "✓ approve this version"
  // action fronting the same promote paths the legacy card exposed.
  async function approveActiveVersion() {
    if (!activeRecon) return;
    onTouch(profile.slug);
    if (activeRecon.mode === "turntable") {
      await approveTurntable();
      return;
    }
    if (selected.size > 0) {
      await approveCanonical();
      return;
    }
    // No explicit selection on a sheet/ring: promote every view of the active recon.
    const idxs = (activeRecon.views as string[]).map((_, i) => i);
    if (idxs.length === 0) {
      setPromoteMsg("This version has no views to approve yet.");
      return;
    }
    setPromoting(true);
    setPromoteMsg(null);
    const res = await recon.promote(activeRecon.recon_id, idxs);
    setPromoting(false);
    if (res.ok) {
      setPromoteMsg(`Approved ${idxs.length} view${idxs.length === 1 ? "" : "s"} to canonical.`);
      reload();
    } else {
      setPromoteMsg(res.message || "Could not approve this version.");
    }
  }

  const readiness = identityReadiness(profile);

  // Everything that used to be a separate STEP on this page (k94): rebuild (new
  // version) / ✓ approve / sheet picking / angle-ring approval / settings /
  // versions / the legacy 2D render actions. Still functional; ONE expander.
  const advancedBody = (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem", flex: "1 1 100%", minWidth: 0 }}>
      {poseNotice && (
        <p className="vi-comfy-hint" role="status" style={{ margin: 0, opacity: 0.85 }}>{poseNotice}</p>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", alignItems: "center" }}>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-accent"
          disabled={genBusy || busy}
          onClick={() => void generateFullIdentity()}
          title="Rebuild: a 3D mesh, a 360° turntable, auto-promoted canonical references — lands as a new version"
        >
          {genBusy ? "Generating…" : "⟳ rebuild 3D (new version)"}
        </button>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          disabled={!canApproveActive}
          onClick={() => void approveActiveVersion()}
          title="Promote this active version's views to the identity's canonical set"
        >
          {promoting ? "Approving…" : "✓ approve this version"}
        </button>
        {genStatus && (
          <span className="vi-comfy-hint" role="status" style={{ opacity: 0.85 }}>{genStatus}</span>
        )}
        {promoteMsg && (
          <span className="vi-comfy-hint" role="status" style={{ opacity: 0.85 }}>{promoteMsg}</span>
        )}
      </div>

      {/* Sheet per-view picking stays available for granular approval. */}
      {activeRecon && activeRecon.mode === "sheet" && (activeRecon.views as string[]).length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem" }}>
          {(activeRecon.views as string[]).map((relPath, idx) => (
            <div key={relPath} style={{ display: "contents" }}>
            <ReconViewTile
              relPath={relPath}
              label={VIEW_LABELS[DEFAULT_VIEWS[idx]] ?? `view ${idx + 1}`}
              canonical={canonicalSet.has(relPath)}
              selected={selected.has(idx)}
              busy={recon.running || promoting}
              onToggle={() => toggleSelected(idx)}
              onRegenerate={() => regenerateView(idx)}
            />
            </div>
          ))}
        </div>
      )}
      {activeRecon && activeRecon.mode === "angle-ring" && (
        <AngleRingEditor reconstruction={activeRecon as any} api={recon} />
      )}

      <SettingsPanel profile={profile} saveSettings={saveSettings} />
      <VersionsPanel
        profile={profile}
        activateVersion={activateVersion}
        renameVersion={renameVersion}
        archiveVersion={archiveVersion}
      />

      {/* Legacy 2D render actions (angle ring / character sheet / orbit-clip 360). */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
        <button type="button" className={`vi-btn vi-btn-sm ${sheetOpen && sheetMode === "angle-ring" ? "vi-btn-accent" : "vi-btn-ghost"}`}
          onClick={() => openPanel("angle-ring")} title="Generate a 36-angle iterative approval ring">✦ Angle ring</button>
        <button type="button" className={`vi-btn vi-btn-sm ${sheetOpen && sheetMode === "sheet" ? "vi-btn-accent" : "vi-btn-ghost"}`}
          onClick={() => openPanel("sheet")} title="Legacy: turnaround character sheet">✦ Character sheet</button>
        <button type="button" className={`vi-btn vi-btn-sm ${sheetOpen && sheetMode === "turntable" ? "vi-btn-accent" : "vi-btn-ghost"}`}
          onClick={() => openPanel("turntable")} title="Legacy: 360° orbit turntable clip">✦ 360° turntable</button>
      </div>
      {sheetOpen && (
        <section className="vi-comfy-bar" aria-label={`Generate a turnaround for ${profile.name}`}
          style={{ padding: "0.5rem", borderRadius: "0.4rem", border: "1px solid var(--vi-border)" }}>
          <label className="vi-comfy-field" style={{ flex: "1 1 100%" }}>
            <span className="vi-comfy-label">Turnaround prompt (optional)</span>
            <textarea className="vi-knob-input" rows={2} value={sheetPrompt} placeholder="defaults to this identity's notes"
              disabled={recon.running} onChange={(e) => setSheetPrompt(e.target.value)} />
          </label>
          <label className="vi-comfy-field" style={{ flexBasis: "8rem" }}>
            <span className="vi-comfy-label">Seed (optional)</span>
            <input type="number" className="vi-knob-input" value={sheetSeed} placeholder="random"
              disabled={recon.running} onChange={(e) => setSheetSeed(e.target.value)} />
          </label>
          <div className="vi-comfy-bar-actions">
            <button type="button" className="vi-btn vi-btn-sm vi-btn-accent" disabled={recon.running}
              onClick={sheetMode === "angle-ring" ? generateAngleRing : sheetMode === "turntable" ? generateTurntable : generateSheet}>
              {recon.running ? `Generating… (${recon.done}/${recon.total})`
                : sheetMode === "angle-ring" ? "✦ Generate Angle Ring"
                  : sheetMode === "turntable" ? "✦ Generate 360° turntable" : "✦ Generate turnaround"}
            </button>
          </div>
          {recon.error && <p className="vi-error" role="alert" style={{ flexBasis: "100%", margin: 0 }}>{recon.error}</p>}
        </section>
      )}
    </div>
  );

  return (
    <article className="vi-identity-detail" style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      {/* ── HEADER: name · created · ✓ ready to bind · version picker · edit/delete ── */}
      <div className="vi-id-detail-head" style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" }}>
        <h3 style={{ margin: 0, fontSize: "1.05rem", flex: "0 0 auto" }}>{profile.name}</h3>
        <span className="vi-comfy-hint" title={createdFull} style={{ opacity: 0.7, fontSize: "0.75rem" }}>
          {createdShort ? `created ${createdShort}` : ""}
        </span>
        <span
          className="vi-comfy-hint"
          role="status"
          title={
            readiness.ready
              ? "Reference views + a 3D model exist — bind it from “+ identity” in Scene/Movie."
              : `Not ready yet — ${!readiness.views ? "no reference views" : ""}${!readiness.views && !readiness.mesh ? ", " : ""}${!readiness.mesh ? "no 3D model" : ""}.`
          }
          style={{
            padding: "0.05rem 0.5rem", borderRadius: "0.4rem", fontSize: "0.75rem",
            border: `1px solid ${readiness.ready ? "var(--vi-accent, #6aa8ff)" : "var(--vi-border)"}`,
            opacity: readiness.ready ? 1 : 0.75,
          }}
        >
          {readiness.ready ? "✓ ready to bind" : genBusy ? "building…" : "○ not ready"}
        </span>
        <span style={{ flex: 1 }} />
        <label className="vi-comfy-label" style={{ opacity: 0.75 }}>Version</label>
        <select
          className="vi-knob-input"
          style={{ flex: "0 1 18rem" }}
          value={profile.active_version ?? ""}
          disabled={(profile.versions ?? []).length === 0}
          onChange={(e) => { if (e.target.value) void activateVersion(profile.slug, e.target.value); }}
          title={(profile.versions ?? []).length === 0 ? "No versions yet" : "Switch the active version"}
        >
          {(profile.versions ?? []).length === 0 && <option value="">— no versions yet —</option>}
          {(profile.versions ?? []).map((v) => {
            const isBase = v.kind === "clay" && v.name === "base";
            return (
              <option key={v.version_id} value={v.version_id}>
                {v.name || v.version_id}{isBase ? " (base)" : ""}
              </option>
            );
          })}
        </select>
        <button type="button" className="vi-btn vi-btn-sm vi-btn-ghost" disabled={busy || genBusy} onClick={editing ? cancelEdit : startEdit}>
          {editing ? "Close edit" : "Edit"}
        </button>
        <button type="button" className="vi-btn vi-btn-sm vi-btn-ghost" disabled={busy} onClick={() => void archive()}>
          {busy ? "Deleting…" : "Delete"}
        </button>
      </div>

      {/* Inline edit (rename display-only + notes) — opened from the header. */}
      {editing && (
        <div style={{ border: "1px solid var(--vi-border)", borderRadius: "0.5rem", padding: "0.6rem", display: "flex", flexDirection: "column", gap: "0.4rem" }}>
          <label className="vi-comfy-field">
            <span className="vi-comfy-label">Name</span>
            <input type="text" className="vi-knob-input" value={name} disabled={busy} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="vi-comfy-field">
            <span className="vi-comfy-label">Notes</span>
            <textarea className="vi-knob-input" rows={2} value={notes} disabled={busy} onChange={(e) => setNotes(e.target.value)} />
          </label>
          {msg && <p className="vi-error" role="alert" style={{ margin: 0 }}>{msg}</p>}
          <div style={{ display: "flex", gap: "0.4rem" }}>
            <button type="button" className="vi-btn vi-btn-sm vi-btn-accent" disabled={busy} onClick={() => void save()}>
              {busy ? "Saving…" : "Save"}
            </button>
            <button type="button" className="vi-btn vi-btn-sm vi-btn-ghost" disabled={busy} onClick={cancelEdit}>Cancel</button>
          </div>
        </div>
      )}

      {/* ── VIEWER: turntable (left) · 3D model (right). Wire-truth off the ACTIVE
          version's reconstruction (mesh.status === "done" + mesh.glb_path). ── */}
      <div className="vi-id-centre" style={{ display: "grid", gridTemplateColumns: "minmax(0, 440px) minmax(0, 1fr)", gap: "1rem", alignItems: "start" }}>
        <div>
          <p className="vi-comfy-label" style={{ margin: "0 0 0.3rem" }}>Turntable {turntableRecon ? "(drag to orbit)" : ""}</p>
          <div style={{ width: "100%", maxWidth: 440, aspectRatio: "440 / 320", border: "1px solid var(--vi-border)", borderRadius: "0.5rem", overflow: "hidden", background: "#0b0b0b", display: "flex", alignItems: "center", justifyContent: "center" }}>
            {turntableRecon ? (
              <TurntableViewer
                frames={turntableRecon.views as string[]}
                degreesPerFrame={(turntableRecon as any).degrees_per_frame}
                frameCount={(turntableRecon as any).frame_count}
              />
            ) : (
              <span className="vi-comfy-hint" style={{ opacity: 0.7, padding: "1rem", textAlign: "center" }}>
                {genBusy ? (genStatus || "Building…") : "No turntable for the active version."}
              </span>
            )}
          </div>
        </div>
        <div>
          <p className="vi-comfy-label" style={{ margin: "0 0 0.3rem" }}>3D model (active version)</p>
          <div style={{ width: "100%", maxWidth: 500, aspectRatio: "500 / 320", border: "1px solid var(--vi-border)", borderRadius: "0.5rem", overflow: "hidden", background: "#0b0b0b", display: "flex", alignItems: "center", justifyContent: "center" }}>
            {meshDone ? (
              <MeshViewer url={mediaBytesUrl(meshDone)} height={320} />
            ) : (
              <span className="vi-comfy-hint" style={{ opacity: 0.7, padding: "1rem", textAlign: "center" }}>
                {genBusy ? (genStatus || "Building…") : "No 3D model for the active version."}
              </span>
            )}
          </div>
          {meshDone && (
            <div style={{ display: "flex", gap: "0.4rem", marginTop: "0.4rem", alignItems: "center" }}>
              <a className="vi-btn vi-btn-sm vi-btn-ghost" href={mediaBytesUrl(meshDone)} download>↓ GLB</a>
            </div>
          )}
        </div>
      </div>

      {/* ── references (left, add/remove) · canonical views (right) ── */}
      <div className="vi-id-bottom" style={{ display: "grid", gridTemplateColumns: "minmax(0, 440px) minmax(0, 1fr)", gap: "1rem", alignItems: "start" }}>
        <div>
          <p className="vi-comfy-label" style={{ margin: "0 0 0.3rem" }}>
            References ({refs.length})
          </p>
          <ReferenceEditor refs={refs} onAdd={addRef} onRemove={removeRef} libraryImages={libraryImages} busy={busy} />
          <div style={{ display: "flex", gap: "0.4rem", marginTop: "0.4rem", alignItems: "center" }}>
            {(() => {
              const refsChanged =
                refs.length !== profile.reference_images.length ||
                refs.some((r, i) => r !== profile.reference_images[i]);
              return (
                <button
                  type="button"
                  className="vi-btn vi-btn-sm vi-btn-accent"
                  disabled={busy || !refsChanged || refs.length === 0}
                  onClick={() => { onTouch(profile.slug); void save(); }}
                  title={refs.length === 0 ? "An identity keeps ≥1 reference" : "Save the reference list"}
                >
                  {busy ? "Saving…" : "Save references"}
                </button>
              );
            })()}
            {msg && <span className="vi-error" role="alert" style={{ fontSize: "0.78rem" }}>{msg}</span>}
          </div>
        </div>

        {/* CANONICAL — the identity DNA: TOP-LEVEL profile.canonical only, with the
            honest angle label (viewLabel reused from GenIdentityBar). */}
        <div>
          <p className="vi-comfy-label" style={{ margin: "0 0 0.3rem" }}>
            Canonical views {canonicalPaths.length ? `(${canonicalPaths.length})` : ""}
          </p>
          {canonicalPaths.length === 0 ? (
            <p className="vi-comfy-hint" style={{ opacity: 0.7 }}>No canonical views yet.</p>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(5rem, 1fr))", gap: "0.4rem" }}>
              {canonicalPaths.map((relPath, idx) => (
                <figure key={`${relPath}_${idx}`} style={{ margin: 0, display: "flex", flexDirection: "column", gap: "0.15rem" }}>
                  <img
                    src={mediaBytesUrl(relPath)}
                    alt={`${profile.name} canonical ${viewLabel(profile, idx)}`}
                    title={`canonical · ${viewLabel(profile, idx)}`}
                    style={{ width: "100%", aspectRatio: "1", objectFit: "cover", borderRadius: "0.35rem", background: "#000", border: "1px solid var(--vi-accent, var(--vi-border))" }}
                  />
                  <figcaption className="vi-comfy-hint" style={{ fontSize: "0.68rem", opacity: 0.8, textAlign: "center" }}>
                    {viewLabel(profile, idx)}
                  </figcaption>
                </figure>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ONE expander for everything else (k94). The sidebar "settings" tab leads
          with it open; every other tab keeps it collapsed. */}
      <AdvancedExpander open={tab === "settings"}>{advancedBody}</AdvancedExpander>
    </article>
  );
}

// ── the station — master/detail per the operator's t17 wireframe ─────────────
// "basically one 'name' per view rather than all in view" — the tab is a FOCUSED
// single-identity detail page: the sidebar (master list + active/session/settings
// tab filter) and the header name picker SWITCH which one identity fills the main
// area. The old grid-of-cards (every identity rendered at once) is gone; each
// ProfileCard body is now the IdentityDetail rendered for exactly the selected
// slug. All data flow, the desync-safe active-version logic, generate/poll, and
// the promote/approve paths are REUSED verbatim from ProfileCard — this is a
// rearrangement, not a rewrite.
type IdTab = "active" | "session" | "settings";

export function IdentitiesStation({ spec }: { spec: StationSpec }) {
  void spec;
  const {
    profiles,
    loading,
    error,
    create,
    update,
    remove,
    reload,
    saveSettings,
    activateVersion,
    renameVersion,
    archiveVersion,
    extractFromVideo,
    createFromVideo,
    createFromImages,
  } = useIdentityProfiles();
  const library = useMediaLibrary();
  const libraryImages = library.filter((it) => it.ref.kind === "image");

  // Sidebar tab (per-tab list filter). "settings" opens the detail on its
  // settings surface; the list stays visible so you can pick which identity.
  const [tab, setTab] = useState<IdTab>("active");
  // The ONE selected identity, keyed by SLUG so it survives the hook's reload
  // cadence (same discipline as the Serving-table multi-select Set) — the
  // profiles array is replaced on every reload, but the slug is stable.
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  // "session" = identities TOUCHED this session (selected / created / generated).
  // There is no server-side last-used timestamp on the profile wire, so this is
  // a client-side Set of slugs, seeded as identities are interacted with. Kept
  // in a ref-backed state so it survives reloads like the selection does.
  const [sessionSlugs, setSessionSlugs] = useState<Set<string>>(() => new Set());
  const touch = useCallback((slug: string) => {
    setSessionSlugs((prev) => (prev.has(slug) ? prev : new Set(prev).add(slug)));
  }, []);

  // Selecting an identity marks it touched-this-session and opens its detail.
  const select = useCallback(
    (slug: string) => {
      setSelectedSlug(slug);
      touch(slug);
    },
    [touch],
  );

  // Keep the selection valid across reloads: if the selected slug vanished
  // (archived elsewhere), fall back to the first identity; if nothing is
  // selected yet, default to the first once the list arrives.
  useEffect(() => {
    if (loading) return;
    const has = (s: string | null) => s != null && profiles.some((p) => p.slug === s);
    if (!has(selectedSlug)) {
      setSelectedSlug(profiles.length ? profiles[0].slug : null);
    }
  }, [loading, profiles, selectedSlug]);

  // Per-tab list filter (operator labels kept; mapping noted in the report):
  //   active   = identities with an active/approved version (active_version set)
  //   session  = identities touched this session (client Set)
  //   settings = ALL identities (the list is the picker for which to configure)
  const listed = profiles.filter((p) => {
    if (tab === "active") return p.active_version != null;
    if (tab === "session") return sessionSlugs.has(p.slug);
    return true; // settings
  });

  const selected = profiles.find((p) => p.slug === selectedSlug) ?? null;

  const TABS: { id: IdTab; label: string; title: string }[] = [
    { id: "active", label: "active", title: "Identities with an approved/active version" },
    { id: "session", label: "session", title: "Identities you've touched this session" },
    { id: "settings", label: "settings", title: "Identity settings — pick one to configure" },
  ];

  return (
    <section className="station-wide vi-identities vi-identities-md" aria-label="Identities workspace"
      style={{ display: "grid", gridTemplateColumns: "300px minmax(0, 1fr)", gap: "1rem", alignItems: "start" }}>
      {/* ── SIDEBAR: tab filter + master list (wireframe b/c/d/f/e) ────────── */}
      <aside className="vi-id-sidebar" aria-label="Identity list"
        style={{ border: "1px solid var(--vi-border)", borderRadius: "0.5rem", padding: "0.5rem", position: "sticky", top: "0.5rem" }}>
        <div role="tablist" aria-label="Identity list filter"
          style={{ display: "flex", gap: "0.25rem", marginBottom: "0.5rem" }}>
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={tab === t.id}
              title={t.title}
              className={`vi-btn vi-btn-sm ${tab === t.id ? "vi-btn-accent" : "vi-btn-ghost"}`}
              style={{ flex: 1 }}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        {loading ? (
          <p className="vi-comfy-hint">Loading…</p>
        ) : error ? (
          <p className="vi-error" role="alert">{error}</p>
        ) : listed.length === 0 ? (
          <p className="vi-comfy-hint" style={{ padding: "0.4rem" }}>
            {tab === "session"
              ? "No identities touched yet this session."
              : tab === "active"
                ? "No identity has an active version yet — generate one."
                : "No saved identities yet — create one above."}
          </p>
        ) : (
          <ul className="vi-id-list" style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: "0.25rem", maxHeight: "640px", overflowY: "auto" }}>
            {listed.map((p) => {
              const isSel = p.slug === selectedSlug;
              const thumb = p.canonical[0] ?? p.reference_images[0];
              return (
                <li key={p.slug}>
                  <button
                    type="button"
                    className="vi-id-list-item"
                    aria-current={isSel}
                    onClick={() => select(p.slug)}
                    style={{
                      display: "flex", alignItems: "center", gap: "0.5rem", width: "100%",
                      textAlign: "left", padding: "0.35rem 0.45rem", borderRadius: "0.4rem",
                      border: `1px solid ${isSel ? "var(--vi-accent, #6aa8ff)" : "var(--vi-border)"}`,
                      background: isSel ? "rgba(106,168,255,0.10)" : "transparent",
                      cursor: "pointer", color: "inherit",
                    }}
                  >
                    {thumb ? (
                      <img src={mediaBytesUrl(thumb)} alt="" aria-hidden="true"
                        style={{ width: "2.2rem", height: "2.2rem", objectFit: "cover", borderRadius: "0.3rem", background: "#000", flex: "0 0 auto" }} />
                    ) : (
                      <span aria-hidden="true" style={{ width: "2.2rem", height: "2.2rem", borderRadius: "0.3rem", background: "var(--vi-border)", flex: "0 0 auto" }} />
                    )}
                    <span style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                      <span style={{ fontSize: "0.85rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.name}</span>
                      <span className="vi-comfy-hint" style={{ fontSize: "0.68rem", opacity: 0.7 }}>
                        {identityReadiness(p).ready ? "✓ ready to bind" : p.active_version != null ? "● active" : "○ not ready"}
                        {p.canonical.length ? ` · ${p.canonical.length} views` : ""}
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </aside>

      {/* ── MAIN: the ONE PATH create panel (k94) · identity picker · detail ── */}
      <div className="vi-id-main" style={{ display: "flex", flexDirection: "column", gap: "0.75rem", minWidth: 0 }}>
        <IdentityCreatePanel
          createFromVideo={createFromVideo}
          createFromImages={createFromImages}
          library={library}
          reload={reload}
          onCreated={(slugs) => {
            for (const s of slugs) touch(s);
            if (slugs[0]) setSelectedSlug(slugs[0]);
          }}
          advancedOpen={tab === "settings"}
          advanced={
            <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", flex: "1 1 100%", minWidth: 0 }}>
              <span className="vi-comfy-label" style={{ opacity: 0.75 }}>Manual create (name + notes + reference images, no 3D build)</span>
              <CreatePanel
                create={async (name, refs, notes) => {
                  const res = await create(name, refs, notes);
                  if (res.ok && res.profile) {
                    touch(res.profile.slug);
                    setSelectedSlug(res.profile.slug);
                  }
                  return res;
                }}
                libraryImages={libraryImages}
              />
              <ExtractFromVideoPanel profiles={profiles} extractFromVideo={extractFromVideo} reload={reload} />
            </div>
          }
        />

        {/* Identity picker — which saved identity fills the detail below. */}
        <div className="vi-id-header" style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" }}>
          <label className="vi-comfy-label" style={{ opacity: 0.75 }}>Identity</label>
          <select
            className="vi-knob-input"
            style={{ flex: "1 1 20rem", maxWidth: "27rem" }}
            value={selectedSlug ?? ""}
            disabled={loading || profiles.length === 0}
            onChange={(e) => e.target.value && select(e.target.value)}
          >
            {profiles.length === 0 && <option value="">— no identities yet —</option>}
            {profiles.map((p) => (
              <option key={p.slug} value={p.slug}>{identityReadiness(p).ready ? "✓ " : ""}{p.name}</option>
            ))}
          </select>
        </div>

        {selected ? (
          <div key={selected.slug} style={{ display: "contents" }}>
          <IdentityDetail
            profile={selected}
            tab={tab}
            update={update}
            remove={(slug) => {
              // After a delete the selection must move off the gone slug.
              const res = remove(slug);
              setSelectedSlug((cur) => (cur === slug ? null : cur));
              return res;
            }}
            reload={reload}
            libraryImages={libraryImages}
            saveSettings={saveSettings}
            activateVersion={activateVersion}
            renameVersion={renameVersion}
            archiveVersion={archiveVersion}
            onTouch={touch}
          />
          </div>
        ) : (
          !loading && (
            <p className="vi-comfy-hint" style={{ padding: "1rem" }}>
              {profiles.length === 0
                ? "No identities yet — create your first above."
                : "Select an identity from the list to view it."}
            </p>
          )
        )}
      </div>
    </section>
  );
}
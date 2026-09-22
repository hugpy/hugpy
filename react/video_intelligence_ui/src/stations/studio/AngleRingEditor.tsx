import { useState } from "react";
import { mediaBytesUrl } from "../../config";
import type { IdentityReconstruction, IdentityAngleView } from "../../video/contract";
import type { ReconstructionApi } from "./useReconstruction";

interface AngleRingEditorProps {
  reconstruction: IdentityReconstruction;
  api: ReconstructionApi;
}

export function AngleRingEditor({ reconstruction, api }: AngleRingEditorProps) {
  const [regeneratePrompt, setRegeneratePrompt] = useState("");

  // Safely default to an empty array if views aren't populated yet
  const views = reconstruction.views ?? [];
  
  // The completion gate for Phase 4 (Mesh Building)
  const allViewsApproved =
    views.length > 0 && views.every((v) => v.status === "approved");

  const meshRunning =
    reconstruction.mesh?.status === "queued" ||
    reconstruction.mesh?.status === "running";

  // Derive an array of 8 geometrically useful anchor views for the mesh build
  // (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
  const getMeshAnchorIds = () => {
    const anchorDegrees = [0, 45, 90, 135, 180, 225, 270, 315];
    return views
      .filter((v) => anchorDegrees.includes(v.azimuthDeg))
      .map((v) => v.viewId);
  };

  return (
    <div className="vi-angle-ring">
      <div className="vi-angle-ring-header">
        <h3>Angle Ring Approval ({views.filter(v => v.status === "approved").length}/{views.length} Approved)</h3>
        
        {/* The Final Gate: Build the 3D Asset */}
        <button
          className="vi-btn vi-btn-accent"
          disabled={!allViewsApproved || meshRunning || api.running}
          onClick={() => api.buildMesh(reconstruction.reconId, getMeshAnchorIds())}
        >
          {meshRunning ? "Building 3D Mesh..." : "Build Approved 3D Model"}
        </button>
      </div>

      {reconstruction.mesh?.error && (
        <p className="vi-error">Mesh Error: {reconstruction.mesh.error}</p>
      )}

      {/* Global override prompt for regenerations */}
      <div className="vi-knob vi-angle-ring-controls">
        <label htmlFor="regen-prompt">Global Regeneration Prompt</label>
        <input
          id="regen-prompt"
          type="text"
          className="vi-knob-input"
          placeholder="e.g. Preserve exact identity, clothing, and proportions..."
          value={regeneratePrompt}
          onChange={(e) => setRegeneratePrompt(e.target.value)}
        />
        <span className="vi-knob-hint">
          Used when regenerating specific rejected angles below. Conditioned on nearest approved neighbors.
        </span>
      </div>

      {/* The 36-Tile Grid */}
      <div className="vi-frames-grid vi-studio-strip vi-angle-ring-grid">
        {views.map((view) => (
          <AngleRingTile
            key={view.viewId}
            view={view}
            reconId={reconstruction.reconId}
            api={api}
            promptOverride={regeneratePrompt}
          />
        ))}
      </div>
    </div>
  );
}

// Sub-component for individual view tiles to keep rendering clean
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
  const hasImage = view.imageUri != null;

  return (
    <div className={`vi-scene-frame-cell vi-angle-tile vi-angle-tile-${view.status}`}>
      {/* Visual Image Area */}
      <div className="vi-angle-img-wrapper">
        {hasImage ? (
          <img
            className="vi-scene-frame-img"
            src={mediaBytesUrl(view.imageUri!)}
            alt={`${view.azimuthDeg}° view`}
            loading="lazy"
          />
        ) : (
          <div className="vi-angle-placeholder">
            {isGenerating ? "Generating..." : "No Image"}
          </div>
        )}
        <span className="vi-frame-tag">{view.azimuthDeg}°</span>
      </div>

      {/* Status & Actions */}
      <div className="vi-angle-actions">
        <span className={`vi-status vi-status-${view.status}`}>
          {view.status} (v{view.version})
        </span>

        <div className="vi-angle-btn-row">
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost vi-btn-approve"
            disabled={!hasImage || view.status === "approved" || isGenerating}
            onClick={() => api.setViewStatus(reconId, view.viewId, "approved")}
            title="Approve this angle"
          >
            ✓
          </button>
          
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost vi-btn-reject"
            disabled={!hasImage || view.status === "rejected" || isGenerating}
            onClick={() => api.setViewStatus(reconId, view.viewId, "rejected")}
            title="Reject this angle"
          >
            ✕
          </button>

          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            disabled={isGenerating}
            onClick={() => api.regenerateView(reconId, view.viewId, promptOverride, undefined, true)}
            title="Regenerate using nearest approved neighbors"
          >
            ↻ Regen
          </button>
        </div>
      </div>
    </div>
  );
}

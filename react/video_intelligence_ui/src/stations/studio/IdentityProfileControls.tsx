// IDENTITY PROFILES — the shared "Identity" control block both studio surfaces
// (the Clip Generate tab + the Movie composer) embed as ONE additive row in their
// existing identity/reference area. It does NOT re-draw the reference thumbnails —
// the host surface's own chip UI already shows whatever is attached; this block adds
// only the profile affordances on top:
//
//   • "👤 From profile" — a picker of saved profiles (name + first-ref thumbnail);
//     picking one RESOLVES its saved paths back to MediaRefs and hands them to the
//     host (replace-semantics: a profile IS the whole identity), and binds the slug.
//   • "💾 Save as identity profile" — enabled once ≥1 reference is attached and the
//     current identity is UNSAVED; prompts for a name (window.prompt, v0) and POSTs.
//
// DOCTRINE (operator 2026-07-12): present ONE concept — "Identity" — backed by
// profiles. A chosen profile shows as a named, saved identity; ad-hoc uploads are an
// honest "unsaved identity" with the save affordance (the incomplete case is STILL
// the identity, just unnamed). The host prefers sending `identity_profile:<slug>` when
// a profile is bound; raw reference_images only for the unsaved case. (Stage (b) —
// turnaround generation + the re-edit loop that promotes an approved rendering to the
// canonical reference — layers on top of this next.)
import { useState, type MouseEvent } from "react";
import { mediaBytesUrl } from "../../config";
import type { MediaRef } from "../../video/contract";
import { useIdentityProfiles } from "./useIdentityProfiles";
import type { IdentityProfile } from "./useIdentityProfiles";

/** The profile currently bound as the identity (or null for an unsaved/empty one). */
export interface SelectedProfile {
  slug: string;
  name: string;
}

export interface IdentityProfileControlsProps {
  /** How many identity references are attached right now (gates Save + status copy). */
  refCount: number;
  /** The raw jailed URIs of the attached references — what Save persists. */
  currentRefUris: string[];
  /** The bound profile, if the current identity came from one (else null = unsaved). */
  selectedProfile: SelectedProfile | null;
  /** Attach a profile's resolved reference set into the host's OWN reference state.
   *  `replace` is true for a profile pick (the profile is the whole identity). */
  onAttach: (refs: MediaRef[], replace: boolean) => void;
  /** Bind (or clear, with null) the profile slug the host sends as identity_profile. */
  onSelectProfile: (p: SelectedProfile | null) => void;
  /** Upper bound on references (4) — advisory copy only. */
  maxRefs: number;
  /** Disable all affordances (host is busy / submitting). */
  disabled?: boolean;
}

export function IdentityProfileControls({
  refCount,
  currentRefUris,
  selectedProfile,
  onAttach,
  onSelectProfile,
  maxRefs,
  disabled,
}: IdentityProfileControlsProps) {
  const { profiles, loading, error, create, remove, resolveRefs } = useIdentityProfiles();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const canSave = refCount > 0 && selectedProfile == null && !disabled && !busy;

  async function pick(profile: IdentityProfile) {
    if (disabled || busy) return;
    setBusy(true);
    setMsg(null);
    try {
      const refs = await resolveRefs(profile);
      if (refs.length === 0) {
        setMsg("Could not load that profile's reference images.");
        return;
      }
      // A profile IS the identity — replace the current set with the profile's.
      onAttach(refs.slice(0, maxRefs), true);
      onSelectProfile({ slug: profile.slug, name: profile.name });
      setOpen(false);
      setMsg(`Identity set to “${profile.name}”.`);
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    if (!canSave) return;
    const name = window.prompt("Name this identity profile:");
    if (name == null) return; // cancelled
    const trimmed = name.trim();
    if (!trimmed) {
      setMsg("A name is required to save an identity profile.");
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      const res = await create(trimmed, currentRefUris);
      if (res.ok && res.profile) {
        // The just-saved profile becomes the bound identity (its refs are already
        // attached), flipping the surface from "unsaved" to a named profile.
        onSelectProfile({ slug: res.profile.slug, name: res.profile.name });
        setMsg(`Saved identity profile “${res.profile.name}”.`);
      } else if (res.duplicate) {
        setMsg(`An identity profile named “${trimmed}” already exists — pick it from 👤 From profile, or choose another name.`);
      } else {
        setMsg(res.message || "Could not save the identity profile.");
      }
    } finally {
      setBusy(false);
    }
  }

  async function archive(profile: IdentityProfile, e: MouseEvent) {
    e.stopPropagation();
    if (disabled || busy) return;
    if (!window.confirm(`Archive identity profile “${profile.name}”? (It is kept, not erased.)`)) return;
    setBusy(true);
    try {
      await remove(profile.slug);
      if (selectedProfile?.slug === profile.slug) onSelectProfile(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ marginTop: "0.4rem" }}>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          disabled={disabled || busy}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          title="Attach a saved identity profile's reference set"
        >
          {open ? "Hide profiles" : `👤 From profile${profiles.length ? ` (${profiles.length})` : ""}`}
        </button>
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          disabled={!canSave}
          onClick={() => void save()}
          title={
            refCount === 0
              ? "Attach at least one reference image first"
              : selectedProfile != null
                ? "This identity is already a saved profile"
                : "Save these references as a reusable identity profile"
          }
        >
          💾 Save as identity profile
        </button>
        {selectedProfile != null ? (
          <span className="vi-comfy-hint" role="status" style={{ opacity: 0.85 }}>
            Identity: <strong>{selectedProfile.name}</strong> — saved profile ✓
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              disabled={disabled || busy}
              onClick={() => onSelectProfile(null)}
              title="Unbind the profile (keep the images as an unsaved identity)"
              style={{ marginLeft: "0.4rem", padding: "0 0.35rem" }}
            >
              unbind
            </button>
          </span>
        ) : refCount > 0 ? (
          <span className="vi-comfy-hint" role="status" style={{ opacity: 0.75 }}>
            Unsaved identity — save it to reuse across clips &amp; movies.
          </span>
        ) : null}
      </div>

      {open && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "0.5rem",
            marginTop: "0.5rem",
            maxHeight: "12rem",
            overflowY: "auto",
          }}
        >
          {loading ? (
            <p className="vi-comfy-hint">Loading identity profiles…</p>
          ) : error ? (
            <p className="vi-error" role="alert">{error}</p>
          ) : profiles.length === 0 ? (
            <p className="vi-comfy-hint">
              No saved identity profiles yet — attach reference image(s) and press
              “💾 Save as identity profile” to create one.
            </p>
          ) : (
            profiles.map((p) => (
              <button
                key={p.slug}
                type="button"
                className="vi-btn vi-btn-ghost"
                disabled={disabled || busy}
                onClick={() => void pick(p)}
                title={`Use “${p.name}” (${p.reference_images.length} reference${
                  p.reference_images.length === 1 ? "" : "s"
                })`}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: "0.2rem",
                  padding: "0.3rem",
                  position: "relative",
                }}
              >
                <img
                  src={mediaBytesUrl(p.reference_images[0])}
                  alt={p.name}
                  style={{
                    width: "4.5rem",
                    height: "4.5rem",
                    objectFit: "cover",
                    borderRadius: "0.35rem",
                    background: "#000",
                  }}
                />
                <span style={{ fontSize: "0.75rem", maxWidth: "5rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {p.name}
                </span>
                <span
                  role="button"
                  tabIndex={-1}
                  aria-label={`Archive ${p.name}`}
                  onClick={(e) => void archive(p, e)}
                  title="Archive (kept, not erased)"
                  style={{
                    position: "absolute",
                    top: -4,
                    right: -4,
                    background: "var(--vi-border)",
                    borderRadius: "0.3rem",
                    padding: "0 0.3rem",
                    fontSize: "0.72rem",
                    lineHeight: 1.4,
                  }}
                >
                  ✕
                </span>
              </button>
            ))
          )}
        </div>
      )}

      {msg && (
        <p className="vi-comfy-hint" role="status" style={{ marginTop: "0.3rem" }}>
          {msg}
        </p>
      )}
    </div>
  );
}

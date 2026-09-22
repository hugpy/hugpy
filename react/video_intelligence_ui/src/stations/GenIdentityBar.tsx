// GENERATION-WIDE IDENTITY — the compact per-part affordance the Image/Scene
// prompt composer embeds next to "+ attach image" / "+ attach video".
//
// DOCTRINE (operator 2026-07-16): "if an identity exists, it can be selected and
// referenced per that prompt for the character association expected within the
// generation." Identity is ONE per generation, NOT one per part — a profile IS the
// whole identity (the same replace-semantics IdentityProfileControls encodes for
// Studio). Every part therefore renders the SAME bound identity and any part can
// bind/clear it; the copy says "this generation" (never "this part") so the user is
// never misled into thinking part #2 holds a different character than part #1.
//
// ONE CARRYING VIEW (operator 2026-07-16, second correction): "selecting an identity
// produces a row of canonical images for that identity in which the user selects one
// that is to be the carrying identity for that prompt."
//
// WHY EXACTLY ONE — this is a TRUTHFULNESS fix, not a preference. The generate runner
// consumes only the FIRST image part:
//     abstract_hugpy_dev/src/abstract_hugpy_dev/video_intel/runners/imagegen.py:117
//       start_frame = image_paths[0] if image_paths else None
//   (scene.py:539 does the same via next(...)). Images 2..N are collected, counted in
// a log line, and NEVER used. The previous "bind attaches every canonical view" flow
// therefore attached 4 parts of which 3 were decorative — the UI implied a multi-view
// conditioning that does not exist. So the user now PICKS the single view that will
// actually carry, and the copy says why. If the runner ever learns to consume real
// multi-view conditioning, this becomes a multi-select — until then, one.
//
// PER-VIEW ANGLE LABELLING — verified against the live payload (GET
// http://127.0.0.1:7002/video/identity-profiles), both cases in the SAME response:
//   * Luigi: canonical=8, canonical_angles=[0.0, 45.0, 90.0, ..., 315.0] — every
//     view now has real azimuth provenance (identity_profiles.py's backfill).
//   * sailor-moon / keeper-mesh-live / video-char-*: canonical=4,
//     canonical_angles ABSENT (key omitted entirely — a pre-backfill profile).
// `profile.canonical` is still a FLAT LIST OF PATH STRINGS; `canonical_angles`,
// when present, is POSITIONALLY ALIGNED to it (canonical_angles[i] describes
// canonical[i]) — additive, never required. The key is omitted entirely when
// unknown for the WHOLE profile; an individual element may be `null` when
// unknown for just that view. On a length mismatch the backend itself drops the
// key rather than mislabel — `viewLabel` below re-checks the lengths anyway
// (never trust the wire honored its own contract). `azimuth_deg` is canonical
// truth (identity_profiles.py:1556) — degrees win over any semantic name on
// disagreement, so names below are ALWAYS derived from the degree, never the
// reverse. When the angle for a specific view is absent/null, THAT view falls
// back to the honest ordinal ("view N") exactly as every view did before this
// slice — never guessed, never inferred from array index.
//
// WHY A SLIM SIBLING, NOT ./studio/IdentityProfileControls:
//   1. That component calls useIdentityProfiles() ITSELF. Mounting it once per part
//      would fire N duplicate GET /video/identity-profiles and hold N independently
//      stale lists (the hook has no cross-instance cache). This component is PURE —
//      the profile list + callbacks arrive as props from ONE hook at the station.
//   2. Its Save/Archive verbs are identity-library management; per prompt part the
//      only sensible verbs are bind + clear (managing profiles lives in the
//      Identities station).
//   3. It is a spacious Studio-sized block (inline styles, a 12rem scrolling panel);
//      the vi-gen-part card family is compact.
// IdentityProfileControls is UNTOUCHED and still used by Studio — nothing superseded.
import { useState } from "react";
import { mediaBytesUrl } from "../config";
import type { IdentityProfile } from "./studio/useIdentityProfiles";

// Mirrors the backend's SEMANTIC_VIEWS (identity_profiles.py) — a convenience
// display name over the canonical degree, NEVER the source of truth. Degrees win
// on disagreement: this map only ever picks the NEAREST name to a real
// azimuth_deg, it never produces a degree itself.
// HANDEDNESS RESOLVED 2026-07-16 — operator eyeballed the labeled rows on two textured
// identities: "0 is front and 90 degrees is stage left". Stage left == the performer's
// left == the camera's right, so at 90° the camera sees the subject's own LEFT side.
// The sided names are mirrored from the original never-eyeballed guess (45<->315,
// 90<->270, 135<->225); 0°/180° are the mirror's fixed points. Kept in lockstep with
// the backend's SEMANTIC_VIEWS (identity_profiles.py) — if these two ever disagree the
// backend wins, and the DEGREES win over both.
const SEMANTIC_VIEWS: ReadonlyArray<readonly [number, string]> = [
  [0, "front"],
  [45, "¾ left"],
  [90, "left profile"],
  [135, "back-left"],
  [180, "back"],
  [225, "back-right"],
  [270, "right profile"],
  [315, "¾ right"],
];

/** Nearest semantic name for a real azimuth degree (wraps 0/360). */
function semanticNameFor(deg: number): string {
  const norm = ((deg % 360) + 360) % 360;
  let best = SEMANTIC_VIEWS[0];
  let bestDist = 360;
  for (const entry of SEMANTIC_VIEWS) {
    const [target] = entry;
    const dist = Math.min(Math.abs(norm - target), 360 - Math.abs(norm - target));
    if (dist < bestDist) {
      bestDist = dist;
      best = entry;
    }
  }
  return best[1];
}

/** The profile bound as this generation's identity (mirrors studio's SelectedProfile). */
export interface BoundIdentity {
  slug: string;
  name: string;
  /** Index into profile.canonical of the ONE view carrying the identity. */
  viewIndex: number;
}

/** The real azimuth for canonical view #i, or null when unknown.
 *
 *  Honest by construction: `canonical_angles` is optional and may be absent
 *  entirely (whole profile pre-backfill), shorter/longer than `canonical` (the
 *  UI's own length re-check — never trust the wire honored its own contract even
 *  though the backend is documented to drop the key itself on mismatch), or hold
 *  a `null` at this specific index (unknown for just that view). Any of those
 *  yields null here, never a guessed 0. */
function angleFor(profile: IdentityProfile | null, index: number): number | null {
  if (!profile) return null;
  const angles = profile.canonical_angles;
  if (!angles || angles.length !== profile.canonical.length) return null;
  const deg = angles[index];
  return typeof deg === "number" ? deg : null;
}

/** The label for canonical view #i — prefers the real azimuth_deg (degrees +
 *  nearest semantic name, degrees are truth per the backend's documented
 *  convention), falling back to today's honest ordinal ("view N") when the
 *  angle for THIS view is absent/null (or the profile itself isn't available,
 *  e.g. the bound chip after its profile was archived out from under it). Never
 *  guesses, never infers an angle from array index — a profile with no angle
 *  data renders byte-identical to before this slice. */
export function viewLabel(profile: IdentityProfile | null, index: number): string {
  const deg = angleFor(profile, index);
  if (deg == null) return `view ${index + 1}`;
  const rounded = Math.round(deg);
  return `${rounded}° ${semanticNameFor(deg)}`;
}

export interface GenIdentityBarProps {
  /** The saved profiles — from the station's SINGLE useIdentityProfiles() instance. */
  profiles: IdentityProfile[];
  /** List still loading (station-level hook). */
  loading: boolean;
  /** List failed to load (station-level hook), or null. */
  error: string | null;
  /** The generation-wide bound identity, or null when none is bound. */
  bound: BoundIdentity | null;
  /** Bind this profile using canonical view `viewIndex` as the ONE carrying view:
   *  the station resolves that single CANONICAL 3D-model view (not the raw
   *  reference_images source uploads) → exactly one image part, then binds.
   *  Resolves to a human error string, or null on success — a profile with no
   *  canonical views (or whose view won't resolve) fails loudly here, never
   *  silently no-ops and never substitutes a different set. */
  onBind: (profile: IdentityProfile, viewIndex: number) => Promise<string | null>;
  /** Unbind: drops the identity's image part and clears the binding. */
  onClear: () => void;
  /** True on the part that owns the expanded picker panel (only one at a time). */
  disabled?: boolean;
}

export function GenIdentityBar({
  profiles,
  loading,
  error,
  bound,
  onBind,
  onClear,
  disabled,
}: GenIdentityBarProps) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  // STEP 2 state: the profile whose canonical row is being shown for view-pick.
  // null = step 1 (choose an identity). Kept as the SLUG (not the object) so a
  // profile-list refresh can't strand us on a stale copy.
  const [pickingSlug, setPickingSlug] = useState<string | null>(null);
  // Failure is LOCAL and explicit: a profile whose views won't resolve must say
  // so here, never silently no-op (the brief's honest-failure rule).
  const [msg, setMsg] = useState<string | null>(null);

  const picking = pickingSlug
    ? profiles.find((p) => p.slug === pickingSlug) ?? null
    : null;

  // The bound chip needs the SAME profile (for its canonical_angles) to label the
  // carrying view honestly. Looked up by slug from the station's live list rather
  // than carried on BoundIdentity itself, so a profile refresh (e.g. the backend
  // angle backfill landing mid-session) is picked up automatically. Falls back to
  // the honest ordinal via viewLabel's own null-handling when the profile isn't
  // found (e.g. it was archived out from under an existing bind).
  const boundProfile = bound ? profiles.find((p) => p.slug === bound.slug) ?? null : null;

  function reset() {
    setOpen(false);
    setPickingSlug(null);
    setMsg(null);
  }

  /** Step 1 → 2. A profile with no canonical set fails loudly here rather than
   *  opening an empty row (which would read as "nothing to pick" — a silent
   *  dead end). */
  function chooseProfile(profile: IdentityProfile) {
    if (disabled || busy) return;
    setMsg(null);
    if (profile.canonical.length === 0) {
      setMsg(
        `“${profile.name}” has no canonical 3D-model views yet — render its 3D model in the Identities station first.`,
      );
      return;
    }
    setPickingSlug(profile.slug);
  }

  /** Step 2 — commit the ONE carrying view. */
  async function pickView(profile: IdentityProfile, viewIndex: number) {
    if (disabled || busy) return;
    setBusy(true);
    setMsg(null);
    try {
      const failure = await onBind(profile, viewIndex);
      if (failure) {
        setMsg(failure);
        return;
      }
      reset();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="vi-gen-part-identity">
      <div className="vi-gen-part-identity-row">
        {bound ? (
          <>
            <span
              className="vi-gen-identity-chip"
              title={`Identity profile: ${bound.name} — carried by canonical ${viewLabel(
                boundProfile,
                bound.viewIndex,
              )}`}
            >
              <span className="vi-gen-identity-chip-icon" aria-hidden>
                👤
              </span>
              {bound.name}
              <span className="vi-gen-identity-chip-view">
                {viewLabel(boundProfile, bound.viewIndex)}
              </span>
            </span>
            {/* Wording is deliberately generation-wide, not part-scoped: one identity
                is shared by every part, and this line renders identically on all of
                them. */}
            <span className="vi-gen-identity-note">
              character for this whole generation — shared by every part
            </span>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              disabled={disabled || busy}
              onClick={() => {
                setMsg(null);
                onClear();
              }}
              title="Unbind this identity and remove its 3D-model view part"
            >
              ✕ identity
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-ghost"
              disabled={disabled || busy || loading}
              aria-expanded={open}
              onClick={() => {
                setMsg(null);
                setPickingSlug(null);
                setOpen((v) => !v);
              }}
              title="Reference a saved character identity for this generation"
            >
              {open
                ? "hide identities"
                : `👤 identity${profiles.length ? ` (${profiles.length})` : ""}`}
            </button>
            {open && (
              <span className="vi-gen-identity-note">
                one character, shared by every part of this generation
              </span>
            )}
          </>
        )}
      </div>

      {open && !bound && (
        <div className="vi-gen-identity-picker">
          {loading ? (
            <p className="vi-gen-part-meta">Loading identities…</p>
          ) : error ? (
            <p className="vi-error" role="alert">
              {error}
            </p>
          ) : profiles.length === 0 ? (
            <p className="vi-gen-part-meta">
              No saved identities yet — create one in the Identities station, then
              reference it here.
            </p>
          ) : picking ? (
            /* STEP 2 — the canonical ROW. Renders whatever profile.canonical holds
               (4 today, 8 once the parallel azimuth slice lands, N tomorrow): the
               row wraps + scrolls rather than assuming a count. */
            <>
              <div className="vi-gen-identity-views-head">
                <button
                  type="button"
                  className="vi-btn vi-btn-sm vi-btn-ghost"
                  disabled={busy}
                  onClick={() => {
                    setMsg(null);
                    setPickingSlug(null);
                  }}
                  title="Back to the identity list"
                >
                  ‹ identities
                </button>
                <span className="vi-gen-identity-note">
                  {/* HONEST COPY: says exactly why it's one — no implication that the
                      other views condition the generation, because they would not. */}
                  pick the ONE view that carries “{picking.name}” — only this image
                  conditions the generation, so choose the angle closest to the shot
                  you want
                </span>
              </div>
              <div className="vi-gen-identity-views" role="list">
                {picking.canonical.map((path, i) => (
                  <button
                    key={path}
                    type="button"
                    role="listitem"
                    className="vi-gen-identity-view"
                    disabled={disabled || busy}
                    onClick={() => void pickView(picking, i)}
                    title={`Carry “${picking.name}” with canonical ${viewLabel(picking, i)}`}
                  >
                    <img
                      className="vi-gen-identity-view-thumb"
                      src={mediaBytesUrl(path)}
                      alt={`${picking.name} canonical ${viewLabel(picking, i)}`}
                      loading="lazy"
                    />
                    {/* Real azimuth (degrees + nearest semantic name) when this
                        view has one; the honest ordinal ("view N") when it
                        doesn't — see viewLabel above. */}
                    <span className="vi-gen-identity-view-label">{viewLabel(picking, i)}</span>
                  </button>
                ))}
              </div>
            </>
          ) : (
            /* STEP 1 — choose an identity. */
            profiles.map((p) => {
              // A bind attaches ONE CANONICAL turntable view (a render of the 3D
              // model), not the raw reference_images — so the picker's
              // preview/count/title must describe canonical, or a user would be
              // told one thing and get another. A profile with no canonical set yet
              // (no 3D model rendered) is still listed — clicking it fails loudly —
              // but the affordance says so up front instead of implying it's ready.
              const hasCanonical = p.canonical.length > 0;
              const title = hasCanonical
                ? `Choose a carrying view for “${p.name}” (${p.canonical.length} 3D-model view${
                    p.canonical.length === 1 ? "" : "s"
                  } to pick from)`
                : `“${p.name}” has no 3D-model views yet — render its 3D model first`;
              return (
                <button
                  key={p.slug}
                  type="button"
                  className="vi-gen-identity-option"
                  disabled={disabled || busy}
                  onClick={() => chooseProfile(p)}
                  title={title}
                >
                  {hasCanonical ? (
                    <img
                      className="vi-gen-identity-option-thumb"
                      src={mediaBytesUrl(p.canonical[0])}
                      alt=""
                      loading="lazy"
                    />
                  ) : (
                    <span
                      className="vi-gen-identity-option-thumb vi-gen-identity-option-thumb-empty"
                      aria-hidden
                    >
                      ⚠
                    </span>
                  )}
                  <span className="vi-gen-identity-option-name">{p.name}</span>
                </button>
              );
            })
          )}
        </div>
      )}

      {msg && (
        <p className="vi-error" role="alert">
          {msg}
        </p>
      )}
    </div>
  );
}

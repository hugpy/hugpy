// IDENTITY PROFILES (studio stage (a)) — the durable "the reference set IS the
// identity" library, as a hook shared by every surface that touches a profile
// (the Clip Generate tab, the Movie composer, and the dedicated Identities
// station). One concept, backed by the server store:
//   • list    — GET  /video/identity-profiles → { profiles: [...] }
//   • create  — POST /video/identity-profiles { name, reference_images[1..4], notes? }
//   • update  — PATCH /video/identity-profiles/<slug> { name?, notes?, reference_images? }
//               (partial — an omitted key is left untouched server-side); renaming is
//               DISPLAY-ONLY, the slug never changes (see the backend docstring —
//               templates/specs reference a profile by slug, so re-slugging on rename
//               would silently strand them).
//   • remove  — DELETE /video/identity-profiles/<slug> (ARCHIVES, never erases)
//   • resolve — ingest a profile's saved reference paths back into MediaRefs so the
//               attach flow lands them in the SAME state a hand-picked upload does
//               (as if hand-picked — the durable form of the identity, not a copy).
//
// Data-not-throws, like useProjects/useStudioClips (from which the fetch idiom is
// cloned): a failed list just yields []; create returns a discriminated result so a
// caller can tell a duplicate-name (409) from other failures without a try/catch.
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import {
  hugpyConfig,
  identityProfileUrl,
  identityProfileSettingsUrl,
  identityVersionUrl,
  identityVersionActivateUrl,
} from "../../config";
import { mediaRefSchema } from "../../video/contract";
import type { MediaRef } from "../../video/contract";

// ── char360 S4 — VIDEO → CHARACTER VIEW-SET extraction ──────────────────────
// All-optional knobs for POST /video/identity-profiles/video-extract's
// `char360_params`; blank/omitted == the service default for every field (the
// Identities station's Advanced section only sends what the user actually set).
export const char360ParamsSchema = z
  .object({
    stride: z.number().optional(),
    yolo_model: z.string().optional(),
    min_h_frac: z.number().optional(),
    cluster_dist: z.number().optional(),
    min_faces: z.number().optional(),
  })
  .partial();
export type Char360Params = z.infer<typeof char360ParamsSchema>;

/** enqueue-only outcome of extractFromVideo — polling for the terminal
 *  result.ok/result.error lives in the caller (mirrors the /generate poll
 *  loop), same reason as CreateProfileResult: a flat shape, not a discriminated
 *  union, since the scoped tsconfig runs strictNullChecks OFF. */
export interface ExtractFromVideoResult {
  ok: boolean;
  jobId?: string;
  message?: string;
}

/** k94 — enqueue-only outcome of createFromVideo / createFromImages. The caller
 *  polls `videoJobUrl(jobId)`; `slug` is the profile the first character lands
 *  in (from-video: extras are `<slug>-2`, `<slug>-3`…; from-images: the profile
 *  already exists when this returns). Flat shape for the same strictNullChecks-
 *  off reason as ExtractFromVideoResult. */
export interface CreateIdentityJobResult {
  ok: boolean;
  jobId?: string;
  slug?: string;
  duplicate?: boolean;
  message?: string;
}

// ── wire schema (tolerant/passthrough — assert only what the UI reads) ──────────

// One TURNAROUND RECONSTRUCTION ("character sheet") a profile has accrued — an
// additive field the reconstruction flow (POST .../reconstruction) writes: an id,
// when it was made, the ordered generated view paths (positionally matching the
// requested view names: front / ¾ / profile / back, each served via mediaBytesUrl),
// and the prompt/seed it was rolled with. Tolerant/passthrough — every field but
// recon_id/views is optional so a lean record still parses.
// src/stations/studio/useIdentityProfiles.ts

// ── wire schema (tolerant/passthrough — assert only what the UI reads) ──────────

export const reconstructionSchema = z
  .object({
    recon_id: z.string(),
    created_at: z.number().nullable().optional(),
    // Lenient: accept missing views and default to empty array
    views: z.array(z.any()).nullable().optional().transform(v => v || []),
    prompt: z.string().nullable().optional(),
    seed: z.number().nullable().optional(),
    // Lenient: accept any string for mode so legacy ones don't break
    mode: z.string().nullable().optional(),
    frame_count: z.number().nullable().optional(),
    degrees_per_frame: z.number().nullable().optional(),
    mesh: z.any().nullable().optional(),
  })
  .passthrough();
export type Reconstruction = z.infer<typeof reconstructionSchema>;

// ── IDENTITY VERSIONS slice (dev/IDENTITY-VERSIONS-SLICE.md) — additive, every
// field defaulted so an old (pre-versions) profile payload still parses clean. ──

// One named render-set an identity has accrued — the clay base (minted on first
// Generate, always kept) plus N derived "textured"/"styled" versions. `kind` is
// lenient: any string that isn't a recognized kind quietly falls back to "clay"
// rather than failing the whole profile parse.
export const identityVersionSchema = z
  .object({
    version_id: z.string(),
    name: z.string().nullable().optional().transform(v => v || ""),
    kind: z
      .string()
      .nullable()
      .optional()
      .transform((v): "clay" | "textured" | "styled" => (v === "textured" || v === "styled" ? v : "clay")),
    recon_id: z.string().nullable().optional(),
    created_at: z.number().nullable().optional(),
    canonical: z.array(z.string()).nullable().optional().transform(v => v || []),
    notes: z.string().nullable().optional(),
  })
  .passthrough();
export type IdentityVersion = z.infer<typeof identityVersionSchema>;

// Per-identity generation defaults (the Settings panel + the happy-path bare
// `/generate` click). Every field individually defaulted so `gen_settings.parse({})`
// (an entirely absent block, on an old payload) yields the full happy-path object.
export const genSettingsSchema = z
  .object({
    texture: z.boolean().nullable().optional().transform(v => v ?? true),
    pose: z
      .string()
      .nullable()
      .optional()
      .transform((v): "none" | "t-pose" => (v === "t-pose" ? "t-pose" : "none")),
    frame_count: z.number().nullable().optional().transform(v => v ?? 72),
    fps: z.number().nullable().optional().transform(v => v ?? 24),
    width: z.number().nullable().optional().transform(v => v ?? 768),
    height: z.number().nullable().optional().transform(v => v ?? 768),
    auto_promote: z.boolean().nullable().optional().transform(v => v ?? true),
    front_ref: z.string().nullable().optional().transform(v => v ?? null),
    remove_background: z.boolean().nullable().optional().transform(v => v ?? true),
    // Per-identity VISION MODEL for the 3D-imaging front-select step. null (default)
    // == the fleet-default VL model (the 3B) — zero regression. A non-null value is an
    // image-text-to-text model key (validated server-side against the live registry).
    vision_model: z.string().nullable().optional().transform(v => v ?? null),
    // CLEANUP-PROMPT slice (C4 — Advanced-panel reachability): a positive-worded AVOID
    // instruction woven into the T-pose front render (e.g. "no object on her back, no
    // logos, clean bare back"). "" (default) == no steer, byte-identical to today.
    cleanup_prompt: z.string().nullable().optional().transform(v => v ?? ""),
    // A TRUE negative forwarded to the studio Wan-VACE render (reconstruction + the
    // mesh route's id_lock front render). "" (default) == today's exact call.
    negative_prompt: z.string().nullable().optional().transform(v => v ?? ""),
  })
  .passthrough();
export type GenSettings = z.infer<typeof genSettingsSchema>;

/** The happy-path defaults — a profile with no `gen_settings` block at all (every
 *  payload before this slice shipped) resolves to exactly this. */
export const DEFAULT_GEN_SETTINGS: GenSettings = genSettingsSchema.parse({});

export const identityProfileSchema = z
  .object({
    slug: z.string(),
    // Lenient: if any core fields are missing/null, safely default them
    name: z.string().nullable().optional().transform(v => v || "Unnamed"),
    reference_images: z.array(z.string()).nullable().optional().transform(v => v || []),
    created_at: z.number().nullable().optional(),
    notes: z.string().nullable().optional(),
    reconstructions: z.array(reconstructionSchema).nullable().optional().transform(v => v || []),
    canonical: z.array(z.string()).nullable().optional().transform(v => v || []),
    // ADDITIVE per-view angle provenance (dev/CHAR360-FEATURE-PLAN.md's parallel
    // slice) — POSITIONALLY ALIGNED to `canonical` (canonical_angles[i] describes
    // canonical[i]). OMITTED ENTIRELY on any profile from before the backend
    // backfill (identity_profiles.py:1556) — optional with no default-to-[]
    // transform, so "absent" stays distinguishable from "present but empty" for
    // the UI's honest-fallback check (see GenIdentityBar.tsx's viewLabel). An
    // individual element may be `null` (unknown for just that view) rather than a
    // guessed 0.0. The backend already drops the key on a length mismatch with
    // `canonical`, but the UI re-checks the lengths itself before trusting it
    // (never assume the wire honored its own contract).
    canonical_angles: z.array(z.number().nullable()).nullable().optional(),
    // versions[] / active_version / gen_settings may be ABSENT entirely on any
    // payload served before an API restart picks up the backend half of this
    // slice — all three default so the station renders exactly as it does today.
    versions: z.array(identityVersionSchema).nullable().optional().transform(v => v || []),
    active_version: z.string().nullable().optional().transform(v => v ?? null),
    gen_settings: genSettingsSchema
      .nullable()
      .optional()
      .transform(v => v ?? genSettingsSchema.parse({})),
  })
  .passthrough();
export type IdentityProfile = z.infer<typeof identityProfileSchema>;

const listResponseSchema = z.object({ profiles: z.array(identityProfileSchema) }).passthrough();
const createResponseSchema = z.object({ profile: identityProfileSchema }).passthrough();

/** create() outcome — a duplicate name (409) is distinguishable from any other
 *  failure WITHOUT a thrown exception. A FLAT shape (not a discriminated union):
 *  the scoped tsconfig runs strictNullChecks OFF, under which the repo's Result
 *  narrowing is unreliable (hence its okValue/errorOf helpers) — so every field is
 *  directly accessible off `ok` instead of relying on control-flow narrowing. */
export interface CreateProfileResult {
  ok: boolean;
  profile?: IdentityProfile;
  duplicate?: boolean;
  message?: string;
}

/** update() outcome — same flat (not discriminated-union) shape as
 *  CreateProfileResult, for the same strictNullChecks-off narrowing reason. A
 *  PATCH never 409s (renaming never re-slugs, so it can't collide with another
 *  profile's slug) — `message` alone carries any failure (404 unknown slug,
 *  400 bad body, etc.). */
export interface UpdateProfileResult {
  ok: boolean;
  profile?: IdentityProfile;
  message?: string;
}

/** Outcome shared by every version mutation (activate / rename / archive) — flat
 *  (not a discriminated union), same reason as CreateProfileResult/UpdateProfileResult.
 *  `profile` is populated whenever the caller could reconcile the identity's new
 *  state (either the response echoed it, or a same-slug re-GET did) — never
 *  required for `ok` to be true, since the wire contract doesn't fix a response
 *  envelope for these three (only the settings PATCH is documented as → {profile}). */
export interface VersionMutationResult {
  ok: boolean;
  profile?: IdentityProfile;
  message?: string;
}

export interface IdentityProfilesState {
  profiles: IdentityProfile[];
  loading: boolean;
  error: string | null;
  /** Re-pull the list (foreground). */
  reload: () => void;
  /** Save a new profile from a name + already-attached reference URIs. */
  create: (name: string, referenceImages: string[], notes?: string) => Promise<CreateProfileResult>;
  /** Partially edit an existing profile — an omitted field is left untouched
   *  server-side. Renaming is display-only (the slug never changes). Passing an
   *  empty `referenceImages` array is rejected by the server (an identity keeps
   *  >=1 reference); omit the field (undefined) to leave the current set alone. */
  update: (
    slug: string,
    fields: { name?: string; notes?: string; referenceImages?: string[] },
  ) => Promise<UpdateProfileResult>;
  /** Archive a profile by slug (never-delete: the server moves it under _deleted). */
  remove: (slug: string) => Promise<boolean>;
  /** Ingest a profile's saved reference paths → MediaRefs (drops any that fail), so
   *  the caller can attach them into its OWN reference state — as if hand-picked. */
  resolveRefs: (profile: IdentityProfile) => Promise<MediaRef[]>;
  /** Ingest a profile's CANONICAL turntable views → MediaRefs, in order (drops any
   *  that fail to ingest, but never silently substitutes a different set). This is
   *  `profile.canonical` — the top-level field, which already reflects the profile's
   *  ACTIVE version (see the `canonical` field doc above) — NOT a walk of
   *  `versions[]`. Distinct from `resolveRefs` (which resolves the raw, messy
   *  `reference_images` source uploads) so Studio's `IdentityProfileControls` /
   *  `StudioMovieComposer` — both still on `resolveRefs` — are byte-for-byte
   *  unaffected by this addition. An EMPTY `profile.canonical` resolves to an EMPTY
   *  array (never falls back to `reference_images`) — the caller must treat that as
   *  an honest failure, not attach nothing silently. */
  resolveCanonical: (profile: IdentityProfile) => Promise<MediaRef[]>;
  /** Ingest ONE of a profile's CANONICAL views (by index into `profile.canonical`)
   *  → a MediaRef, or null when the index is out of range or the image won't ingest.
   *
   *  Additive sibling to `resolveCanonical`, for the Generate station's
   *  one-carrying-view bind: the generate runner consumes only the FIRST image part
   *  (imagegen.py:117 `start_frame = image_paths[0]`), so a bind attaches exactly one
   *  view — resolving the whole set to keep a single ref would ingest N-1 images for
   *  nothing (visible cost once canonical grows to 8 azimuth views). Returns null
   *  rather than throwing: the caller turns it into a loud, human message and NEVER
   *  falls back to `reference_images`. `resolveRefs` / `resolveCanonical` are both
   *  untouched — Studio is byte-for-byte unaffected. */
  resolveCanonicalView: (
    profile: IdentityProfile,
    index: number,
  ) => Promise<MediaRef | null>;
  /** Partially save this identity's generation settings (texture/pose/turntable
   *  geometry/auto-promote/front-ref/background-removal) — an omitted key is left
   *  untouched server-side, same partial-PATCH idiom as `update`. Prefills the
   *  happy-path bare `/generate` click. */
  saveSettings: (slug: string, fields: Partial<GenSettings>) => Promise<UpdateProfileResult>;
  /** Make one version the identity's ACTIVE version — the resolver takes ITS
   *  canonical as the id_lock DNA for any future generation that doesn't name a
   *  specific `identity_version`. */
  activateVersion: (slug: string, versionId: string) => Promise<VersionMutationResult>;
  /** Rename one version (display-only, mirrors profile rename). */
  renameVersion: (slug: string, versionId: string, name: string) => Promise<VersionMutationResult>;
  /** Archive one version (never-delete: recoverable, not erased). The server
   *  refuses this with a 400 for the base (clay) version and for the currently
   *  active version — surfaced via `message`. */
  archiveVersion: (slug: string, versionId: string) => Promise<VersionMutationResult>;
  /** char360 S4 — enqueue a video → character-view-set extraction: POST
   *  /video/identity-profiles/video-extract { source, target, char360_params? }
   *  → { job_id }. `target` is "create" (mints new profile(s)) or an existing
   *  profile's slug (adds views to it). ENQUEUE ONLY — this just returns the
   *  job id; the caller polls `videoJobUrl(jobId)` (GET /video/jobs/<id>) until
   *  `result.ok` is true/false, same poll shape as the /generate flow, then
   *  calls `reload()` on success (there is no slug list in the payload). */
  extractFromVideo: (
    source: MediaRef,
    target: "create" | string,
    params?: Char360Params,
  ) => Promise<ExtractFromVideoResult>;
  /** k94 ONE PATH — a video in, one bindable profile per detected character out
   *  (ONE chained char360 + GLB job). ENQUEUE ONLY; poll `videoJobUrl(jobId)`. */
  createFromVideo: (source: MediaRef, name: string) => Promise<CreateIdentityJobResult>;
  /** k94 ONE PATH — a few photos in, the profile is created immediately and the
   *  full 3D build (mesh → turntable → canonical) is enqueued in the same request.
   *  A duplicate name is `duplicate: true` (nothing enqueued). */
  createFromImages: (sources: MediaRef[], name: string) => Promise<CreateIdentityJobResult>;
}

/** Ingest ONE server path (already under the storage jail — a profile only ever
 *  stores jailed paths) into a resolved MediaRef, or null if it can't be resolved.
 *  Reuses POST /video/ingest, the exact first hop the upload flow already runs. */
async function ingestPath(path: string): Promise<MediaRef | null> {
  const res = await request<unknown>(hugpyConfig.videoIngestUrl, {
    method: "POST",
    body: JSON.stringify({ path }),
    headers: { "Content-Type": "application/json" },
    meta: { specKey: "studio", operation: "identity.profile.ingest" },
  });
  if (!res.ok) return null;
  const parsed = mediaRefSchema.safeParse(okValue(res));
  return parsed.success ? parsed.data : null;
}

export function useIdentityProfiles(): IdentityProfilesState {
  const [profiles, setProfiles] = useState<IdentityProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const mounted = useRef(true);
  const inFlight = useRef(false);

  const load = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const res = await request<unknown>(hugpyConfig.identityProfilesUrl, {
        meta: { specKey: "studio", operation: "identity.profiles.list" },
      });
      if (!mounted.current) return;
      if (!res.ok) {
        setLoading(false);
        setError(describeAppError(errorOf(res)));
        return;
      }
      const parsed = listResponseSchema.safeParse(okValue(res));
      if (!parsed.success) {
        console.error("Profile parsing error:", parsed.error); // Add this line!
        setLoading(false);
        setError("Malformed identity-profiles response.");
        return;
      }
      setProfiles(parsed.data.profiles);
      setLoading(false);
      setError(null);
    } finally {
      inFlight.current = false;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void load();
    return () => {
      mounted.current = false;
    };
  }, [load]);

  const create = useCallback<IdentityProfilesState["create"]>(
    async (name, referenceImages, notes) => {
      const res = await request<unknown>(hugpyConfig.identityProfilesUrl, {
        method: "POST",
        body: JSON.stringify({
          name,
          reference_images: referenceImages,
          ...(notes ? { notes } : {}),
        }),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "identity.profile.create" },
      });
      if (!res.ok) {
        const err = errorOf(res);
        const duplicate = err.kind === "server" && err.status === 409;
        return { ok: false, duplicate, message: describeAppError(err) };
      }
      const parsed = createResponseSchema.safeParse(okValue(res));
      if (!parsed.success) {
        return { ok: false, duplicate: false, message: "Malformed create response." };
      }
      const profile = parsed.data.profile;
      // Optimistic prepend (newest-first, matching the server sort) so the new
      // profile is pickable immediately; a later reload reconciles.
      if (mounted.current) setProfiles((prev) => [profile, ...prev.filter((p) => p.slug !== profile.slug)]);
      return { ok: true, profile };
    },
    [],
  );

  const update = useCallback<IdentityProfilesState["update"]>(async (slug, fields) => {
    const body: Record<string, unknown> = {};
    if (fields.name !== undefined) body.name = fields.name;
    if (fields.notes !== undefined) body.notes = fields.notes;
    if (fields.referenceImages !== undefined) body.reference_images = fields.referenceImages;
    const res = await request<unknown>(identityProfileUrl(slug), {
      method: "PATCH",
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "studio", operation: "identity.profile.update" },
    });
    if (!res.ok) {
      return { ok: false, message: describeAppError(errorOf(res)) };
    }
    // The PATCH response is the same {profile} envelope create() parses.
    const parsed = createResponseSchema.safeParse(okValue(res));
    if (!parsed.success) {
      return { ok: false, message: "Malformed update response." };
    }
    const profile = parsed.data.profile;
    // Replace in place (slug is stable across a PATCH, so this is a straight swap,
    // never a re-sort); a later reload still reconciles.
    if (mounted.current) {
      setProfiles((prev) => prev.map((p) => (p.slug === profile.slug ? profile : p)));
    }
    return { ok: true, profile };
  }, []);

  const remove = useCallback(async (slug: string) => {
    const res = await request<unknown>(identityProfileUrl(slug), {
      method: "DELETE",
      meta: { specKey: "studio", operation: "identity.profile.delete" },
    });
    if (!res.ok) return false;
    if (mounted.current) setProfiles((prev) => prev.filter((p) => p.slug !== slug));
    return true;
  }, []);

  const resolveRefs = useCallback(async (profile: IdentityProfile) => {
    const refs = await Promise.all(profile.reference_images.map((p) => ingestPath(p)));
    return refs.filter((r): r is MediaRef => r != null);
  }, []);

  // Additive sibling to resolveRefs — resolves the profile's CANONICAL turntable
  // views (top-level `profile.canonical`, already active-version-aware) instead of
  // the raw `reference_images` source uploads. Order is preserved (ref_00..ref_03,
  // 90° apart) since it is semantic for turntable views — Promise.all over an
  // already-ordered array preserves index order regardless of individual ingest
  // timing. resolveRefs is left completely untouched so Studio's
  // IdentityProfileControls / StudioMovieComposer keep today's exact behavior.
  const resolveCanonical = useCallback(async (profile: IdentityProfile) => {
    const refs = await Promise.all(profile.canonical.map((p) => ingestPath(p)));
    return refs.filter((r): r is MediaRef => r != null);
  }, []);

  // Additive sibling to resolveCanonical — ingests exactly ONE canonical view (the
  // Generate station's carrying view). Bounds-checked because the index comes from a
  // UI row that may have been rendered against a since-refreshed profile list: an
  // out-of-range index must resolve to null (→ a loud caller message), never to the
  // wrong view. resolveRefs / resolveCanonical are untouched.
  const resolveCanonicalView = useCallback(
    async (profile: IdentityProfile, index: number) => {
      const path = profile.canonical[index];
      if (path == null) return null;
      return await ingestPath(path);
    },
    [],
  );

  const saveSettings = useCallback<IdentityProfilesState["saveSettings"]>(async (slug, fields) => {
    const res = await request<unknown>(identityProfileSettingsUrl(slug), {
      method: "PATCH",
      body: JSON.stringify(fields),
      headers: { "Content-Type": "application/json" },
      meta: { specKey: "studio", operation: "identity.profile.settings" },
    });
    if (!res.ok) {
      return { ok: false, message: describeAppError(errorOf(res)) };
    }
    // Documented envelope for this one — → {profile}, the same shape update() parses.
    const parsed = createResponseSchema.safeParse(okValue(res));
    if (!parsed.success) {
      return { ok: false, message: "Malformed settings response." };
    }
    const profile = parsed.data.profile;
    if (mounted.current) {
      setProfiles((prev) => prev.map((p) => (p.slug === profile.slug ? profile : p)));
    }
    return { ok: true, profile };
  }, []);

  /** Shared runner for the three version verbs below. The wire contract only fixes
   *  a response envelope for the settings PATCH — activate/rename/archive aren't
   *  documented, so this tries the fast path (the response itself echoes {profile})
   *  and otherwise falls back to a same-slug re-GET (`identityProfileUrl`, already
   *  documented as → {profile}) so local state reconciles regardless of what shape
   *  the backend ships once it's live. */
  const runVersionMutation = useCallback(
    async (slug: string, url: string, method: string, body?: unknown): Promise<VersionMutationResult> => {
      const res = await request<unknown>(url, {
        method,
        ...(body !== undefined
          ? { body: JSON.stringify(body), headers: { "Content-Type": "application/json" } }
          : {}),
        meta: { specKey: "studio", operation: "identity.version.mutate" },
      });
      if (!res.ok) {
        return { ok: false, message: describeAppError(errorOf(res)) };
      }
      const direct = createResponseSchema.safeParse(okValue(res));
      let profile: IdentityProfile | undefined = direct.success ? direct.data.profile : undefined;
      if (!profile) {
        const refetch = await request<unknown>(identityProfileUrl(slug), {
          meta: { specKey: "studio", operation: "identity.profile.refetch" },
        });
        if (refetch.ok) {
          const rp = createResponseSchema.safeParse(okValue(refetch));
          if (rp.success) profile = rp.data.profile;
        }
      }
      if (profile && mounted.current) {
        const resolved = profile;
        setProfiles((prev) => prev.map((p) => (p.slug === resolved.slug ? resolved : p)));
      }
      return { ok: true, profile };
    },
    [],
  );

  const extractFromVideo = useCallback<IdentityProfilesState["extractFromVideo"]>(
    async (source, target, params) => {
      const parsedParams = params ? char360ParamsSchema.safeParse(params) : undefined;
      const body: Record<string, unknown> = {
        source,
        target,
        ...(parsedParams && parsedParams.success && Object.keys(parsedParams.data).length > 0
          ? { char360_params: parsedParams.data }
          : {}),
      };
      const res = await request<unknown>(hugpyConfig.videoExtractUrl, {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "identity.video_extract.enqueue" },
      });
      if (!res.ok) {
        return { ok: false, message: describeAppError(errorOf(res)) };
      }
      const enq = okValue(res) as { job_id?: string };
      if (!enq.job_id) {
        return { ok: false, message: "Malformed video-extract response." };
      }
      return { ok: true, jobId: enq.job_id };
    },
    [],
  );

  const createFromVideo = useCallback<IdentityProfilesState["createFromVideo"]>(
    async (source, name) => {
      const res = await request<unknown>(hugpyConfig.identityFromVideoUrl, {
        method: "POST",
        body: JSON.stringify({ source, name }),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "identity.from_video.enqueue" },
      });
      if (!res.ok) return { ok: false, message: describeAppError(errorOf(res)) };
      const enq = okValue(res) as { job_id?: string; slug?: string };
      if (!enq.job_id) return { ok: false, message: "Malformed from-video response." };
      return { ok: true, jobId: enq.job_id, slug: enq.slug };
    },
    [],
  );

  const createFromImages = useCallback<IdentityProfilesState["createFromImages"]>(
    async (sources, name) => {
      const res = await request<unknown>(hugpyConfig.identityFromImagesUrl, {
        method: "POST",
        body: JSON.stringify({ sources, name }),
        headers: { "Content-Type": "application/json" },
        meta: { specKey: "studio", operation: "identity.from_images.enqueue" },
      });
      if (!res.ok) {
        const err = errorOf(res);
        const status = (err as { status?: number } | null)?.status;
        const message = describeAppError(err);
        return { ok: false, duplicate: status === 409 || /already exists/i.test(message), message };
      }
      const enq = okValue(res) as { job_id?: string; slug?: string; profile?: unknown };
      if (!enq.job_id || !enq.slug) return { ok: false, message: "Malformed from-images response." };
      // The profile exists NOW — fold it into the list so the tab can select it
      // before the build finishes (the poll's reload refreshes it afterwards).
      const parsed = identityProfileSchema.safeParse(enq.profile);
      if (parsed.success && mounted.current) {
        const created = parsed.data;
        setProfiles((prev) => (prev.some((p) => p.slug === created.slug) ? prev : [created, ...prev]));
      }
      return { ok: true, jobId: enq.job_id, slug: enq.slug };
    },
    [],
  );

  const activateVersion = useCallback<IdentityProfilesState["activateVersion"]>(
    (slug, versionId) => runVersionMutation(slug, identityVersionActivateUrl(slug, versionId), "POST"),
    [runVersionMutation],
  );

  const renameVersion = useCallback<IdentityProfilesState["renameVersion"]>(
    (slug, versionId, name) =>
      runVersionMutation(slug, identityVersionUrl(slug, versionId), "PATCH", { name }),
    [runVersionMutation],
  );

  const archiveVersion = useCallback<IdentityProfilesState["archiveVersion"]>(
    (slug, versionId) => runVersionMutation(slug, identityVersionUrl(slug, versionId), "DELETE"),
    [runVersionMutation],
  );

  return {
    profiles,
    loading,
    error,
    reload: () => void load(),
    create,
    update,
    remove,
    resolveRefs,
    resolveCanonical,
    resolveCanonicalView,
    saveSettings,
    activateVersion,
    renameVersion,
    archiveVersion,
    extractFromVideo,
    createFromVideo,
    createFromImages,
  };
}

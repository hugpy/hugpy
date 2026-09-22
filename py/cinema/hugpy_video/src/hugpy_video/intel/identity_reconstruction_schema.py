"""Identity RECONSTRUCTION job schema (studio stage (b)) — the durable, bus-carried
intent for "generate an identity-locked turnaround of a character from its reference
images + description, for approval".

A reconstruction job renders ONE identity-locked still per requested VIEW
(``front`` / ``three_quarter`` / ``profile`` / ``back`` by default), then hands the
produced stills to ``identity_profiles.attach_reconstruction`` so they land in the
identity's own directory awaiting the operator's approval. It is an ORCHESTRATOR job
(like ``generate_studio_movie``): the route enqueues ONE of these and the runner
(``runners/identity_reconstruction.py``) drives the per-view renders INLINE through
the shared render primitive, so a single ``job_id`` covers the whole set.

House style mirrors ``studio.job`` / ``studio_movie_schema``: a frozen, JSON-safe,
validate-at-construction spec built ONLY via ``make_identity_reconstruction``; the
bus rehydrates it through ``identity_reconstruction_from_dict`` (reconstruct +
RE-VALIDATE). All fields are primitives / string tuples so ``asdict`` -> ``json``
round-trips cleanly with zero enum/dataclass ceremony.

Geometry defaults to the Wan-VACE id_lock ceiling (<=480p — a 512² id_lock fails
``no_capable_model``): a square 480×480 @ 16fps is a sensible portrait default. The
budget defaults to ``None`` = AUTOFIT (blank budget sizes to the serving worker's
measured free VRAM at render time; never a guaranteed-fail low guess).

No pathlib anywhere. os.path only (none needed here — pure data).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# The subject reference images cap — an id_lock render consumes at most this many
# (diffusers 0.39 prepends each as a VACE reference latent). Mirrors
# ``studio.job._MAX_CANONICAL_IMAGES`` / ``identity_profiles.MAX_CANONICAL_IMAGES``.
_MAX_CANONICAL_IMAGES = 4

# The default set of turnaround views (order preserved end-to-end). A single-view
# request (e.g. ``["front"]``) is valid — the cheap way to verify one render first.
DEFAULT_VIEWS: Tuple[str, ...] = ("front", "three_quarter", "profile", "back")

# Wan-VACE id_lock geometry ceiling is 480p; a square portrait still default @ 16fps.
_DEFAULT_WIDTH, _DEFAULT_HEIGHT, _DEFAULT_FPS = 480, 480, 16

# Reconstruction MODE (shared contract):
#   "sheet"     — the EXISTING behaviour: N INDEPENDENT view-stills (one id_lock clip
#                 per named view, only frame 0 kept). The default — wire shape unchanged.
#   "turntable" — NEW: ONE id_lock ORBIT clip whose EVERY frame is kept, each frame a
#                 degree-view; the record's ordered ``views`` hold the frames in angular
#                 order. The runner branches on this.
RECON_MODES: Tuple[str, ...] = ("sheet", "turntable", "angle-ring")
_DEFAULT_MODE = "sheet"

# Turntable frame cap: a studio id_lock clip is spine-derived (~2s, model-capped), so at
# <=16fps it yields well under this. A HIGH cap (never a truncator on the happy path) so
# the frame-extract runner keeps EVERY frame of the orbit clip as a dense angular sequence.
_DEFAULT_TURNTABLE_MAX_FRAMES = 240


@dataclass(frozen=True)
class IdentityReconstructionSpec:
    """Frozen currency of an ``identity_reconstruction`` bus job.

        slug             the identity profile this reconstruction belongs to (the
                         store key ``attach_reconstruction`` writes back to).
        recon_id         the minted id of THIS turnaround set (the route mints it and
                         returns it alongside the job_id so the UI can correlate).
        source_images the ORDERED, jailed abs paths of the subject reference
                         image(s) the id_lock render conditions on (1..4). CANONICAL —
                         they define the identity. The route resolves + validates them
                         (canonical set preferred over the raw uploads) before enqueue.
        base_prompt      the extra description woven into every view prompt (the
                         profile's notes by default; may be empty).
        negative_prompt  a TRUE negative forwarded to the studio Wan-VACE render alongside
                         base_prompt (CLEANUP-PROMPT slice). base_prompt already covers
                         POSITIVE steering here (an avoid instruction can ride base_prompt),
                         so the reconstruction spec adds only the NEGATIVE channel. Default
                         "" -> today's exact call (defaults-are-promises).
        views            the ORDERED tuple of view names to render (>=1). A view-
                         specific prompt is derived per view in the runner.
        seed             base render seed (all views share it; the view differs by
                         prompt, not seed, so the identity stays put).
        width/height/fps id_lock render geometry (<=480p ceiling — see module header).
        vram_budget_gb   AUTOFIT tier. ``None`` (default) = size to the serving
                         worker's measured free VRAM at render time; a number PINS it.
        mode             "sheet" (default — N independent view-stills, existing path)
                         or "turntable" (one 360° orbit clip, every frame kept). The
                         runner branches on this; absent => "sheet" (backward-compat).
        turntable_max_frames  turntable-only hint: the frame-extract cap (every frame
                         of the orbit clip is kept up to this). Ignored in "sheet" mode.
    """
    slug: str
    recon_id: str
    source_images: Tuple[str, ...]
    views: Tuple[str, ...] = DEFAULT_VIEWS
    base_prompt: str = ""
    # additive (CLEANUP-PROMPT slice): a TRUE negative forwarded to the studio Wan-VACE
    # render alongside base_prompt. Default "" -> byte-identical to today.
    negative_prompt: str = ""
    seed: int = 0
    width: int = _DEFAULT_WIDTH
    height: int = _DEFAULT_HEIGHT
    fps: int = _DEFAULT_FPS
    vram_budget_gb: Optional[float] = None
    mode: str = _DEFAULT_MODE
    turntable_max_frames: int = _DEFAULT_TURNTABLE_MAX_FRAMES
    # Angle-ring specific properties
    angle_step_deg: Optional[int] = None
    elevations_deg: Optional[tuple[int, ...]] = None
    
    def __post_init__(self):
        if not self.slug or not self.recon_id:
            raise ValueError("slug and recon_id are required")
        if self.mode not in ("sheet", "turntable", "angle-ring"):
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.mode == "angle-ring":
            if not self.angle_step_deg or self.angle_step_deg <= 0:
                raise ValueError("angle_step_deg must be > 0 for angle-ring")

def make_identity_reconstruction(
    *,
    slug: str,
    recon_id: str,
    source_images: Tuple[str, ...],
    views: Optional[Tuple[str, ...]] = None,
    base_prompt: str = "",
    negative_prompt: str = "",
    seed: int = 0,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    fps: int = _DEFAULT_FPS,
    vram_budget_gb: Optional[float] = None,
    mode: str = _DEFAULT_MODE,
    turntable_max_frames: int = _DEFAULT_TURNTABLE_MAX_FRAMES,
    angle_step_deg: Optional[int] = None,
    elevations_deg: Optional[Sequence[int]] = None,
) -> IdentityReconstructionSpec:
    """Validate every field and build the frozen ``IdentityReconstructionSpec``.
    Raises ``ValueError``/``TypeError`` LOCALLY on any structural violation (house
    discipline: a structurally-invalid spec is caller error caught at the boundary,
    never carried across the bus). Runtime policy failures (an unroutable render, an
    archived profile) are NOT validated here — they surface as errors-as-data from the
    runner / store."""
    if not (isinstance(slug, str) and slug.strip()):
        raise ValueError(f"slug must be a non-empty string; got {slug!r}")
    if not (isinstance(recon_id, str) and recon_id.strip()):
        raise ValueError(f"recon_id must be a non-empty string; got {recon_id!r}")

    # reference images: coerce list/tuple -> tuple; each a non-empty string; 1..4.
    if isinstance(source_images, (list, tuple)):
        source_images = tuple(source_images)
    else:
        raise ValueError(
            f"source_images must be a list/tuple of paths; got {source_images!r}")
    if not source_images:
        raise ValueError("at least one reference_image is required")
    for i, r in enumerate(source_images):
        if not (isinstance(r, str) and r.strip()):
            raise ValueError(f"source_images[{i}] must be a non-empty string; got {r!r}")
    if len(source_images) > _MAX_CANONICAL_IMAGES:
        raise ValueError(
            f"at most {_MAX_CANONICAL_IMAGES} source_images are accepted; "
            f"got {len(source_images)}")

    # views: None -> the default turnaround set; coerce list/tuple -> tuple; >=1;
    # each a non-empty string.
    if views is None:
        views = DEFAULT_VIEWS
    if isinstance(views, (list, tuple)):
        views = tuple(views)
    else:
        raise ValueError(f"views must be a list/tuple of view names; got {views!r}")
    if not views:
        raise ValueError("at least one view is required")
    for i, v in enumerate(views):
        if not (isinstance(v, str) and v.strip()):
            raise ValueError(f"views[{i}] must be a non-empty string; got {v!r}")

    if base_prompt is not None and not isinstance(base_prompt, str):
        raise ValueError(f"base_prompt must be a string or None; got {base_prompt!r}")
    base_prompt = base_prompt or ""
    # negative_prompt: additive TRUE-negative channel (CLEANUP-PROMPT slice), validated
    # exactly like base_prompt — a str (None coerces to ""), default "" -> today's call.
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValueError(f"negative_prompt must be a string or None; got {negative_prompt!r}")
    negative_prompt = negative_prompt or ""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int; got {seed!r}")
    for name, val in (("width", width), ("height", height), ("fps", fps)):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{name} must be a positive int; got {val!r}")
    # AUTOFIT: None is a LEGAL value (blank -> fit to the worker's free VRAM at render
    # time). A NUMBER is the manual override and must be positive.
    if vram_budget_gb is not None:
        if not isinstance(vram_budget_gb, (int, float)) or isinstance(vram_budget_gb, bool) \
                or vram_budget_gb <= 0:
            raise ValueError(
                f"vram_budget_gb must be a positive number or None (autofit); "
                f"got {vram_budget_gb!r}")

    # mode: absent/None -> the default "sheet" (backward-compat); else one of RECON_MODES.
    if mode is None:
        mode = _DEFAULT_MODE
    if not (isinstance(mode, str) and mode in RECON_MODES):
        raise ValueError(f"mode must be one of {list(RECON_MODES)}; got {mode!r}")

    # turntable_max_frames: a positive int cap (only consulted in turntable mode).
    if not isinstance(turntable_max_frames, int) or isinstance(turntable_max_frames, bool) \
            or turntable_max_frames <= 0:
        raise ValueError(
            f"turntable_max_frames must be a positive int; got {turntable_max_frames!r}")

    return IdentityReconstructionSpec(
        slug=slug,
        recon_id=recon_id,
        source_images=source_images,
        views=views,
        base_prompt=base_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        vram_budget_gb=(float(vram_budget_gb) if vram_budget_gb is not None else None),
        mode=mode,
        turntable_max_frames=turntable_max_frames,
        # angle-ring geometry (forwarded so the spec's __post_init__ can validate
        # angle_step_deg>0; None for sheet/turntable). elevations normalized to a tuple.
        angle_step_deg=angle_step_deg,
        elevations_deg=(tuple(elevations_deg) if elevations_deg is not None else None),
    )


def identity_reconstruction_from_dict(d: dict) -> IdentityReconstructionSpec:
    """Rebuild an ``IdentityReconstructionSpec`` from its ``asdict`` form THROUGH the
    validating factory (deserialize-then-revalidate, like every other studio spec).
    Registered in ``media_bus.SPEC_DESERIALIZERS`` under ``"identity_reconstruction"``."""
    return make_identity_reconstruction(
        slug=d["slug"],
        recon_id=d["recon_id"],
        source_images=d.get("source_images") or (),
        views=d.get("views"),
        base_prompt=d.get("base_prompt", ""),
        # additive (CLEANUP-PROMPT slice): round-trip the negative channel; an OLD spec has
        # no key -> "" (== today's byte-identical render).
        negative_prompt=d.get("negative_prompt", ""),
        seed=d.get("seed", 0),
        width=d.get("width", _DEFAULT_WIDTH),
        height=d.get("height", _DEFAULT_HEIGHT),
        fps=d.get("fps", _DEFAULT_FPS),
        vram_budget_gb=d.get("vram_budget_gb"),
        # Backward-compat: an old serialized spec has no ``mode`` -> "sheet".
        mode=d.get("mode") or _DEFAULT_MODE,
        turntable_max_frames=d.get("turntable_max_frames", _DEFAULT_TURNTABLE_MAX_FRAMES),
        # Preserve angle-ring geometry across the bus (asdict->from_dict), or an
        # angle-ring spec would re-validate with angle_step_deg=None and be rejected.
        angle_step_deg=d.get("angle_step_deg"),
        elevations_deg=d.get("elevations_deg"),
    )

@dataclass(frozen=True)
class IdentitySingleViewRegenSpec:
    slug: str
    recon_id: str
    view_id: str
    prompt: str
    seed: int
    use_neighbors: bool
    # Will be populated by the router with the actual neighbor image URIs
    neighbor_images: tuple[str, ...] = ()
    
@dataclass(frozen=True)
class IdentityMeshSpec:
    """Frozen, JSON-safe currency of an ``identity_mesh_build`` bus job — the durable
    intent for "build a 3D mesh (+ optional turntable) of an identity from its
    reference images, on the remote GPU render service".

    Central has NO GPU: the runner (``runners/identity_render_relay.py``) RELAYS this
    over HTTP to the ``IDENTITY_RENDER_URL`` service. Every field is a primitive /
    string tuple so ``asdict`` -> ``json`` round-trips cleanly and the bus can rehydrate
    it through ``identity_mesh_from_dict`` (reconstruct + RE-VALIDATE).

    EXISTING fields (``view_ids``/``backend``/``workflow``/``output_format``) are kept
    for backward-compat — the old ComfyUI stub runner and ``make_mesh_reconstruction_spec``
    still construct them. NEW (additive, all defaulted):

        view_sources     ordered ``((view_name, abs_path), ...)`` pairs assigning a
                         profile reference/canonical image to a cardinal view
                         (front/right/back/left). ``front`` is required by Hunyuan3D-2mv;
                         the route jails every path to the profile's own refs.
        seed/num_inference_steps/octree_resolution/texture  mesh_params for the service.
        chain_turntable  when True (default) the service renders the mesh AND a real
                         Blender turntable orbit in one job (kind=``mesh_and_turntable``);
                         False = mesh only (kind=``mesh_build``).
        frame_count/fps/width/height/elevation_deg/transparent  turntable_params.
        auto_promote      when True (default False) the relay, AFTER a successful
                         turntable attach AND ONLY when the profile's canonical set is
                         still EMPTY, promotes the 4 cardinal turntable frames into
                         ``canonical`` (the ONE-CLICK full-identity template sets this;
                         a curated canonical set is never clobbered). A promotion
                         failure never fails the job.
        view_candidates  ordered tuple of abs paths — the profile's OTHER existing
                         source reference images, populated by the route ONLY when the
                         caller did NOT explicitly assign a front (an explicit
                         ``views.front`` disables auto-selection: empty tuple). When
                         this has >=2 entries, the relay runner asks the fleet vision
                         amenity which candidate shows the character's FULL BODY and,
                         if one qualifies, swaps it in as the mesh ``front`` view
                         BEFORE the render-service POST (see
                         ``runners/identity_render_relay.py``). Never fails the job —
                         a vision miss/error just keeps the existing default front.
    """
    slug: str
    recon_id: str
    view_ids: tuple[str, ...] = ()
    backend: str = "comfyui"
    workflow: str = "hunyuan3d-2mv"
    output_format: str = "glb"
    # --- additive: remote render-service relay contract ---------------------- #
    view_sources: tuple[tuple[str, str], ...] = ()
    seed: int = 12345
    num_inference_steps: int = 30
    octree_resolution: int = 380
    texture: bool = False
    chain_turntable: bool = True
    frame_count: int = 72
    fps: int = 24
    width: int = 768
    height: int = 768
    elevation_deg: float = 8.0
    transparent: bool = False
    # additive: one-click full-identity template flag (see docstring above)
    auto_promote: bool = False
    # additive: fleet-VLM front auto-selection candidates (see docstring above)
    view_candidates: tuple[str, ...] = ()
    # additive (VERSIONS slice): pose-normalization stage. "none" (default) meshes the
    # input pose as-is; "t-pose" makes the relay render ONE id_lock T-pose still FIRST
    # and use it as the mesh front (clears crossed-arm occlusion). Honest degrade — a
    # failed pose render never fails the job, it falls back to the normal front.
    pose: str = "none"
    # additive (per-identity VISION MODEL, operator-requested): the VL model key the
    # relay's FRONT-SELECT step (runners/identity_render_relay._select_front_view) asks
    # "does this show the FULL body?" with, before meshing. None (default) == the
    # fleet-default VL model (the 3B): the relay sends NO ``model`` field on its
    # /ml/vision POST, byte-identical to before this field existed (zero regression;
    # defaults-are-promises). A non-null key routes that vision call at THAT model (e.g. a
    # 7B) — resolved by the route with precedence request-body > the identity's
    # gen_settings.vision_model > None. Structural-only here (None or a non-empty string);
    # the LIVE image-text-to-text registry check lives in the store (set_gen_settings) so
    # the spec factory stays import-cheap on the bus rehydrate path.
    vision_model: Optional[str] = None
    # additive (CLEANUP-PROMPT slice, operator-requested): steer the FRONT render so the
    # meshed front never carries an unwanted baked-in element (Hunyuan3D is image-to-3D —
    # you cannot prompt the mesh step, so steering must happen on the source view that gets
    # meshed). TWO channels, both default "" -> byte-identical to today (defaults-are-
    # promises):
    #   cleanup_prompt   a POSITIVE-worded avoid instruction (e.g. "no object on her back,
    #                    clean bare back") WOVEN INTO the T-pose front render prompt. A
    #                    cleanup clause removes a prop; it does NOT fight the T-pose stance,
    #                    so (unlike an identity DESCRIPTION) it is allowed into _TPOSE_PROMPT.
    #                    Empty -> the current constant verbatim.
    #   negative_prompt  a TRUE negative forwarded to the studio Wan-VACE render (which
    #                    ALREADY accepts negative_prompt). Empty -> today's exact call.
    # Structural-only here (str, default ""); the relay weaves cleanup_prompt into the pose
    # render and forwards negative_prompt to the id_lock render seam.
    cleanup_prompt: str = ""
    negative_prompt: str = ""


# Cardinal views the remote render service accepts (front is required; the others
# are optional multi-view conditioning inputs). Mirrors identity_pipeline/mesh.py.
MESH_VIEW_NAMES: Tuple[str, ...] = ("front", "right", "back", "left")

# Pose-normalization choices (IDENTITY-VERSIONS-SLICE.md slice 3). "none" meshes the
# input pose as-is (today's behavior); "t-pose" asks the relay to render an id_lock
# T-pose still FIRST and mesh THAT (clears crossed-arm occlusion). Mirrors the store's
# identity_profiles._POSE_CHOICES so the wire vocabulary is one word everywhere.
POSE_CHOICES: Tuple[str, ...] = ("none", "t-pose")


def make_identity_mesh(
    *,
    slug: str,
    recon_id: str,
    view_sources,
    seed: int = 12345,
    num_inference_steps: int = 30,
    octree_resolution: int = 380,
    texture: bool = False,
    chain_turntable: bool = True,
    frame_count: int = 72,
    fps: int = 24,
    width: int = 768,
    height: int = 768,
    elevation_deg: float = 8.0,
    transparent: bool = False,
    auto_promote: bool = False,
    view_ids: Tuple[str, ...] = (),
    backend: str = "comfyui",
    workflow: str = "hunyuan3d-2mv",
    output_format: str = "glb",
    view_candidates=(),
    pose: str = "none",
    vision_model: Optional[str] = None,
    cleanup_prompt: str = "",
    negative_prompt: str = "",
) -> IdentityMeshSpec:
    """Validate every field and build the frozen ``IdentityMeshSpec`` (house discipline:
    a structurally-invalid spec is caller error caught at the boundary, never carried
    across the bus). Runtime policy failures (the render service being unconfigured, a
    ref image gone missing) are NOT validated here — they surface as errors-as-data
    from the relay runner."""
    if not (isinstance(slug, str) and slug.strip()):
        raise ValueError(f"slug must be a non-empty string; got {slug!r}")
    if not (isinstance(recon_id, str) and recon_id.strip()):
        raise ValueError(f"recon_id must be a non-empty string; got {recon_id!r}")

    # view_sources: coerce list/tuple of (name, path) pairs -> tuple of tuples. Each
    # name must be a cardinal view; each path a non-empty string. ``front`` is REQUIRED
    # (Hunyuan3D-2mv keys on it). Later duplicates of a view win (last assignment).
    if not isinstance(view_sources, (list, tuple)):
        raise ValueError(
            f"view_sources must be a list/tuple of (view, path) pairs; got {view_sources!r}")
    norm_pairs: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for i, pair in enumerate(view_sources):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"view_sources[{i}] must be a (view, path) pair; got {pair!r}")
        name, path = pair[0], pair[1]
        if not (isinstance(name, str) and name in MESH_VIEW_NAMES):
            raise ValueError(
                f"view_sources[{i}] view must be one of {list(MESH_VIEW_NAMES)}; got {name!r}")
        if not (isinstance(path, str) and path.strip()):
            raise ValueError(f"view_sources[{i}] path must be a non-empty string; got {path!r}")
        seen[name] = path
    # Preserve cardinal order (front first) for a stable, self-describing payload.
    for name in MESH_VIEW_NAMES:
        if name in seen:
            norm_pairs.append((name, seen[name]))
    if not norm_pairs:
        raise ValueError("at least one view_sources pair is required")
    if "front" not in seen:
        raise ValueError("view_sources must include a 'front' view (Hunyuan3D-2mv requires it)")

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int; got {seed!r}")
    for pname, pval in (("num_inference_steps", num_inference_steps),
                        ("octree_resolution", octree_resolution),
                        ("frame_count", frame_count), ("fps", fps),
                        ("width", width), ("height", height)):
        if not isinstance(pval, int) or isinstance(pval, bool) or pval <= 0:
            raise ValueError(f"{pname} must be a positive int; got {pval!r}")
    if not isinstance(elevation_deg, (int, float)) or isinstance(elevation_deg, bool):
        raise ValueError(f"elevation_deg must be a number; got {elevation_deg!r}")
    for bname, bval in (("texture", texture), ("chain_turntable", chain_turntable),
                        ("transparent", transparent), ("auto_promote", auto_promote)):
        if not isinstance(bval, bool):
            raise ValueError(f"{bname} must be a bool; got {bval!r}")

    # view_candidates: coerce list/tuple -> tuple of non-empty strings (mirrors
    # view_sources' list-of-lists -> tuple coercion below/at the deserializer). Any
    # non-string / empty entry is a caller error caught at the boundary, never carried.
    if not isinstance(view_candidates, (list, tuple)):
        raise ValueError(
            f"view_candidates must be a list/tuple of paths; got {view_candidates!r}")
    norm_candidates: list[str] = []
    for i, cpath in enumerate(view_candidates):
        if not (isinstance(cpath, str) and cpath.strip()):
            raise ValueError(
                f"view_candidates[{i}] must be a non-empty string; got {cpath!r}")
        norm_candidates.append(cpath)

    # pose: validated at the boundary like every other field (a structurally-invalid
    # spec is caller error caught here, never carried across the bus).
    if not (isinstance(pose, str) and pose in POSE_CHOICES):
        raise ValueError(f"pose must be one of {list(POSE_CHOICES)}; got {pose!r}")

    # vision_model: STRUCTURAL only (None or a non-empty string). ""/whitespace is
    # normalized to None (== the fleet-default VL model), so a blank setting is exactly
    # today's behavior. The image-text-to-text REGISTRY-membership check is the store's
    # (set_gen_settings) + route's job — NOT here, so the bus rehydrate path stays cheap.
    if vision_model is not None:
        if not isinstance(vision_model, str):
            raise ValueError(f"vision_model must be None or a string; got {vision_model!r}")
        vision_model = vision_model.strip() or None

    # cleanup_prompt / negative_prompt: additive render-steer channels (CLEANUP-PROMPT
    # slice). Both are plain strings validated exactly like the reconstruction spec's
    # base_prompt — must be a str (None coerces to ""), default "" -> byte-identical to
    # today's render (defaults-are-promises). The SEMANTICS (positive-avoid vs true
    # negative) live in the relay/reconstruction wiring; here they are structural only.
    if cleanup_prompt is not None and not isinstance(cleanup_prompt, str):
        raise ValueError(f"cleanup_prompt must be a string or None; got {cleanup_prompt!r}")
    cleanup_prompt = cleanup_prompt or ""
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValueError(f"negative_prompt must be a string or None; got {negative_prompt!r}")
    negative_prompt = negative_prompt or ""

    return IdentityMeshSpec(
        slug=slug,
        recon_id=recon_id,
        view_ids=tuple(view_ids) if isinstance(view_ids, (list, tuple)) else (),
        backend=backend,
        workflow=workflow,
        output_format=output_format,
        view_sources=tuple(norm_pairs),
        seed=seed,
        num_inference_steps=num_inference_steps,
        octree_resolution=octree_resolution,
        texture=bool(texture),
        chain_turntable=bool(chain_turntable),
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        elevation_deg=float(elevation_deg),
        transparent=bool(transparent),
        auto_promote=bool(auto_promote),
        view_candidates=tuple(norm_candidates),
        pose=pose,
        vision_model=vision_model,
        cleanup_prompt=cleanup_prompt,
        negative_prompt=negative_prompt,
    )


def identity_mesh_from_dict(d: dict) -> IdentityMeshSpec:
    """Rebuild an ``IdentityMeshSpec`` from its ``asdict`` form THROUGH the validating
    factory (deserialize-then-revalidate, like every other bus spec). Registered in
    ``media_bus.SPEC_DESERIALIZERS`` under ``"identity_mesh_build"``. ``asdict`` turns the
    ``view_sources`` tuple-of-tuples into list-of-lists over JSON, so coerce it back."""
    raw = d.get("view_sources") or ()
    pairs = [tuple(p) for p in raw if isinstance(p, (list, tuple)) and len(p) == 2]
    return make_identity_mesh(
        slug=d["slug"],
        recon_id=d["recon_id"],
        view_sources=pairs,
        seed=d.get("seed", 12345),
        num_inference_steps=d.get("num_inference_steps", 30),
        octree_resolution=d.get("octree_resolution", 380),
        texture=bool(d.get("texture", False)),
        chain_turntable=bool(d.get("chain_turntable", True)),
        frame_count=d.get("frame_count", 72),
        fps=d.get("fps", 24),
        width=d.get("width", 768),
        height=d.get("height", 768),
        elevation_deg=d.get("elevation_deg", 8.0),
        transparent=bool(d.get("transparent", False)),
        auto_promote=bool(d.get("auto_promote", False)),
        view_ids=tuple(d.get("view_ids") or ()),
        backend=d.get("backend", "comfyui"),
        workflow=d.get("workflow", "hunyuan3d-2mv"),
        output_format=d.get("output_format", "glb"),
        # asdict->json keeps view_candidates a plain list; coerce back like view_ids.
        view_candidates=tuple(d.get("view_candidates") or ()),
        # additive (VERSIONS slice): round-trip pose; an OLD spec (pre-pose) has no key
        # -> defaults to "none" (today's behavior), so deserialization stays backward-compat.
        pose=d.get("pose", "none"),
        # additive (per-identity VISION MODEL): round-trip vision_model; an OLD spec has no
        # key -> None (== the fleet-default VL model), so deserialization stays backward-compat.
        vision_model=d.get("vision_model"),
        # additive (CLEANUP-PROMPT slice): round-trip the render-steer channels; an OLD spec
        # has neither key -> "" (== today's byte-identical render), so deserialization stays
        # backward-compat.
        cleanup_prompt=d.get("cleanup_prompt", ""),
        negative_prompt=d.get("negative_prompt", ""),
    )


def make_identity_reconstruction(**kwargs) -> IdentityReconstructionSpec:
    return IdentityReconstructionSpec(**kwargs)

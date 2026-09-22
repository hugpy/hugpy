"""RENDER PRESETS — the exhaustive, declarative set of recipes that ACTUALLY work
on this fleet, as data (§2, data-over-code).

WHY THIS MODULE EXISTS. The zoo (``models_seed``) is a catalogue of INTENT: 23
``ModelConfig`` rows, 16 declared ``Capability`` values, 27 ``RunnerSpec`` entries.
The router joins them structurally and the k1 gate prunes runners whose MODULE is
absent — but neither answers the only question a caller actually has: *what can I
ask for today and get a clip back?* Measured on the live fleet 2026-07-27
(CAPABILITY-VIABILITY-MAP.md, MODEL-POOL-INVENTORY.md, ROUTE-MODEL-SEQUENCE.md),
the honest answer is much smaller than the catalogue:

  * exactly THREE video models have BOTH weights on disk AND a non-stub runner —
    wan2.1-t2v-1.3b, wan2.1-vace-1.3b, wan2.1-i2v-14b-720p;
  * TWELVE registry rows hold ZERO bytes on the shared store (``ZERO_BYTE_MODELS``);
  * TWO runner modules exist but return ``Err`` on EVERY path (``ltx_upscale``,
    ``rife_interpolate``) — and by merely existing they satisfy the k1 gate's
    ``find_spec`` check, which is exactly how UPRES/INTERP came to bind a stub on
    the only render box (CAPABILITY-VIABILITY-MAP.md, "Cross-cutting", root cause);
  * TWO no-weights ffmpeg transforms are real, GPU-less, and work today.

A preset is therefore a PROVEN TUPLE, not a wish: (capability, model, precision,
geometry, frame budget) that either has produced pixels on this fleet or is a real
transform of real pixels. Everything a preset does not cover is something we must
REFUSE by name rather than route into a failure three layers down — see
``capability_verdict``. That is defaults-are-promises applied to the catalogue.

⚠ SECOND PASS, SAME DAY (2026-07-27) — THE TABLE WAS ITSELF PUBLISHING A PHANTOM.
The first cut of this file shipped a row called ``clip-control-480p`` advertising
FOUR capabilities (motion / inpaint / outpaint / retake) and an input named
"control". An adversarial review measured it on the live route and found it was the
exact defect this module exists to delete:

  * ``video_routes.py`` rejected ``control_image``/``control_kind`` for every
    capability except ``id_lock``, so all four 400'd the moment a control was
    supplied;
  * WITHOUT a control all four returned 200 and rendered a PLAIN FULL RESTYLE —
    byte-for-byte the ``clip-v2v-480p`` path wearing a different capability string;
  * and no caller-supplied MASK, EXPANDED CANVAS or FRAME RANGE exists anywhere in
    the spine. Verified across video_routes.py, studio/job.py, studio/schemas.py,
    studio/produce.py and runners/wan_vace.py: wan_vace has exactly THREE control
    branches (``vace_context_frames`` → the extend idiom, ``source_video`` → decoded
    per-frame control, ``control_image`` → one still repeated) and not one of them
    reads a mask, a canvas or a range.

KEEPER RULING: do not publish a capability that renders something else. The row was
split on the evidence, one capability at a time, and the two halves went opposite
ways:

  * MOTION SURVIVED, as its own row (``clip-motion-480p``). ``control_image`` IS a
    real, structurally distinct VACE branch — wan_vace.py:534-547 loads the still,
    resizes it and repeats it across ``num_frames`` as the pipeline's ``video=``
    control channel, which is a different conditioning tensor from v2v's decoded
    source frames and from id_lock's reference latents. The ONLY thing stopping it
    was the route's id_lock-only gate, so the gate was widened (a small, honest fix)
    and the preset kept — with its limits stated on the row, not buried.
  * INPAINT / OUTPAINT / RETAKE BECAME REFUSALS. There is no input channel for any
    of them and no arrangement of the three existing branches produces one: a render
    with no mask makes diffusers fill ``mask = ones_like(video)`` (verified in the
    INSTALLED diffusers 0.39.0: pipeline_wan_vace.py:457, then :558-560 split
    ``inactive = video*(1-mask)`` / ``reactive = video*mask``) so EVERY frame is
    reactive; ``_read_control_frames`` RESIZES the source into the target geometry
    rather than padding a larger canvas around it, so there is nothing outside the
    frame to outpaint into; and nothing carries a frame range to retake. Serving
    them would mean handing back a full restyle and calling it an edit. See
    ``_UNSERVABLE_WHY``.

NOMENCLATURE. These are PRESETS, never "templates": GLOSSARY.md:166 already binds
"template" to fleet config profiles, and a second meaning in the same tree would be
a naming collision the keeper owns and refuses.

NO I/O, NO HTTP, NO FLASK. Everything here is frozen data + pure functions, so the
route layer, the composer, the console and the tests all read the SAME table.
``tests/studio/test_presets.py`` proves each row rather than asserting the table
against itself: weights on disk, runner registered and non-stub, geometry the model
really declares, Wan's 4k+1 cadence, ``proven`` checked against the CLIP STORE, and
the inverse — that every uncovered capability refuses.
"""

from __future__ import annotations

from dataclasses import dataclass

from hugpy_video.intel.studio.enums import Capability, Framework, Precision, Task
from hugpy_video.intel.studio.schemas import (
    DEFAULT_FRAMES_REAL,
    WAN_FRAME_CADENCE,
    WAN_MAX_FRAMES,
    snap_wan_frames,
)

# --------------------------------------------------------------------------- #
# Fleet facts these presets are sized against (measured 2026-07-27; do not guess)
# --------------------------------------------------------------------------- #
# ae's RTX 3090 is the ONLY studio render target. computron's 4060 (8 GiB) cannot
# hold a Wan pipeline; central has no GPU at all (it still serves the two ffmpeg
# presets, which is the whole point of keeping them).
RENDER_BOX = "ae / RTX 3090"
RENDER_BOX_VRAM_GIB = 23.56

# CLIP LENGTH — IMPORTED, NEVER RESTATED (fixed 2026-07-27).
#
# ``WAN_FRAME_CADENCE`` / ``WAN_MAX_FRAMES`` / ``snap_wan_frames`` and the default
# below all come from ``studio/schemas.py``, which is where the renderer's own
# ``resolve_frames`` reads them. They used to be typed in here as literals, and the
# literals DRIFTED: this file published ``default_frames=29`` on every Wan row while
# ``runners/synthetic.resolve_frames`` defaulted a real model to 81. GET
# /video/render/presets therefore promised a ~1.8 s clip and the renderer produced a
# 5.06 s one — 2.8x the latent tokens and 2.8x the GPU minutes the caller was quoted.
# That is a defaults-are-promises violation on the most expensive axis the studio
# has, and it is only possible when the same fact is spelled in two places. One
# literal, two importers, no second place to disagree; ``tests/studio/
# test_clip_length.py`` and ``test_presets.py`` both fail if this ever re-diverges.
#
# ⚠ 29 WAS NEVER "THE LENGTH THIS FLEET RENDERS AT", which is what the old comment
# here claimed. ffprobe over all 47 landed Wan clips (2026-07-27): 42 are 29 frames,
# 3 are 45, 1 is 81, 1 is 21. The 29-frame majority is an ARTEFACT of the retired
# ``fps * 2`` default at the fps those clips asked for (16 -> 32 -> snapped 29);
# the fps-24 clips landed 45 and the single fps-48 clip landed 81 (the ceiling
# clamp). Nobody ever chose 29.
WAN_DEFAULT_FRAMES = DEFAULT_FRAMES_REAL


def is_wan_cadence(n_frames: int) -> bool:
    """True iff ``n_frames`` is a legal Wan frame count (4k+1, k >= 0).

    The predicate half of ``snap_wan_frames`` (imported from ``schemas``, which is
    where the renderer reads it too). Kept here because it is a QUESTION the wire
    layer asks — "is the number I am about to publish one the pipeline accepts?" —
    while the snap is an ACTION the renderer takes."""
    return n_frames >= 1 and (n_frames - 1) % WAN_FRAME_CADENCE == 0


# --------------------------------------------------------------------------- #
# The two disqualifying classes, recorded so a preset can never quietly acquire one
# --------------------------------------------------------------------------- #
# ZERO BYTES ON THE SHARED STORE (verified 2026-07-27: the store holds exactly the
# 6 Wan-AI dirs + Lightricks/ltxv-spatial-upscaler-0.9.7 — nothing else). These rows
# are documented INTENT and must stay in the registry (validate_registry raises if a
# declared Capability is served by NO model, and codeformer/framepack are the sole
# providers of RESTORE/STREAM) — but no preset may ever name one.
ZERO_BYTE_MODELS: frozenset[str] = frozenset({
    "hunyuanvideo", "hunyuanvideo-avatar", "open-sora-v2", "skyreels-v1",
    "ltx-2.3", "ltx-video-0.9.7-dev", "mochi-1-preview", "cogvideox-5b",
    "animatediff-lightning", "framepack-i2v-hy", "rife-practical", "codeformer",
})

# UNCONDITIONAL-Err STUBS: the module EXISTS (so registry.runner_available's
# find_spec gate passes it) but every return in the entrypoint is an Err. This is
# strictly worse than no module — it defeats the very gate built to stop unroutable
# rows from binding. ltx_upscale even returns WEIGHTS_MISSING while its 3.1 GB of
# weights ARE on disk. Named here for documentation; test_presets.py does NOT trust
# this list — it re-derives stub-ness structurally from each runner's AST, and uses
# these two only as a positive control that the detector still detects.
STUB_RUNNER_MODULES: frozenset[str] = frozenset({
    "hugpy_video.intel.studio.runners.ltx_upscale",
    "hugpy_video.intel.studio.runners.rife_interpolate",
})


# --------------------------------------------------------------------------- #
# RenderPreset
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RenderPreset:
    """One render recipe that works on this fleet.

    Frozen + slotted like every other studio schema. ``evidence`` is load-bearing,
    not decoration: it records WHY we believe this tuple produces pixels (a clip
    count on disk, a measured VRAM figure, "no weights and no GPU"), so a future
    reader can tell a proven row from an argued one without re-running the fleet.

    GEOMETRY may be ``None`` on the enhance presets and ONLY there: an upres/interp
    transform has no geometry of its own — it takes whatever the source clip and the
    manifest target say. ``None`` is the honest encoding of that; a 0 would be a lie
    with a shape.
    """

    preset_id: str
    title: str
    description: str                      # one line, user-facing
    capabilities: tuple[Capability, ...]  # every capability this preset serves
    model_id: str                         # a MODEL_REGISTRY key, always
    framework: Framework
    task: Task
    precision: Precision
    width: int | None                     # None = inherited from the source clip
    height: int | None
    fps: int | None
    default_frames: int | None
    max_frames: int | None
    inputs: tuple[str, ...]               # what the caller must supply
    # ⚠ PROVEN IS A CHECKED CLAIM, NOT A MOOD (tightened 2026-07-27). It means "this
    # exact path has produced pixels ON THIS FLEET", and a reviewer's finding that it
    # was the one field NO test validated is why it now has two independent proofs in
    # ``tests/studio/test_presets.py``:
    #   1. STORE PROOF — a proven=True row must have >= 1 completed clip in the studio
    #      clip store whose manifest names its ``model_id``. Read off the store, not
    #      off this table.
    #   2. COMPOSITION PROOF — a preset can never be more proven than the weakest
    #      thing it ``composes``. movie-480p asserted proven=True while composing
    #      clip-i2v-480p (proven=False) whose 14B binding has produced 0 of the 47
    #      landed Wan clips, i.e. the ``still`` joint has never once completed here.
    # False is not a defect and not a warning to hide — it is the row telling the
    # truth about itself, and the wire publishes it so a caller can prefer a proven
    # path.
    proven: bool
    evidence: str
    # MOVIE ONLY: the preset_ids whose bindings a multi-segment render composes, and
    # the joint modes it supports. Empty on every single-clip preset.
    composes: tuple[str, ...] = ()
    joints: tuple[str, ...] = ()

    @property
    def capability(self) -> Capability:
        """The PRIMARY capability — what a caller asks for to reach this preset.

        Every ratified row serves exactly ONE capability today; the tuple shape is
        kept because a genuinely multi-capability binding is possible in principle
        (and because collapsing it would hide the lesson of clip-control-480p, which
        claimed four and served one)."""
        return self.capabilities[0]

    @property
    def geometry(self) -> str:
        """Human-readable geometry, or "source" for the enhance presets."""
        if self.width is None or self.height is None:
            return "source"
        return f"{self.width}x{self.height}"


# --------------------------------------------------------------------------- #
# THE PRESETS. Eight rows, ratified by the operator 2026-07-27 and RE-RATIFIED the
# same day after the phantom-control split (motion kept as its own row; inpaint /
# outpaint / retake demoted to refusals). This tuple is the deliverable: adding a
# ninth means proving a ninth.
#
# EVERY NUMBER BELOW WAS RE-MEASURED IN THAT SECOND PASS. The first cut carried
# figures copied out of CAPABILITY-VIABILITY-MAP.md that the store no longer agrees
# with (tree sizes short by 0.4-3.7 GB, a clip count short by one, and a whole-GPU
# fit that the corrected placement arithmetic turns into a miss). Tree sizes are
# file-sums under /mnt/llm_storage/video_intel/studio/weights; clip counts are
# manifests with a non-empty clip.mp4 under .../studio/clips; VRAM needs come from
# ``runners.wan_i2v._placement_need_gib`` as it stands today, not from prose.
# --------------------------------------------------------------------------- #
_PRESETS: tuple[RenderPreset, ...] = (
    # 1 ---------------------------------------------------------------------
    # The calibration render. This is the tuple that landed the most clips on ae, and
    # it is the cheapest real render the fleet has: a 1.419e9-param DiT, and the
    # whole-GPU need at 832x480 derives to 17.90 GiB (29f) / 19.20 GiB (81f) of ae's
    # 23.56 — a comfortable FIT at both ends of the frame budget.
    RenderPreset(
        preset_id="clip-t2v-480p",
        title="Text to clip (480p)",
        description="Generate a 480p clip from a prompt alone — the fleet's proven baseline render.",
        capabilities=(Capability.T2V,),
        model_id="wan2.1-t2v-1.3b",
        framework=Framework.WAN, task=Task.T2V, precision=Precision.FP16,
        width=832, height=480, fps=16,
        default_frames=WAN_DEFAULT_FRAMES, max_frames=WAN_MAX_FRAMES,
        inputs=("prompt",),
        proven=True,
        evidence="PROVEN: 14 t2v-1.3b clips in the studio clip store (of 47 real Wan "
                 "clips total; the other 33 are vace-1.3b). Diffusers tree 28.94 GB / "
                 "26.95 GiB complete on the shared store (file-sum over 65 files). "
                 "Whole-GPU need derives to 17.90 GiB at 832x480x29f and 19.20 GiB at "
                 "the 81-frame default — both fit ae's 23.56 GiB. The longest clip this "
                 "fleet has ever produced came from this row: 81 frames at fps 48, "
                 "2026-07-18 12:32.",
    ),
    # 2 ---------------------------------------------------------------------
    # The identity flagship. VACE reference-to-video is the ONLY path on this fleet
    # that genuinely CONSUMES reference_images (wan_vace.py:549-559 -> VACE reference
    # latents). Everything else that claims id_lock either ignores the references or
    # now refuses the manifest outright — which is why id_lock is pinned to this row
    # and to 480p, its declared ceiling. Above 832x480 no VACE model fits and the
    # router used to fall through to the 14B i2v, whose runner never read the refs:
    # a plausible clip of the WRONG PERSON, with no error at all (models_seed.py:172
    # corrective note, 2026-07-27).
    RenderPreset(
        preset_id="clip-idlock-480p",
        title="Identity-locked clip (480p)",
        description="Hold one character's identity across a 480p clip from up to 4 reference images.",
        capabilities=(Capability.ID_LOCK,),
        model_id="wan2.1-vace-1.3b",
        framework=Framework.WAN, task=Task.VACE_CONTROL, precision=Precision.FP16,
        width=832, height=480, fps=16,
        default_frames=WAN_DEFAULT_FRAMES, max_frames=WAN_MAX_FRAMES,
        inputs=("prompt", "reference_images"),
        proven=True,
        evidence="PROVEN, and the largest proven population on this fleet: 33 vace-1.3b "
                 "clips in the studio clip store, 32 of them capability id_lock. The "
                 "ONLY runner here that consumes reference_images. Tree 19.04 GB / "
                 "17.74 GiB complete (file-sum over 57 files); whole-GPU need 19.27 GiB "
                 "at 832x480x29f and 20.57 GiB at the 81-frame default, of 23.56 — the "
                 "tightest fit the fleet routes today, with 2.99 GiB to spare.",
    ),
    # 3 ---------------------------------------------------------------------
    # SAME runner and SAME weights as #2 — a different control wiring, not a different
    # model. With a source_video and NO mask, diffusers 0.39 fills mask as
    # torch.ones_like(video) (installed diffusers 0.39.0, pipeline_wan_vace.py:457;
    # the inactive/reactive split is :558-560) => inactive=0, reactive=video:
    # a full restyle driven by the source structure. That IS v2v semantics — and it is
    # ALSO precisely why inpaint/outpaint/retake cannot be served on the same wiring:
    # "every frame reactive" is the opposite of a confined edit.
    RenderPreset(
        preset_id="clip-v2v-480p",
        title="Restyle a clip (480p)",
        description="Restyle or re-light an existing 480p clip while keeping its motion and structure.",
        capabilities=(Capability.V2V,),
        model_id="wan2.1-vace-1.3b",
        framework=Framework.WAN, task=Task.VACE_CONTROL, precision=Precision.FP16,
        width=832, height=480, fps=16,
        default_frames=WAN_DEFAULT_FRAMES, max_frames=WAN_MAX_FRAMES,
        inputs=("prompt", "source_video"),
        proven=True,
        evidence="PROVEN: same runner + same weights as clip-idlock-480p (33 clips on "
                 "disk), reached with source_video instead of reference_images — and 1 "
                 "of those 33 landed clips took exactly this path (capability v2v). "
                 "wan_vace decodes the source to a per-frame `video=` control channel "
                 "with no mask, so the pipeline treats every frame as reactive: a full "
                 "restyle driven by the source's structure.",
    ),
    # 4 ---------------------------------------------------------------------
    # WHAT IS LEFT OF clip-control-480p, and it is one capability, not four.
    #
    # THE BRANCH IS REAL. wan_vace.py:534-547: control_image is loaded as PIL, resized
    # to the render geometry, and repeated across num_frames as the pipeline's
    # `video=` control channel. That is a genuinely different conditioning tensor from
    # v2v (which decodes the SOURCE CLIP into that same channel) and from id_lock
    # (which fills `reference_images=` instead and leaves `video=` empty). Three
    # branches, three different renders, same proven weights.
    #
    # ⚠ WHAT IT IS NOT, STATED ON THE ROW RATHER THAN DISCOVERED AT RENDER TIME: the
    # control is ONE STILL REPEATED, so this is a STATIC structural anchor — pose /
    # depth / sketch composition blocking. Per-frame motion TRANSFER (driving a shot
    # from a pose SEQUENCE) is not wired: nothing in the spine carries a control VIDEO
    # as distinct from a source video, and handing the source clip in instead makes
    # wan_vace take the v2v branch (source_video wins the elif at :524), silently
    # dropping the control. The route refuses that combination by name rather than
    # rendering the wrong thing — see video_routes.py's motion gate.
    RenderPreset(
        preset_id="clip-motion-480p",
        title="Pose / structure control (480p)",
        description="Anchor a 480p clip to a pose, depth or sketch still — VACE structural control, no source clip.",
        capabilities=(Capability.MOTION,),
        model_id="wan2.1-vace-1.3b",
        framework=Framework.WAN, task=Task.VACE_CONTROL, precision=Precision.FP16,
        width=832, height=480, fps=16,
        default_frames=WAN_DEFAULT_FRAMES, max_frames=WAN_MAX_FRAMES,
        inputs=("prompt", "control_image", "control_kind"),
        proven=False,
        evidence="NOT YET PROVEN, but a genuinely DISTINCT branch rather than a rename: "
                 "wan_vace.py:534-547 repeats the control still across num_frames as the "
                 "VACE `video=` channel — a different conditioning tensor from v2v's "
                 "decoded source frames and from id_lock's reference latents, on the "
                 "same weights that produced 33 clips. 0 of the 47 landed Wan clips "
                 "reached that branch, because the route gated control_image to id_lock "
                 "until 2026-07-27. The control is a STATIC still repeated, so this is "
                 "composition/pose anchoring, not per-frame motion transfer.",
    ),
    # 5 ---------------------------------------------------------------------
    # THE ROW THE ARITHMETIC MOVED, and it is labelled as such. The studio clip store
    # holds 47 real Wan clips and NOT ONE is from any 14B row — this path is 0 of 47.
    #
    # PRECISION CHOICE — FP8, and it is the cheapest one there is.
    # The docs talk about "4-bit nf4"; the code's ``Precision`` enum has NO NF4 member
    # (enums.py:74-79 = fp32/bf16/fp16/fp8/int8). ``Precision.FP8`` IS the fleet's
    # 4-bit lever: wan_i2v._bnb_config maps FP8 -> BitsAndBytesConfig(load_in_4bit=True,
    # bnb_4bit_quant_type="nf4"). So "fp8" is the NAME and nf4 is the BYTES, and the
    # registry row's fp8=18.0 GB envelope is the entry the router picks.
    #
    # ⚠ CORRECTED 2026-07-27 (second pass). This row previously claimed fp8 "fits
    # 23.56, by 0.08 GiB". It does not, and the 0.08 was never a margin — it was the
    # residue of an activation-workspace line whose INTERCEPT missed both of its own
    # calibration points by 2.64 GiB, making every estimate in the module that much
    # light. With the intercept derived from the measurements instead of typed in,
    # ``wan_i2v._placement_need_gib`` at 832x480x29f now returns:
    #     fp8 -> nf4   26.12 GiB   does NOT fit 23.56 (miss by 2.56)
    #     int8         32.84 GiB   does not fit
    #     bf16         48.07 GiB   does not fit (the DiT alone is 30.5 GiB)
    # and at the 81-frame default fp8 needs 29.20 GiB. So NO precision places this row
    # whole-on-GPU on ae at any supported geometry. That is not a refusal: the
    # placement decision correctly declines whole-GPU and the render falls back to
    # sequential CPU offload, whose peak resident is max(module) = the 10.582 GiB bf16
    # UMT5-XXL text encoder. It renders; it renders slowly. Keeping the row with
    # proven=False and this arithmetic on it is the honest encoding — deleting it
    # would hide a path that works, and marking it proven would invent 47 clips.
    RenderPreset(
        preset_id="clip-i2v-480p",
        title="Image to clip (480p)",
        description="Animate a still image into a 480p clip.",
        capabilities=(Capability.I2V,),
        model_id="wan2.1-i2v-14b-720p",
        framework=Framework.WAN, task=Task.I2V, precision=Precision.FP8,
        width=832, height=480, fps=16,
        default_frames=WAN_DEFAULT_FRAMES, max_frames=WAN_MAX_FRAMES,
        inputs=("prompt", "start_image"),
        proven=False,
        evidence="NOT PROVEN: 0 of the 47 real Wan clips on this fleet came from any 14B "
                 "row. Weights complete (90.10 GB / 83.92 GiB, file-sum over 95 files). "
                 "⚠ CORRECTED 2026-07-27: this row does NOT place whole-on-GPU at ANY "
                 "precision — derived need at 832x480x29f is 26.12 GiB (fp8->nf4), "
                 "32.84 (int8), 48.07 (bf16) against ae's 23.56, and 29.20 GiB at the "
                 "81-frame default. It still renders, via sequential CPU offload whose "
                 "peak is the 10.582 GiB UMT5-XXL encoder — slow, not impossible.",
    ),
    # 6 ---------------------------------------------------------------------
    # The multi-segment composer. It is a PRESET even though it binds no model of its
    # own: Capability.ASSEMBLE is explicitly an orchestration stage (registry
    # PLANNED_CAPABILITIES), and what makes it a preset is that every segment it emits
    # is itself a preset.
    #
    # ``composes`` names the three ratified segment bindings. The joint modes map onto
    # them like this, and the mapping is worth stating because it is NOT one-to-one:
    #   cut          -> every segment capability t2v            -> clip-t2v-480p
    #   vace_extend  -> segments 1..N forced to capability v2v  -> the SAME binding as
    #                   clip-v2v-480p (studio_movie.py:696)
    #   still        -> segments 1..N capability i2v            -> clip-i2v-480p
    # With movie-level reference_images or identity_profile, EVERY segment (seg 0
    # included) becomes id_lock -> clip-idlock-480p (studio_movie.py:627/645).
    #
    # ⚠ proven=False, CORRECTED 2026-07-27. This row shipped proven=True on evidence
    # that contradicted the table it sits in: it composes clip-i2v-480p, which is
    # proven=False, and the clip store holds ZERO clips from any 14B row — so the
    # ``still`` joint has never once completed on this fleet. A composite cannot be
    # more proven than the weakest thing it composes, and test_presets.py now enforces
    # exactly that. Narrowing ``composes`` to hide the unproven joint was the other
    # available "fix" and would have been a worse lie: studio_movie really does force
    # capability i2v on a still joint, so the 14B binding belongs in this list.
    #
    # COST NOTE, not a defect: joint=still binds the 14B per segment and re-reads
    # 90.10 GB from the spinning shared store each time (131 MB/s measured on ae =
    # ~11.5 min of pure I/O before the first denoise step, because every segment is a
    # fresh spawned child with no pipeline cache). cut and vace_extend keep every
    # segment on a 1.3B tree and come off page cache after the first load. Prefer them.
    RenderPreset(
        preset_id="movie-480p",
        title="Multi-segment movie (480p)",
        description="Chain several 480p segments into one movie, joined by cut, motion-carry or a still.",
        capabilities=(Capability.ASSEMBLE,),
        model_id="wan2.1-t2v-1.3b",   # segment 0's binding; later segments per joint
        framework=Framework.WAN, task=Task.T2V, precision=Precision.FP16,
        width=832, height=480, fps=16,
        default_frames=WAN_DEFAULT_FRAMES, max_frames=WAN_MAX_FRAMES,
        inputs=("goals",),
        proven=False,
        evidence="NOT PROVEN END TO END, and the weak joint is named: joint=cut and "
                 "joint=vace_extend keep every segment on a proven 1.3B binding (14 t2v "
                 "clips / 33 vace clips on disk), but joint=still binds clip-i2v-480p "
                 "and 0 of the 47 landed Wan clips came from any 14B row — that joint "
                 "has never completed here. Assembly itself is ffmpeg concat over "
                 "uniformly re-encoded contributions. COST: a still joint re-reads the "
                 "90.10 GB i2v tree per segment at a measured 131 MB/s = ~11.5 min of "
                 "pure disk before the first denoise step. Prefer cut / vace_extend.",
        composes=("clip-t2v-480p", "clip-idlock-480p", "clip-i2v-480p"),
        joints=("cut", "vace_extend", "still"),
    ),
    # 7 ---------------------------------------------------------------------
    # No weights, no GPU, no download — a real transform of real pixels via the system
    # ffmpeg binary, and it runs on GPU-less central. It is the ONLY working UPRES on
    # this fleet: the premium row (ltxv-spatial-upscaler) has its 3.1 GB of weights on
    # disk but an unconditional-Err runner, and because that stub MODULE exists the k1
    # find_spec gate passes it and router._score's real_first ranks it ABOVE this row.
    # The router's viability gate (2026-07-27) is what stops that today.
    RenderPreset(
        preset_id="enhance-upres",
        title="Upscale a clip",
        description="Spatially upscale an existing clip to the target resolution — no GPU, no weights.",
        capabilities=(Capability.UPRES,),
        model_id="ffmpeg-lanczos-upscale",
        framework=Framework.FFMPEG, task=Task.UPSCALE, precision=Precision.INT8,
        width=None, height=None, fps=None,
        default_frames=None, max_frames=None,
        inputs=("source_video",),
        proven=True,
        evidence="PROVEN: 1 ffmpeg-lanczos-upscale clip in the studio clip store. An "
                 "ffmpeg transform of real pixels, no weights, no GPU — verified "
                 "end-to-end 2026-07-27 on GPU-less central: 416x240 -> 832x480 lanczos, "
                 "8 frames, clip.mp4 + manifest.json + provenance.json into the "
                 "content-addressed layout. DeterminismClass.EXACT (-threads 1, "
                 "bit-stable).",
    ),
    # 8 ---------------------------------------------------------------------
    # Same story as #7 on the INTERP side: real motion-compensated interpolation
    # (minterpolate mi_mode=mci) on the system binary, outranked until today by the
    # rife_interpolate stub whose Practical-RIFE arch is not vendored and whose
    # weights are 0 bytes on the store.
    RenderPreset(
        preset_id="enhance-interp",
        title="Interpolate frames",
        description="Raise an existing clip's frame rate with motion-compensated interpolation — no GPU, no weights.",
        capabilities=(Capability.INTERP,),
        model_id="ffmpeg-minterpolate",
        framework=Framework.FFMPEG, task=Task.INTERPOLATE, precision=Precision.INT8,
        width=None, height=None, fps=None,
        default_frames=None, max_frames=None,
        inputs=("source_video",),
        proven=True,
        evidence="PROVEN: 1 ffmpeg-minterpolate clip in the studio clip store. An ffmpeg "
                 "transform of real pixels, no weights, no GPU — verified end-to-end "
                 "2026-07-27 on GPU-less central: 8 frames @8fps -> 13 frames @16fps, "
                 "real motion-compensated output written to the content-addressed "
                 "layout. DeterminismClass.EXACT.",
    ),
)

_BY_ID: dict[str, RenderPreset] = {p.preset_id: p for p in _PRESETS}


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #
def all_presets() -> tuple[RenderPreset, ...]:
    """Every preset, in ratified order (clips, then movie, then enhance)."""
    return _PRESETS


def preset(preset_id: str) -> RenderPreset | None:
    """The preset with this id, or None. Never raises — an unknown id is caller
    data, not programmer error (the route takes it straight off the wire)."""
    return _BY_ID.get(preset_id)


def presets_for(capability: Capability) -> tuple[RenderPreset, ...]:
    """Every preset that serves ``capability``, in ratified order. Empty tuple means
    UNSERVABLE — see ``capability_verdict`` for what to tell the caller."""
    return tuple(p for p in _PRESETS if capability in p.capabilities)


def servable_capabilities() -> frozenset[Capability]:
    """The capabilities at least one preset covers — what the fleet can actually do
    today. Everything in ``Capability`` outside this set must be REFUSED."""
    return frozenset(c for p in _PRESETS for c in p.capabilities)


def unservable_capabilities() -> frozenset[Capability]:
    """The inverse: declared in the enum, covered by no preset. The point of the
    whole exercise — these are the ones we refuse by name instead of routing."""
    return frozenset(Capability) - servable_capabilities()


def presets_using(model_id: str) -> tuple[RenderPreset, ...]:
    """Every preset bound to ``model_id``. The "what breaks if this model goes away"
    query (and how the movie preset's segment models are found: also check
    ``composes``)."""
    return tuple(p for p in _PRESETS if p.model_id == model_id)


# --------------------------------------------------------------------------- #
# Honest refusal
# --------------------------------------------------------------------------- #
# WHY each uncovered capability is uncovered. These are the MEASURED blockers from
# CAPABILITY-VIABILITY-MAP.md 2026-07-27 plus the second-pass source reading, not
# guesses — a refusal that says "not supported" teaches the caller nothing, and a
# refusal that cites a download they can act on (or a contract gap no download fixes)
# teaches them everything.
#
# NOTE THE TWO KINDS, deliberately worded differently:
#   * MISSING BYTES / MISSING MODULE (stream, restore, part of lipsync) — a download
#     or a wiring errand could change the answer, so the size is named.
#   * MISSING INPUT CHANNEL (keyframe, audio, inpaint, outpaint, retake) — no
#     download changes anything, because the gap is in the CONTRACT: there is nowhere
#     for the caller to put the thing that would make the capability mean something.
#     Those refusals must say so plainly, and must NOT imply the model is absent.
_UNSERVABLE_WHY: dict[Capability, str] = {
    Capability.KEYFRAME: (
        # ⚠ THE ASYMMETRY, STATED (2026-07-27). A reviewer flagged that keyframe is
        # refused as "a rename of i2v" while the SAME weights are blessed as
        # clip-i2v-480p, and that the old wording ("has never produced a clip on this
        # fleet") read as if the model were missing. It is not. What is missing is the
        # END-FRAME INPUT, and saying that is the difference between a caller
        # re-downloading 90 GB and a caller using the route that works.
        "the MODEL is present and already blessed — wan2.1-i2v-14b-720p is the same "
        "row clip-i2v-480p binds. What is missing is the INPUT CHANNEL: nothing in "
        "the spine carries an end/last frame (StudioI2VSpec has no field for one, "
        "RenderManifest carries none, and run_wan_i2v passes no last_image), so a "
        "keyframe request today would render an ordinary i2v clip and silently ignore "
        "the frame you asked it to land on. Use i2v with your first frame as "
        "start_image; keyframe becomes a preset the day an end-frame field is threaded "
        "end to end"),
    Capability.STREAM: (
        "the only model declaring it (framepack-i2v-hy) is 0 bytes on the store and its "
        "runner module does not exist; the real download is ~43 GB against one contended "
        "23.56 GiB card"),
    Capability.AUDIO: (
        "the studio clip contract has NO audio track at all — assembly is PNG frames -> "
        "libx264 -> yuv420p with no audio input — so no download fixes this; the only "
        "model declaring it (ltx-2.3) also exceeds the card at every published precision"),
    Capability.LIPSYNC: (
        "same missing audio track as audio, plus both declaring models are 0 bytes on the "
        "store (hunyuanvideo-avatar is 80.8 GB upstream with no diffusers pipeline)"),
    Capability.RESTORE: (
        "the only model declaring it (codeformer) is 0 bytes on the store, its runner "
        "module does not exist, and its license is non-commercial so it could never "
        "auto-route anyway"),
    # ---- the three demoted from clip-control-480p, 2026-07-27 ----------------
    # All three read the same way on purpose: the WEIGHTS ARE FINE (VACE-1.3B is the
    # most proven row on this fleet), the gap is that there is no channel for the one
    # input that distinguishes the capability from a plain restyle. Each names the
    # restyle explicitly, because "what you would actually get" is the fact a caller
    # needs — it is what the route was silently handing back until today.
    Capability.INPAINT: (
        "no MASK input exists in this spine — StudioI2VSpec carries none, "
        "RenderManifest carries none, and wan_vace's three control branches "
        "(vace_context_frames / source_video / control_image) read none — so diffusers "
        "fills mask=ones_like(video) (installed diffusers 0.39.0, "
        "pipeline_wan_vace.py:457) and EVERY frame is reactive. What you would get is "
        "a full restyle of the whole clip, not an edit confined to a region; ask for "
        "that deliberately with v2v"),
    Capability.OUTPAINT: (
        "no EXPANDED-CANVAS input exists in this spine: the render geometry comes from "
        "the resolution ladder and wan_vace RESIZES the source into it "
        "(_read_control_frames, PIL .resize) rather than padding a larger canvas around "
        "it, and there is no mask to mark the new border reactive. There is literally "
        "nothing outside the frame to paint into; what you would get is a full restyle "
        "at the same framing — use v2v"),
    Capability.RETAKE: (
        "no FRAME-RANGE input exists in this spine: nothing on the spec, the manifest "
        "or the VACE call names a start/end frame to regenerate, and with no mask the "
        "pipeline marks every frame reactive. What you would get is a full restyle of "
        "the entire clip rather than a targeted re-take of a segment — use v2v, or cut "
        "the clip and re-render the piece you want"),
}


@dataclass(frozen=True, slots=True)
class CapabilityVerdict:
    """The answer to "can this fleet do X, and if not what do I tell the user".

    Errors-as-data in the same spirit as ``StageError``: a verdict is a VALUE the
    route turns into a 400 with a named reason, never an exception and never a
    generic "unsupported". ``refusal`` is empty exactly when ``servable`` is True."""
    capability: Capability
    servable: bool
    preset_ids: tuple[str, ...]
    reason: str      # why not; "" when servable
    refusal: str     # full user-facing text; "" when servable


def available_menu() -> str:
    """One line naming everything the fleet CAN render, grouped by capability, e.g.
    ``t2v (clip-t2v-480p, 832x480)``. Sorted by capability value so the text is
    stable across calls — a refusal message that reorders itself looks like a
    different error to anyone diffing logs."""
    parts: list[str] = []
    for cap in sorted(servable_capabilities(), key=lambda c: c.value):
        for p in presets_for(cap):
            parts.append(f"{cap.value} ({p.preset_id}, {p.geometry})")
    return "; ".join(parts)


def capability_verdict(capability: Capability) -> CapabilityVerdict:
    """Is ``capability`` servable on this fleet, and if not, what do we say?

    PURE — no registry mutation, no I/O, no HTTP. TWO layers consume it, and both
    are load-bearing:

      * the HTTP BOUNDARY (``video_routes.video_studio_i2v``) turns a non-servable
        verdict into a 400 BEFORE a job is enqueued. That hop was the whole point of
        this function and it did not exist until 2026-07-27 — measured on the live
        route that day, ``audio`` / ``lipsync`` / ``restore`` / ``stream`` /
        ``keyframe`` all returned 200 + a job_id and burned a queue slot on their way
        to dying in a runner three layers down;
      * the ROUTER (``CapabilityRouter.resolve``) refuses the same set again, so a
        non-HTTP caller (the movie composer, the bus rehydrate path, a test) gets the
        identical answer. Two gates, ONE wording — this function's — so a console and
        a log line can never disagree about why.

    8 of the 16 declared capabilities are servable today; the other 8 refuse here."""
    covered = presets_for(capability)
    if covered:
        return CapabilityVerdict(
            capability=capability, servable=True,
            preset_ids=tuple(p.preset_id for p in covered),
            reason="", refusal="",
        )
    reason = _UNSERVABLE_WHY.get(
        capability,
        "no render preset covers it: no model with weights on disk and a working "
        "runner provides it on this fleet")
    refusal = (
        f"studio cannot render {capability.value!r} on this fleet: {reason}. "
        f"What IS available today: {available_menu()}."
    )
    return CapabilityVerdict(
        capability=capability, servable=False, preset_ids=(),
        reason=reason, refusal=refusal,
    )


def refusal_for(capability: Capability) -> str | None:
    """The refusal text, or None when the capability IS servable. The one-liner form
    of ``capability_verdict`` for a caller that only needs the message."""
    verdict = capability_verdict(capability)
    return None if verdict.servable else verdict.refusal


__all__ = [
    "RENDER_BOX", "RENDER_BOX_VRAM_GIB",
    "WAN_FRAME_CADENCE", "WAN_MAX_FRAMES", "WAN_DEFAULT_FRAMES",
    "ZERO_BYTE_MODELS", "STUB_RUNNER_MODULES",
    "RenderPreset", "CapabilityVerdict",
    "all_presets", "preset", "presets_for", "presets_using",
    "servable_capabilities", "unservable_capabilities",
    "available_menu", "capability_verdict", "refusal_for",
    "is_wan_cadence", "snap_wan_frames",
]

"""Server-side operator authentication gate for console-side management routes.

WHY
---
Authentication for the console used to be enforced ONLY in the React UI. The
Flask layer was wide open, so anyone who could reach the API origin could mint
API keys, mint/revoke worker enrollment tokens, admit/assign workers (→ prompt
exfiltration + SSRF), and drive the Discord console — all unauthenticated. This
gate closes that by validating the operator at the server.

DESIGN
------
A single ``before_request`` matches the request against an explicit allowlist of
SENSITIVE routes (operator-only, mutating, or secret-bearing) and requires
operator auth for those. Everything else — health/readiness, model reads,
inference (``/v1`` is gated by the API-key system), and the machine-to-machine
endpoints the worker / bot / phone arms depend on — is left untouched, so this
never breaks those flows.

Auth modes (resolved from ``HUGPY_AUTH_MODE``, same as ``/auth/config``):
  * ``external`` — validate the first-party session cookie by forwarding it to
    the upstream auth service's ``/me`` (the same session the React UI uses),
    with a short positive/negative cache. A configured ``HUGPY_OPERATOR_TOKEN``
    is also accepted (CLI/automation). Fails CLOSED (deny) if the auth service
    is unreachable — a sensitive route must not open up during an auth outage.
  * ``open`` — the self-hosted single-operator default (``pip install hugpy``).
    Permissive UNLESS ``HUGPY_OPERATOR_TOKEN`` is set, in which case that token
    is required. So the localhost product keeps its no-login UX, while a public
    open deployment can still lock the management surface with one env var.

The gate only ENFORCES in external mode (or when an operator token is set), so
installing it changes nothing until the operator flips ``HUGPY_AUTH_MODE`` —
making rollout safe to deploy and verify before activation.

ROLES (2026-08-06) — the MEMBER tier
------------------------------------
Until today every valid central session was treated as a full operator: the
gate's only question was "does this cookie authenticate", so a registered
Clownworld member who logged in could mint API keys, drive worker admission and
delete models. The upstream ``/me`` payload (``{username, email, is_admin,
status, sites:[...]}``) was fetched and THROWN AWAY.

It is now parsed and kept (``current_principal()``), and the request resolves to
exactly one of three roles (``principal_role()``):

  * ``operator`` — a valid ``HUGPY_OPERATOR_TOKEN`` (CLI/automation, unchanged),
    ``open`` mode with no token set (the self-hosted product, unchanged), or a
    session whose ``/me`` says ``is_admin: true``.
  * ``member``   — a session with ``status == "approved"`` that carries the
    ``hugpy`` or ``clownworld`` site grant. A member may use the Studio/Media
    plane (``/media``, ``/ml``, ``/uploads``, ``/session``, ``/chat``,
    ``/video``) but is REFUSED (403) on every console mutation in ``_SENSITIVE``.
  * ``anonymous`` — everything else (no cookie, unapproved, no site grant, or
    the auth service is unreachable — this layer stays fail-CLOSED).

The console gate's deny SHAPE is role-dependent: operator -> allow, member ->
403 (an honest "you are logged in, this is not yours"), anonymous -> 401
(unchanged, so the SPA's login-probe behavior is untouched).
"""
from __future__ import annotations

import os
import re
import time
import hashlib
import logging
from typing import Optional

from flask import request, abort, jsonify, g

logger = logging.getLogger(__name__)

# Sensitive routes: (allowed-methods-that-require-auth, normalized-path regex).
# Paths are matched AFTER stripping a leading "/api" (gunicorn dual-mounts the
# worker/discord/phone blueprints under /api as well as bare). Only the listed
# methods are gated, so e.g. GET /discord/bridges (bot M2M) stays open while
# POST /discord/bridges (operator) is gated.
_SENSITIVE = [
    # API key management (key minting was anonymously reachable — CRITICAL)
    ({"GET", "POST"},            re.compile(r"^/keys$")),
    ({"DELETE"},                 re.compile(r"^/keys/[^/]+$")),
    ({"PUT"},                    re.compile(r"^/keys/require$")),
    # k9 VIDEO-SHARE key management: mint/list/revoke the video-scoped share
    # links. Operator-only, and deliberately NOT on the /video surface — so a
    # video-share principal (which CAN pass the /video gate) can never reach the
    # mint route to bootstrap another key ("no key-minting-by-key"). The list GET
    # doubles as the SPA's operator-auth probe (200 => show the Share button).
    # (The generic DELETE /keys/<id> rule above matches DELETE /keys/video-share
    # too, but never the 2-segment /keys/video-share/<id> revoke — hence its own.)
    ({"GET", "POST"},            re.compile(r"^/keys/video-share$")),
    ({"DELETE"},                 re.compile(r"^/keys/video-share/[^/]+$")),
    # Studio TESTER sweep (video_routes.video_studio_tester): one prompt fanned out
    # across EVERY servable model of a category — many GPU generations spent on a
    # caller-chosen prompt, the same "spends disk/bandwidth/GPU" tier as
    # /llm/review/run above. Operator INTENT, not a share-link action: gated HERE
    # (not just by the /video session-or-share gate) so a video-share principal —
    # which CAN pass the /video gate — can never trigger a fleet-wide sweep. The
    # /video/studio render routes stay share-reachable; only the sweep is operator-only.
    ({"POST"},                   re.compile(r"^/video/studio/tester$")),
    # Worker enrollment tokens (minting/revoking enrollment — CRITICAL)
    ({"GET", "POST"},            re.compile(r"^/llm/enroll-tokens$")),
    ({"DELETE"},                 re.compile(r"^/llm/enroll-tokens/[^/]+$")),
    # One-step WireGuard join (2026-09-24): allocates a fleet IP, registers a
    # wg0 peer via a privileged helper, and MINTS an enrollment token + returns a
    # private key. Strictly more powerful than /llm/enroll-tokens (it also grants
    # network reachability into the hub subnet), so it is operator-only, same
    # CRITICAL tier — and it must never be reachable anonymously.
    ({"POST"},                   re.compile(r"^/llm/fleet/wg-join$")),
    # FLEET DISTRIBUTION MODE (operator ruling 2026-09-24): feasible|designated,
    # the catch-all that flips the whole fleet between feasible-set routing and
    # the legacy sealed-designation scope. A fleet-wide routing-registry write of
    # the same tier as the per-worker wildcard opt-in, so the POST is operator-
    # only. The GET (/llm/fleet/distribution) stays open — same tier as the
    # /llm/workers roster and the wildcard map it sits beside.
    ({"POST"},                   re.compile(r"^/llm/fleet/distribution$")),
    # Model review (review_routes): /run spends disk, bandwidth and GPU time on
    # weights chosen by the caller, and /criteria writes the saved queries the
    # unattended timer later executes — both are operator intent, not public
    # reads. /screen and the result GETs stay open: metadata only, no side
    # effects. Anonymous /run would be a remote "fill the model store" button.
    ({"POST"},                   re.compile(r"^/llm/review/run$")),
    ({"PUT"},                    re.compile(r"^/llm/review/criteria/[^/]+$")),
    # Worker admission / control — operator actions (register & heartbeat are
    # M2M and deliberately NOT here). Admission is what makes a worker
    # dispatch-eligible, so gating it closes anonymous self-admission → SSRF.
    # (alloc-all = bulk GPU-allocation write for a selection of a worker's models
    #  — worker_routes._apply_alloc_map; same registry-write privilege as assign.)
    ({"POST"},                   re.compile(r"^/llm/workers/[^/]+/(admit|block|admission|assign|unassign|alloc-all|unload|probe|fetch|pool|limits|load|moe|bnb|auto-reap)$")),
    # PIN + automated-designation PRUNE (2026-09-23): the 📌 pin is operator
    # intent (what a worker reloads after a boot / central restart) and the
    # prune drops designations — both registry writes of the assign tier. The
    # prune is gated even as a dry run (one verb, one rule).
    ({"POST"},                   re.compile(r"^/llm/workers/[^/]+/(pin|designations/prune)$")),
    # Per-worker ⭐ STAR ("star") — operator intent that projects onto the fleet
    # (a routing tie-break designation; it loads nothing), same registry-write
    # privilege tier as assign. The GET map
    # (/llm/workers/boot-prewarm) and the per-worker read (surfaced on the roster)
    # stay open — only the write is gated. Two path segments (worker id + verb)
    # with a hyphen, so it needs its own rule (the single-segment worker-verb rule
    # above does not match it).
    ({"POST"},                   re.compile(r"^/llm/workers/[^/]+/boot-prewarm$")),
    # Per-worker WILDCARD routing opt-in ("take all comers", operator doctrine
    # 2026-07-23) — a routing-registry write, same tier as assign/boot-prewarm.
    # The GET map (/llm/workers/wildcard) and the roster surfacing stay open —
    # only the write is gated. Sibling rule to boot-prewarm above (the
    # worker-verb alternation rule doesn't list this verb).
    ({"POST"},                   re.compile(r"^/llm/workers/[^/]+/wildcard$")),
    ({"DELETE"},                 re.compile(r"^/llm/workers/[^/]+$")),
    # k10: sanctioned ghost-cleanup for the assignment-memory sidecar
    # (worker_assignments.json) — same operator-only tier as the row DELETE above.
    ({"DELETE"},                 re.compile(r"^/llm/workers/[^/]+/memory$")),
    # Model-level BLOCK from the serving pool (operator pool primitive — the
    # global sibling of the per-worker block verb above). Same operator-only tier
    # as assign: block/unblock are routing-registry writes. model_key can contain
    # slashes (`<path:...>`), so `.+` spans it; the GET placement/meta reads stay
    # open. Matched after the /api strip, bare and dual-mounted.
    ({"POST"},                   re.compile(r"^/llm/models/.+/(un)?block$")),
    # ARCHIVE MARK (2026-09-23): mark (POST) / unmark (DELETE) a model for
    # archive — a placement/routing-registry write of the same tier as block.
    # The GET on the same path is the worker tar stream (its own credential
    # gate in worker_routes), deliberately not listed here.
    ({"POST", "DELETE"},         re.compile(r"^/llm/models/.+/archive$")),
    # Post-download admission re-run (2026-09-23): re-queues the static audit +
    # benchmark for a model and flips a HELD model back to routable pending.
    ({"POST"},                   re.compile(r"^/llm/admission/.+/rerun$")),
    # EXPLICIT model PRIORITY GROUPS (operator directive 2026-08-06): an ordered
    # fallback list that decides WHICH model key a request resolves to. That is
    # a routing-registry write of exactly the same tier as assign/block, so
    # every mutation is operator-only. The two GETs — the listing and
    # /llm/model-groups/resolve (the preview) — are deliberately NOT here: they
    # are console reads, member-visible like every other read, and the preview
    # routes nothing. Note the resolve path would be caught by the <id> rule
    # below if it were ever given a write verb; it has none.
    ({"POST"},                   re.compile(r"^/llm/model-groups$")),
    ({"PUT", "PATCH", "DELETE"}, re.compile(r"^/llm/model-groups/[^/]+$")),
    # Group ALLOCATE (2026-08-25): fans the group's members out as designations
    # (worker["models"] writes) — same tier as /assign, so operator-only.
    ({"POST"},                   re.compile(r"^/llm/model-groups/[^/]+/allocate$")),
    # Group MEMBER move (2026-08-25): the model table's Group column — a
    # membership write like POST/PUT above, so operator-only.
    ({"POST"},                   re.compile(r"^/llm/model-groups/member$")),
    # TASK TEMPLATES (2026-08-26): blueprint writes + activate/deactivate.
    # Activation POOLS workers (removes them from general serving) and fans out
    # designations — the same tier as /pool + /assign, so operator-only. The
    # GET listing stays member-visible like every other read.
    ({"POST"},                   re.compile(r"^/llm/templates$")),
    ({"PUT", "PATCH", "DELETE"}, re.compile(r"^/llm/templates/[^/]+$")),
    ({"POST"},                   re.compile(r"^/llm/templates/[^/]+/(activate|deactivate)$")),
    # Serving / slot control (operator) — the GET status reads stay open.
    ({"POST"},                   re.compile(r"^/llm/serving/[^/]+$")),
    ({"POST"},                   re.compile(r"^/llm/slots/(load|unload)$")),
    # File uploads are intentionally NOT operator-gated: the media-intelligence
    # arm needs them for any authenticated user (upload -> /ml/vision|/ml/extract).
    # Same exposure tier as /chat/stream and /ml/* — the user-facing product routes.
    # Discord HUMAN console routes. The bot's M2M calls (GET /discord/resolve,
    # POST /discord/outbox/drain, POST /discord/channels, POST /discord/users,
    # POST /discord/inbox, GET /discord/bridges) are intentionally excluded.
    ({"GET", "POST", "DELETE"},  re.compile(r"^/discord/bindings(/[^/]+)?$")),
    ({"POST"},                   re.compile(r"^/discord/bridges$")),
    ({"DELETE"},                 re.compile(r"^/discord/bridges/[^/]+$")),
    ({"POST"},                   re.compile(r"^/discord/bridges/[^/]+/(send|keeper-reply|approve|reject)$")),
    ({"GET", "DELETE"},          re.compile(r"^/discord/bridges/[^/]+/messages$")),
    # Comms sessions: minting/listing/revoking the scoped bearer tokens is
    # operator-only. The /discord/session/<token>/… verbs are deliberately NOT
    # here — the session token IS their credential (same rationale as
    # principal tokens below).
    ({"GET", "POST"},            re.compile(r"^/discord/sessions$")),
    ({"DELETE"},                 re.compile(r"^/discord/sessions/[^/]+$")),
    # F2 principals: minting identities/tokens is operator-only. The
    # /auth/discord-link handshake and /auth/whoami stay open — the principal
    # token IS their credential.
    ({"GET", "POST"},            re.compile(r"^/auth/principals$")),
    ({"DELETE"},                 re.compile(r"^/auth/principals/[^/]+$")),
    ({"POST"},                   re.compile(r"^/auth/principals/[^/]+/token$")),
    # F4 settings: reads stay open (UIs render from them); writes are the
    # console's authoritative control plane (CON-08) -> operator-only.
    ({"POST", "PUT", "DELETE"},  re.compile(r"^/settings/.+$")),
    # Fleet templates (FLEET-TEMPLATES-DESIGN §6): the template DEFINITIONS are
    # operator intent that projects onto the fleet, so writes are operator-only.
    # GET (list/get/active) and POST .../diff stay OPEN — diff is a read-only
    # dry-run (no writes, no relays), the review gate the console renders before
    # any (Slice 1+) apply. Save/delete a named template + snapshot-the-live-fleet
    # are the writes gated here. (The "snapshot" literal is caught by the
    # <name> rule under PUT/DELETE too, harmlessly — it's a write either way.)
    ({"PUT", "DELETE"},          re.compile(r"^/fleet/templates/[^/]+$")),
    ({"POST"},                   re.compile(r"^/fleet/templates/snapshot$")),
    # Worker ops (CON-05/06, UTIL-02): restart / module update / pip install /
    # serving-config are privileged executor actions on a worker —
    # operator-only, audited. (config added 2026-07-03: it re-execs the agent
    # and rewrites its runtime settings — same privilege tier as update.)
    # pin-all/unpin-all relay the SAME /ops/config write in bulk (see
    # worker_routes._relay_pin_all) — same privilege tier as config.
    # residency-all (todo t12) sets the RESIDENCY tier of a SELECTED set of a
    # worker's models in one /ops/config write (worker_routes._relay_residency_map)
    # — the same privilege tier as config/pin-all; residency only, never pin.
    # reap-approve = operator-approved eviction of cold local models (drives the
    # same guarded reaper as /reap, with a central intersection second guard).
    # free-ram = non-destructive host-RAM reclaim (gc + malloc_trim + CUDA
    # empty_cache on the worker); it runs a privileged executor op on the box,
    # so it sits in the same operator-only tier as the other worker ops.
    # evict = targeted per-model RAM+VRAM reclaim (slot child kill / in-process
    # ref-drop / comfy /free) — a privileged destructive executor op on the box,
    # same operator-only tier as unload/free-ram.
    # reap-orphans (k32) = kill a worker's own-venv GPU children whose slot
    # claim cleared but whose process never exited — the ONE place central
    # reaches a worker's raw PIDs (agent._reap_gpu_orphans, fail-closed 4-gate
    # admission). Same operator-only, destructive-executor tier as evict/reap;
    # dry_run defaults true so a bare POST previews before it ever kills.
    # external-set (2026-08-12) = adjust a running gpu_lease's evictable/resume
    # policy (console twin of hugpy-lease-set) — flipping evictable can pause a
    # live batch job, so it sits in the same tier as evict. The GET .../external
    # read stays open like every other roster/residents read.
    ({"POST"},                   re.compile(r"^/llm/workers/[^/]+/(restart|update|pip|config|reap|reap-approve|reap-orphans|pin-all|unpin-all|residency-all|free-ram|cache-evict|evict|external-set)$")),
    # FLEET-WIDE eviction policy (2026-07-25): the drop-pass switch applies to
    # EVERY worker at once and changes which models an admission unloads, so the
    # write sits in the same operator-only tier as the per-worker config it
    # complements. The GET stays open (same tier as the /llm/workers roster).
    ({"POST"},                   re.compile(r"^/llm/evict-policy$")),
    # k14: relaunch a worker's slot child with a new GPU-offload depth / context
    # (the offload speed-cliff sweep lever). A privileged executor op on the box —
    # it STOP->RESPAWNs a llama-server child — so it sits in the same operator-only
    # tier as the other worker ops above. Two path segments (slot id + verb), so it
    # needs its own rule (the single-segment worker-verb rule does not match it).
    ({"POST"},                   re.compile(r"^/llm/workers/[^/]+/slots/[^/]+/relaunch$")),
    # stranded-slot fix (2026-07-25): unconditional slot-id-addressed unload —
    # kills whatever a slot's child currently is regardless of its model_key
    # claim (the gap /evict's model_key-addressing and /reap-orphans' claimed-
    # pid gate both miss). Same privileged-executor, operator-only tier as
    # relaunch/evict; sibling rule (same two-segment shape as relaunch above).
    ({"POST"},                   re.compile(r"^/llm/workers/[^/]+/slots/[^/]+/unload$")),
    # P3.1 agent-node fleet: the operator-facing routes only. GET /agent/nodes
    # (the fleet roster) and POST /agent/<id>/dispatch (queue a task on a node)
    # are operator intent — gated here too, belt-and-suspenders with the
    # blueprint's own operator_authenticated() check. The node-facing M2M routes
    # (POST /agent/register, POST /agent/<id>/heartbeat, GET /agent/<id>/tasks)
    # are deliberately NOT here — their credential is the node's enroll token.
    ({"GET"},                    re.compile(r"^/agent/nodes$")),
    ({"POST"},                   re.compile(r"^/agent/[^/]+/dispatch$")),
    # 2026-07-23 secure one-time install links: NOT in this operator-only
    # inventory as of 2026-08-06. A MEMBER may mint an install link for their
    # OWN machine (that is the product: install the fleet console you are
    # entitled to), so the whole surface is gated by the blueprint's own
    # _require_member_strict / _require_link_admin instead of a blanket rule
    # here — the gate that can express "operator OR the link's creator", which a
    # (methods, path) rule cannot. What that route layer enforces:
    #   * POST   -> member-or-operator; the creator's username is RECORDED on the
    #              link, and a non-operator's scope request is CLAMPED to the
    #              product scopes (never "full", never "agent-register").
    #   * GET    -> member sees ONLY their own links; an operator sees all.
    #   * DELETE -> operator, or the member who created that link.
    # It still fails CLOSED for anonymous (401) and still ignores the
    # HUGPY_AGENT_OPEN waiver — credential-minting is never waivable.
    # The download GET /agent/install/<link_id> is, as before, deliberately
    # ungated — the unguessable link_id IS its capability (like a share link).
    # P3.1b: the single-task detail read (a run's full row incl. its result) is
    # the operator's drill-in for the P3.3 console — gated like /agent/nodes.
    # Scoped to GET and to the /tasks/<seq> shape (with a trailing seq), so the
    # node-token pull (GET /agent/<id>/tasks — no seg) and the node-token result
    # POST (POST /agent/<id>/tasks/<seq>/result — extra seg) both stay M2M-open.
    ({"GET"},                    re.compile(r"^/agent/[^/]+/tasks/[^/]+$")),
    # The console Help button's help agent (routes/help_routes.py): its backend
    # reads logs and EDITS CODE, so every verb is operator-only. Enforced in the
    # route too, without the HUGPY_AGENT_OPEN waiver (path is /llm/, not /agent/).
    ({"GET", "POST", "DELETE"},  re.compile(r"^/llm/help(/.*)?$")),
    # Civitai checkpoint download — writes multi-GB files into central's
    # /checkpoints store (which self-registers models) — operator-only.
    ({"POST"},                   re.compile(r"^/civitai/download$")),
    # Disk discovery sweep — rebuilds the discovery report (walks the whole
    # model tree + hub enrichment); the GET state poll stays open.
    ({"POST"},                   re.compile(r"^/models/discover$")),
    # Hugging Face credentials (k29): the stored HF token is a secret and the
    # write path mutates central's auth to HF — operator-only for GET/POST/DELETE.
    # GET is gated too (it validates the token against HF and reveals its source);
    # the token itself is never returned (only last4).
    ({"GET", "POST", "DELETE"},  re.compile(r"^/llm/hf/auth$")),
    # HF metadata cache forget (fetch-once policy hatch): dropping a repo's
    # cached rows re-arms a LIVE HF fetch on next access — an operator-only
    # refresh affordance. The GET /hf/cache stats read stays open.
    ({"DELETE"},                 re.compile(r"^/hf/cache/.+$")),
    # Store reconcile (the flattening migration) — MOVES/ARCHIVES model dirs and
    # rewrites the registry + markers when {"apply": true}. A mutating store op,
    # same operator-only tier as discover/delete. The dry-run is also POST (it
    # writes a plan report), so the whole route is gated.
    ({"POST"},                   re.compile(r"^/models/reconcile$")),
    # ---------------------------------------------------------------------- #
    # 2026-08-06 MEMBER TIER — the console-plane mutations that were never in
    # this inventory at all. They were reachable by ANY valid session (and,
    # before the roles landed, that meant every member was a full operator).
    # Each of these spends disk/bandwidth/GPU or destroys state on the box, so
    # they sit in exactly the same operator-only tier as the model/worker verbs
    # above. Reads (GET /models, GET /jobs, GET /llm/repos, …) stay OPEN — this
    # is a MUTATION inventory, and the console must still render for a member.
    # ---------------------------------------------------------------------- #
    # Model store writes: pull multi-GB weights, delete a model tree, prune its
    # revisions, or set the media/poster image the console renders for it.
    # model_key is a plain segment on these rules (the <path:...> variants live
    # under /llm/models, covered above).
    ({"POST"},                   re.compile(r"^/models/[^/]+/download$")),
    ({"DELETE"},                 re.compile(r"^/models/[^/]+$")),
    ({"POST"},                   re.compile(r"^/models/[^/]+/(prune|media|media-default)$")),
    # Bulk re-classification rewrites the image/model split across the whole
    # store — the same mutating-store tier as reconcile/discover.
    ({"POST"},                   re.compile(r"^/models/reclassify-images$")),
    # Job control on the download/maintenance queue: cancelling or retrying
    # someone else's fleet job is an operator action (the media/studio jobs a
    # member owns are cancelled through /video/jobs/<id>/cancel, which the
    # ownership filter guards — deliberately NOT here).
    # discard = dismissing a persistent failure record; diagnose = spending
    # keeper inference. Both are the same operator tier as cancel/retry (k121).
    ({"POST"},                   re.compile(r"^/jobs/[^/]+/(cancel|retry|discard|diagnose)$")),
    # Bulk repo pull — the same "fill the model store from the internet" tier as
    # /civitai/download and /models/<key>/download.
    ({"POST"},                   re.compile(r"^/llm/repos/download$")),
    # Phone-brick pool: enrolling a phone, running/cancelling a task on one, and
    # deleting a phone row are privileged executor actions on real devices. The
    # M2M device paths (POST /phone-brick/register, POST /phone-brick/<id>/heartbeat)
    # are deliberately excluded — their credential is the enroll flow — as are the
    # installer asset GETs (code.tar.gz / install.sh / files/<...>) the device pulls.
    ({"POST"},                   re.compile(r"^/phone-brick/run$")),
    ({"POST"},                   re.compile(r"^/phone-brick/runs/[^/]+/cancel$")),
    ({"DELETE"},                 re.compile(r"^/phone-brick/phones/[^/]+$")),
]

_SESSION_CACHE: dict[str, tuple[bool, float]] = {}
# The PARSED upstream /me payload for the same cookie hash, same 30s window as
# _SESSION_CACHE (both are filled by the one _fetch_me call, so adding roles
# costs ZERO extra round trips to the auth service). Kept as a separate dict so
# the existing tests that reach in and clear _SESSION_CACHE keep working; the
# clear helper below empties both.
_PRINCIPAL_CACHE: dict[str, tuple[Optional[dict], float]] = {}
_CACHE_TTL = 30.0

# Site grants that make an approved central account a hugpy MEMBER. "clownworld"
# is here per the 2026-08-06 spec: a registered Clownworld account is entitled to
# hugpy's studio/media/fleet surface without a second registration.
MEMBER_SITES = frozenset({"hugpy", "clownworld"})

ROLE_OPERATOR = "operator"
ROLE_MEMBER = "member"
ROLE_ANONYMOUS = "anonymous"


def _clear_session_caches() -> None:
    """Drop both cookie-hash caches (used by tests / a forced re-validate)."""
    _SESSION_CACHE.clear()
    _PRINCIPAL_CACHE.clear()


def _auth_mode() -> str:
    mode = (os.environ.get("HUGPY_AUTH_MODE") or "external").lower()
    return mode if mode in ("open", "external") else "external"


def _operator_token() -> str:
    return (os.environ.get("HUGPY_OPERATOR_TOKEN") or "").strip()


def _provided_token() -> str:
    t = request.headers.get("X-Operator-Token")
    if t:
        return t.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _cookie_key() -> str:
    """Cache key for this request's cookie jar ("" when it carries none)."""
    cookie_hdr = request.headers.get("Cookie", "")
    if not cookie_hdr:
        return ""
    return hashlib.sha256(cookie_hdr.encode("utf-8")).hexdigest()


def _site_tokens(sites) -> set:
    """Normalized (lowercased) site names from a ``/me`` ``sites`` value.

    The central service returns a list of site names today; dict rows are
    accepted too, so a payload shape we did not expect can never silently read
    as "zero site grants"."""
    out: set = set()
    if isinstance(sites, str):
        sites = [sites]
    if not isinstance(sites, (list, tuple, set)):
        return out
    for entry in sites:
        val = None
        if isinstance(entry, str):
            val = entry
        elif isinstance(entry, dict):
            for k in ("site", "name", "slug", "domain", "id"):
                if isinstance(entry.get(k), str):
                    val = entry[k]
                    break
        if val:
            out.add(val.strip().lower())
    return out


def _normalize_principal(data) -> Optional[dict]:
    """The upstream ``/me`` body -> the principal shape this module exposes, or
    None when the body is not a usable identity (error envelope / wrong type)."""
    if not isinstance(data, dict) or data.get("error"):
        return None
    username = data.get("username") or data.get("user") or data.get("email")
    if not isinstance(username, str) or not username.strip():
        return None
    return {
        "username": username.strip(),
        "email": data.get("email") if isinstance(data.get("email"), str) else None,
        "is_admin": bool(data.get("is_admin")),
        "status": (data.get("status") or "").strip().lower()
                  if isinstance(data.get("status"), str) else "",
        "sites": sorted(_site_tokens(data.get("sites"))),
    }


def _fetch_me():
    """(ok, principal|None) for this request's cookies, straight from upstream.

    ``ok`` mirrors the pre-roles boolean EXACTLY (200 + a non-error body), so the
    console gate's accept set does not move. ``principal`` is the parsed identity
    when the body is one — that payload used to be fetched and discarded, which
    is precisely why every session read as a full operator.

    Raises nothing: an unreachable/misbehaving auth service returns
    ``(False, None)`` and is NOT cached (fail CLOSED, retry next request)."""
    try:
        import requests
        from hugpy_server.app.routes.auth_proxy_routes import upstream_base
        resp = requests.get(f"{upstream_base()}/me", cookies=request.cookies,
                            timeout=8)
    except Exception as exc:  # noqa: BLE001
        # Fail closed: a sensitive route must not open up if auth is unreachable.
        logger.warning("operator session validation failed (auth service): %s", exc)
        return None
    if resp.status_code != 200:
        return (False, None)
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001 — a 200 with an unparseable body
        return (True, None)
    ok = bool(data) and not (isinstance(data, dict) and data.get("error"))
    return (ok, _normalize_principal(data) if ok else None)


def _resolve_session(key: str):
    """(ok, principal|None) for a cookie hash, through the 30s cache. Returns
    None (uncached, fail-closed) when the auth service could not be reached."""
    now = time.time()
    cached_ok = _SESSION_CACHE.get(key)
    cached_p = _PRINCIPAL_CACHE.get(key)
    if cached_ok and cached_ok[1] > now and cached_p and cached_p[1] > now:
        return (cached_ok[0], cached_p[0])
    fetched = _fetch_me()
    if fetched is None:
        return None
    ok, principal = fetched
    _SESSION_CACHE[key] = (ok, now + _CACHE_TTL)
    _PRINCIPAL_CACHE[key] = (principal, now + _CACHE_TTL)
    return (ok, principal)


class _Unset:
    """Sentinel so a per-request cache of None is a HIT, not a re-fetch."""
    __slots__ = ()


_UNSET = _Unset()


def current_principal() -> Optional[dict]:
    """The authenticated central identity behind THIS request, or None.

    ``{"username", "email", "is_admin", "sites": [...], "status"}``. Cached per
    request on ``flask.g`` (so a gate, a route and a listing filter that each ask
    cost one upstream call at most) on top of the 30s cookie-hash cache.

    None for: no cookie at all, an anonymous/expired session, or an auth-service
    outage (fail CLOSED — never a synthesized identity)."""
    try:
        cached = getattr(g, "_hugpy_principal", _UNSET)
        if cached is not _UNSET:
            return cached
    except RuntimeError:      # outside an app context (CLI import, unit call)
        pass
    key = _cookie_key()
    principal = None
    if key:
        resolved = _resolve_session(key)
        if resolved is not None:
            principal = resolved[1]
    try:
        g._hugpy_principal = principal
    except RuntimeError:
        pass
    return principal


def _validate_session_external() -> bool:
    """True iff the request's cookies authenticate against the upstream /me.

    Signature and semantics are UNCHANGED (the boolean the console gate has
    always used, and the seam tests/test_video_gate.py monkeypatches); it now
    shares one fetch + cache with current_principal()."""
    key = _cookie_key()
    if not key:
        return False
    resolved = _resolve_session(key)
    if resolved is None:
        return False          # auth service unreachable -> fail closed
    return resolved[0]


def is_member_principal(p) -> bool:
    """The MEMBER rule: an approved central account holding a hugpy-family site
    grant (or an admin, who is a member of everything by definition)."""
    if not isinstance(p, dict):
        return False
    if p.get("status") != "approved":
        return False
    if p.get("is_admin"):
        return True
    return bool(MEMBER_SITES & set(p.get("sites") or ()))


def principal_role() -> str:
    """``"operator" | "member" | "anonymous"`` for THIS request.

    Resolution order (first match wins):
      1. the configured ``HUGPY_OPERATOR_TOKEN`` presented as ``X-Operator-Token``
         or a bearer -> operator (CLI/automation; unchanged);
      2. ``open`` mode with no token configured -> operator (the self-hosted
         single-operator product; unchanged);
      3. a session whose ``/me`` says ``is_admin`` -> operator;
      4. an approved session with a hugpy/clownworld site grant -> member;
      5. anything else -> anonymous.
    """
    tok = _operator_token()
    provided = _provided_token()
    if tok and provided and provided == tok:
        return ROLE_OPERATOR
    # DEV control plane: a valid hugpy API key IS full authority. This is a
    # LAN/WireGuard-allowlisted dev box — there is nothing to guard behind a
    # SEPARATE operator credential, and requiring one only broke automation
    # (the console ⬆ Update, /evict, /serving, /cache-evict all 401'd on a
    # perfectly valid key). The key already gates /v1; honor it here too.
    if provided and provided.startswith("hp_"):
        try:
            from hugpy_server.app.functions.imports.utils.api_keys import verify_api_key
            if verify_api_key(provided):
                return ROLE_OPERATOR
        except Exception:
            pass
    if _auth_mode() == "open" and not tok:
        return ROLE_OPERATOR
    p = current_principal()
    if p is None:
        # No parseable identity. Either there is no session at all, or the
        # upstream answered 200 with a body we could not read as an identity —
        # in which case the legacy boolean is still the authority, and it means
        # "operator" exactly as it did before roles existed. (This is also the
        # branch the gate tests exercise when they stub the boolean seam.)
        return ROLE_OPERATOR if _validate_session_external() else ROLE_ANONYMOUS
    if p.get("is_admin"):
        return ROLE_OPERATOR
    if is_member_principal(p):
        return ROLE_MEMBER
    return ROLE_ANONYMOUS


def member_authenticated() -> bool:
    """True for a member OR an operator — the Studio/Media plane's accept rule
    (see member_auth.py). Never true for anonymous."""
    return principal_role() in (ROLE_OPERATOR, ROLE_MEMBER)


def principal_username() -> Optional[str]:
    """The central username behind this request, or None (operator-token M2M,
    a share credential, open mode, or anonymous). This is the ARTIFACT OWNER
    string threaded onto media jobs and upload namespaces."""
    p = current_principal()
    return p.get("username") if isinstance(p, dict) else None


# Uploads are namespaced per account (upload_routes): UPLOADS_HOME/<namespace>/…
# The namespace is derived HERE so the writer (upload_routes) and every reader
# (video_routes' raw-path ownership check) can never drift on the mapping.
_NS_UNSAFE = re.compile(r"[^A-Za-z0-9_.@=+-]+")
# Directory names under UPLOADS_HOME that are NOT account namespaces: the upload
# session registry and the imagegen runner's output dir. A username that
# sanitizes onto one of these is prefixed rather than allowed to collide.
_RESERVED_NAMESPACES = frozenset({".sessions", "sessions", "generated"})


def upload_namespace(username: Optional[str]) -> Optional[str]:
    """The uploads subdirectory name for ``username``, or None when there is no
    account (operator-token M2M, open mode, a share link) — in which case the
    caller keeps the historical FLAT upload path.

    Sanitized to a single safe path segment: no separators, no leading dot, and
    never one of the reserved directory names."""
    if not username or not isinstance(username, str):
        return None
    ns = _NS_UNSAFE.sub("_", username.strip()).lstrip(".")
    if not ns:
        return None
    if ns in _RESERVED_NAMESPACES:
        ns = "u_" + ns
    return ns[:64]


def operator_authenticated() -> bool:
    mode = _auth_mode()
    tok = _operator_token()
    if tok and _provided_token() and _provided_token() == tok:
        return True
    if mode == "open":
        # Self-hosted single-operator default: permissive unless a token is set.
        return not tok
    # A session is an OPERATOR session only when the upstream says is_admin (or
    # when there is no parseable principal — the pre-roles boolean, see
    # principal_role()). A plain member no longer satisfies the console gate.
    return principal_role() == ROLE_OPERATOR


def _agent_gates_open() -> bool:
    """Mirror of agent_routes._agent_gates_open (operator-directed 2026-07-15:
    agents feature ungated "for now"): ``HUGPY_AGENT_OPEN`` truthy exempts the
    /agent/* operator rules in THIS belt-and-suspenders layer too, so the flag
    opens the feature end-to-end. Every other sensitive path stays gated."""
    return (os.environ.get("HUGPY_AGENT_OPEN", "") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _path_is_sensitive() -> bool:
    path = request.path or "/"
    if path == "/api" or path.startswith("/api/"):
        path = path[len("/api"):] or "/"
    method = request.method
    for methods, rx in _SENSITIVE:
        if method in methods and rx.match(path):
            # The agent-fleet VIEW rules honor the open flag. (The install-link
            # rules used to be carved out of this waiver because they MINT
            # credentials; they no longer live in _SENSITIVE at all — see the
            # note there — and agent_routes' own _require_member_strict does not
            # honor the waiver either, so that carve-out is preserved where it
            # is now enforced.)
            if path.startswith("/agent/") and _agent_gates_open():
                continue
            return True
    return False


def install_operator_gate(app) -> None:
    """Register the before_request gate on a Flask app (idempotent)."""
    if getattr(app, "_operator_gate_installed", False):
        return
    app._operator_gate_installed = True

    @app.before_request
    def _operator_gate():
        if request.method == "OPTIONS":
            return None  # never block CORS preflight
        # ── TESTING LOCKDOWN (operator ask 2026-08-13) ─────────────────────
        # `HUGPY_TESTING_LOCKDOWN` truthy => the console serves OPERATOR
        # requests ONLY. Everything else — members, the Discord bot, demo
        # traffic — gets an honest 503, so a model battery / fleet test runs
        # with nothing else able to trigger loads or renders mid-measurement.
        # Exempt: worker seams (heartbeat/registration/eviction ingest — the
        # fleet must keep breathing) and health probes. Flip on/off via the
        # service drop-in + restart; no code change.
        if (os.environ.get("HUGPY_TESTING_LOCKDOWN", "")
                or "").strip().lower() in ("1", "true", "yes", "on"):
            _p = request.path or "/"
            if _p == "/api" or _p.startswith("/api/"):
                _p = _p[len("/api"):] or "/"
            # /llm/models covers the worker manifest/list GETs the provision
            # sweep needs (lockdown 503s here silently broke model registration
            # on ae, 2026-08-13); mutations under it stay operator-gated by the
            # _SENSITIVE check below regardless of this exemption.
            _exempt = ("/health", "/llm/workers", "/llm/evictions", "/llm/models")
            if not (_p.startswith(_exempt) or principal_role() == ROLE_OPERATOR):
                return jsonify({"error": (
                    "console is in TESTING LOCKDOWN — operator key required "
                    "(model battery / fleet test in progress)")}), 503
        if not _path_is_sensitive():
            return None
        role = principal_role()
        if role == ROLE_OPERATOR:
            return None
        if role == ROLE_MEMBER:
            # A LOGGED-IN member on a console mutation. 403, not 401: the caller
            # IS authenticated, so bouncing them to the login wall (401 is the
            # SPA's "you need to log in" signal) would be a lie and a redirect
            # loop. The console plane is read-mostly for members by design —
            # their surface is Studio/Media (see member_auth.py).
            return jsonify({"error": "forbidden: read-only member access"}), 403
        abort(401, description="Operator authentication required for this route.")

    logger.info("operator auth gate installed (mode=%s, token_set=%s)",
                _auth_mode(), bool(_operator_token()))

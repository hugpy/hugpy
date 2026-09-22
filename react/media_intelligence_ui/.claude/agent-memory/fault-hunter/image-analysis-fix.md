---
name: image-analysis-fix
description: Audit notes for the 2026-06-25 image-analysis/vision fix surface on dev VM — file roles, the two chat_schemas, /uploads exposure verdict
metadata:
  type: project
---

The image-analysis fix (vision now delegates to a worker like any task, with local fallback; inline base64 images instead of /uploads round-trip).

**The two chat_schemas are NOT duplicates of the same class — do not flag as divergence.**
- `imports/src/schemas/chat_schemas.py` = the real `ChatRequest` prompt model (frozen pydantic, has `_normalize_multimodal` multimodal funnel + `images: list[str]`). This is the live one (imported via `imports/src/schemas/__init__` and managers/*). 
- `flask_app/app/functions/imports/utils/schemas/chat_schemas.py` = `ChatBody`/`Message` — the HTTP request-body model for the Flask route, a DIFFERENT class. So "the fix landed in only one copy" is a non-issue; they were never copies.

**`supports_vision` lives in `gguf_worker/agent.py` (engine self-report), NOT in `flask_app/.../utils/workers.py`.** workers.py has `_engine_unusable` (skips engine.installed==False) but no vision-specific capability filter — the design intentionally does NOT second-guess a live worker's vision capability (see remote.py make_delegating_runner comment). Accepted: a CPU-only worker assigned a vision model can still be picked; guard is the operator's assignment choice + mmproj completeness, not a routing filter.

**/uploads security verdict (operator_auth.py removed it from the gate INTENTIONALLY):** NOT an arbitrary-write/traversal/DoS hole.
- `upload_routes.py`: filename = `uuid4().hex[:8] + "_" + secure_filename(f.filename)` → werkzeug secure_filename strips `../`, UUID prefix prevents overwrite. Dest is always under fixed `UPLOADS_HOME` (DEFAULT_ROOT/uploads).
- Body size bounded globally: `wsgi_app.py` sets `MAX_CONTENT_LENGTH` (HUGPY_MAX_UPLOAD_MB, default 100MB).
- Residual (low, accepted-tier): /uploads has NO API-key / auth at all (anonymous upload to disk within UPLOADS_HOME, unbounded file COUNT/total-disk, no content-type validation, no TTL/GC). Same exposure tier as /chat/stream + /ml/* per the design comment.

**Confirmed-clean files in this surface (as of 2026-06-25):** remote.py (no dead _vision_local ref — only legit _force_local loop-guard; local fallback intact), config/main.py get_gguf_file (deterministic, handles empty list), workers.py required_pkg_version (no +local leak — _public() filters file/installed, env honored verbatim by design), ChatPanel.jsx (payload.images correctly gated on att.isImage && att.dataUrl; non-image path unaffected), AttachmentBar.tsx (accent-bg white-fg contrast fix correct, no dead style), no cached asyncio primitives anywhere in surface.

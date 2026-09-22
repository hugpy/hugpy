"""In-memory fakes for the chaos exerciser tests (no live fleet needed).

Not a test_ module (pytest won't collect it). Provides a FakeClient that models
just enough of central for the runner/alloc/observe/assortment code to exercise
end-to-end offline: /models, /llm/workers, /llm/jobs, /health, /models/<k>/meta,
the operator-gated /assign (with an optional engine-gate 409), and /chat/stream
(canned terminal + optional allocation materialisation)."""
from __future__ import annotations

import copy

GIB = 1 << 30


def make_models():
    return [
        {"model_key": "small-gguf", "framework": "gguf",
         "effective_bytes": 2 * GIB, "size_bytes": 2 * GIB,
         "model_max_length": 32768, "tasks": ["text-generation"],
         "primary_task": "text-generation", "blocked": False},
        {"model_key": "tf-model", "framework": "transformers",
         "effective_bytes": 700 * (1 << 20), "size_bytes": 700 * (1 << 20),
         "model_max_length": 4096, "tasks": ["text-generation"],
         "primary_task": "text-generation", "blocked": False},
        {"model_key": "huge-gguf", "framework": "gguf",
         "effective_bytes": 400 * GIB, "size_bytes": 400 * GIB,
         "model_max_length": 32768, "tasks": ["text-generation"],
         "primary_task": "text-generation", "blocked": False},
        {"model_key": "blocked-model", "framework": "gguf",
         "effective_bytes": 2 * GIB, "size_bytes": 2 * GIB,
         "model_max_length": 32768, "tasks": ["text-generation"],
         "primary_task": "text-generation", "blocked": True},
        {"model_key": "image-only", "framework": "comfy",
         "effective_bytes": 6 * GIB, "size_bytes": 6 * GIB,
         "tasks": ["text-to-image"], "primary_task": "text-to-image",
         "blocked": False},
        {"model_key": "unassigned-gguf", "framework": "gguf",
         "effective_bytes": GIB, "size_bytes": GIB,
         "model_max_length": 8192, "tasks": ["text-generation"],
         "primary_task": "text-generation", "blocked": False},
    ]


def make_workers():
    return [
        {"id": "wid-comp", "name": "computron", "status": "online",
         "vram_total": 8 * GIB, "ram_total": 16 * GIB, "vram_free": 7 * GIB,
         "vram_used": GIB, "free_ram": 8 * GIB, "vram_evictions": 0,
         "gpus": [{"index": 0, "memory_free": 7 * GIB}],
         "models": ["small-gguf", "tf-model"],
         "models_local": ["small-gguf", "tf-model"],
         "loaded_models": ["small-gguf"],
         # model_last_picked is additive (ranking evidence for the sweep); other
         # chaos tests ignore it. small-gguf pinned here (1000 > ae's 100).
         "model_last_picked": {"small-gguf": 1000.0},
         "spill_by_model": {"small-gguf": {"n_gpu_layers": -1}},
         "allocations": [], "loaded_detail": {}, "last_load_error": None},
        {"id": "wid-ae", "name": "ae", "status": "online",
         "vram_total": 24 * GIB, "ram_total": 128 * GIB, "vram_free": 6 * GIB,
         "vram_used": 18 * GIB, "free_ram": 90 * GIB, "vram_evictions": 3,
         "gpus": [{"index": 0, "memory_free": 6 * GIB}],
         "models": ["small-gguf", "huge-gguf"],
         "models_local": ["small-gguf", "huge-gguf"],
         "loaded_models": [],
         "model_last_picked": {"huge-gguf": 2000.0, "small-gguf": 100.0},
         "spill_by_model": {}, "allocations": [], "loaded_detail": {},
         "last_load_error": None},
    ]


class FakeClient:
    def __init__(self, *, health_code=200, jobs=None, engine_gate=True,
                 chat_terminal=None, materialize_alloc=None, reservations=None):
        self._models = make_models()
        self._workers = make_workers()
        self._health = health_code
        self._jobs = jobs if jobs is not None else {"jobs": [], "counts": {}}
        self.engine_gate = engine_gate
        self.assign_calls = []          # (worker_id, model_key, spill)
        self.chat_calls = []
        self.unload_calls = []          # (worker_id, model_key)
        self._reservations = reservations if reservations is not None else []
        # a canned /chat/stream terminal (or a default success)
        self._chat_terminal = chat_terminal or {
            "outcome": "done", "served_worker": "computron", "error": None,
            "finish_reason": "stop", "ttft_s": 0.4, "load_duration_s": 1.2,
            "wall_s": 2.0, "tokens": 3, "stages": [["awaiting-load", None, 0.5]],
            "last_token_s": 1.6, "gen_s": 0.4, "tokens_per_s": 5.0}
        # optional: {worker_name: allocation_row} to drop in after a fire
        self._materialize = materialize_alloc or {}
        self.base = "http://fake"

    # reads
    def health(self):
        return self._health

    def models(self):
        return copy.deepcopy(self._models)

    def workers(self):
        return copy.deepcopy(self._workers)

    def jobs(self):
        return copy.deepcopy(self._jobs)

    def reservations(self, include_terminal=False):
        return copy.deepcopy(self._reservations)

    def unload(self, worker_id, model_key):
        """Force a model cold: drop it from the worker's live residency (loaded_
        models + allocations). Registry (spill_by_model) is untouched."""
        self.unload_calls.append((worker_id, model_key))
        w = next((x for x in self._workers if x["id"] == worker_id), None)
        if w is None:
            return 404, {"ok": False, "error": "unknown worker id"}
        w["loaded_models"] = [m for m in (w.get("loaded_models") or [])
                              if m != model_key]
        w["allocations"] = [a for a in (w.get("allocations") or [])
                            if a.get("model_key") != model_key]
        return 200, {"ok": True, "evicted": True}

    def model_meta(self, model_key, vram_gib=None, ctx_pct=None):
        m = next((x for x in self._models if x["model_key"] == model_key), None)
        if not m:
            return {}
        weights = m["effective_bytes"]
        ctx_max = m.get("model_max_length") or 32768
        frac = (ctx_pct or 100) / 100.0
        kv = int(0.3 * weights * frac)
        need = weights + kv
        fits = (vram_gib is None) or (need <= vram_gib * GIB)
        return {"model_key": model_key, "framework": m["framework"],
                "size_bytes": weights, "ctx_max": ctx_max, "params_b": 3.0,
                "quant": "q4_k_m",
                "recommended": {"need_bytes": need, "ctx": int(ctx_max * frac),
                                "fits_vram": fits,
                                "n_gpu_layers": (-1 if fits else 12),
                                "reason": ("fits" if fits else "partial")}}

    # the only mutation
    def assign(self, worker_id, model_key, spill):
        self.assign_calls.append((worker_id, model_key, dict(spill or {})))
        w = next((x for x in self._workers if x["id"] == worker_id), None)
        if w is None:
            return 404, {"error": "unknown worker id"}
        m = next((x for x in self._models if x["model_key"] == model_key), None)
        if m is None:
            return 404, {"error": f"unknown model key '{model_key}'"}
        # engine gate: a GGUF-only spill on a non-gguf model -> 409
        gguf_only = "n_gpu_layers" in (spill or {})
        if self.engine_gate and gguf_only and m["framework"] != "gguf":
            return 409, {"error": f"'{model_key}' is not GGUF — explicit "
                                  "layer/budget spill refused"}
        sbm = w.setdefault("spill_by_model", {})
        if spill:
            sbm[model_key] = dict(spill)
        else:
            sbm.pop(model_key, None)   # {} clears the override (autofit)
        return 200, copy.deepcopy(w)

    # the fire
    def chat_stream(self, model_key, prompt, request_id, max_new_tokens,
                    ceiling_s, read_timeout_s=None, unbounded=None,
                    max_collect_tokens=None):
        self.chat_calls.append(request_id)
        term = dict(self._chat_terminal)
        served = term.get("served_worker")
        # materialise an allocation row on the served worker if configured
        if served and served in self._materialize:
            w = next((x for x in self._workers if x["name"] == served), None)
            if w is not None:
                row = dict(self._materialize[served])
                row["model_key"] = model_key
                w["allocations"] = [a for a in w.get("allocations", [])
                                    if a.get("model_key") != model_key] + [row]
        return term

"""ModelIndexService — the business layer over the repositories (explicit
wiring 2026-09-10, modeled on solcatcher's repos/*/service). Owns: the
enabled() gate, locking, transactions, alias forms, the empty-report guard,
and fail-open degradation. Callers use the module-level functions in
``__init__``; nothing outside this package touches a repository directly.
"""
from __future__ import annotations

import logging

from hugpy_engine.model_index.client import DatabaseClient, enabled
from hugpy_engine.model_index.repositories import (
    CallsRepository,
    DiscoveryRepository,
    MetricsRepository,
    ModelsRepository,
    QuantsRepository,
)
import os.path as _osp
_os_path_basename = _osp.basename
from hugpy_engine.model_index.query_registry import (PresenceQueries, PAIR_KNOB_KEYS, KV_CACHE_TYPES, KV_CACHE_TYPES_NEED_FLASH, MODE_FALLBACK_ORDER, EXPLICIT_SPILL_CHOICES, quant_fit_walk, PLACEMENT_KNOB_KEYS, ModelQueries,
                                                     PairKnobQueries)

logger = logging.getLogger("abstract_hugpy_dev.model_index")


def name_forms(model_name: str) -> list:
    """Alias forms one model may be recorded under: the given key and its
    owner-stripped tail ('Jackrong~X' and 'X' are one model — the serve path
    records under the RESOLVED key, which may drop the owner prefix)."""
    forms = [str(model_name)]
    tail = str(model_name).split("~")[-1]
    if tail and tail not in forms:
        forms.append(tail)
    return forms


class ModelIndexService:

    def __init__(self, db: "DatabaseClient | None" = None):
        self.db = db or DatabaseClient()
        self.discovery = DiscoveryRepository(self.db)
        self.models = ModelsRepository(self.db)
        self.quants = QuantsRepository(self.db)
        self.metrics = MetricsRepository(self.db)
        self.calls = CallsRepository(self.db)
        self._started_pid = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self, cur) -> None:
        """Create every table/index — once per process, on the first call."""
        import os
        if self._started_pid == os.getpid():
            return
        self.discovery.create_table(cur)
        self.quants.create_table(cur)
        self.metrics.create_table(cur)
        self.calls.create_table(cur)
        self.models.create_table(cur)
        self._started_pid = os.getpid()

    # ── discovery report ──────────────────────────────────────────────────
    def save_discovery(self, report: dict) -> bool:
        """Persist a walk's full report in ONE transaction (delete-absent +
        upsert-present — rebuildable-index semantics); refresh model_quants
        from each row's marker `quants`. An EMPTY report never truncates
        (degraded-mount caution — the remedy is 'skip this write', never
        'keep stale rows')."""
        if not enabled() or not isinstance(report, dict):
            return False
        rows = {str(k): v for k, v in report.items() if isinstance(v, dict)}
        if not rows:
            return False
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    names = list(rows)
                    self.discovery.delete_absent(cur, names)
                    self.quants.delete_absent_models(cur, names)
                    for name, row in rows.items():
                        self.discovery.upsert_row(cur, name, row)
                        self.quants.replace_for_model(cur, name,
                                                      row.get("quants"))
            logger.info("registry DB: saved discovery report (%d rows)",
                        len(rows))
            return True
        except Exception as exc:  # noqa: BLE001 — never break the walk
            self.db.mark_unavailable(exc, "saving the discovery report")
            return False

    def load_discovery(self) -> "dict | None":
        """The report as a dict, or None (off/empty/unreachable — the caller
        falls back to the JSON file; empty is None on purpose: before the
        first walk writes, JSON remains authoritative)."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    self.start(cur)
                    rows = self.discovery.fetch_all(cur)
            return rows or None
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "loading the discovery report")
            return None

    def sync_worker_settings(self, model_name: str, worker_settings: dict,
                             model_settings: dict | None = None) -> bool:
        """Mirror per-worker settings into the canonical model relationship.

        The JSON override remains the serving configuration source of truth;
        this keeps its model/worker association searchable in the database.
        """
        if not enabled() or not model_name or not isinstance(worker_settings, dict):
            return False
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    self.models.sync_model_settings(
                        cur, str(model_name), model_settings or {})
                    for worker_id, record in worker_settings.items():
                        if isinstance(record, dict):
                            # The DB OWNS the pair knobs (operator ruling 2026-10-02,
                            # "retire the json write"): the JSON override's mirror
                            # may carry the association + rank, never a knob — so a
                            # legacy /llm/serving write can no longer overwrite what
                            # the console wrote through /knobs.
                            settings = {k: v for k, v in (record.get("settings") or {}).items()
                                        if k not in PAIR_KNOB_KEYS}
                            self.models.sync_worker_settings(
                                cur, str(model_name), str(worker_id),
                                record.get("allocation_rank"),
                                settings)
            return True
        except Exception as exc:  # noqa: BLE001 — settings still persist to JSON
            self.db.mark_unavailable(exc, "syncing model worker settings")
            return False

    # ── serve overrides store (detach step 3, 2026-10-02) ────────────────
    def fetch_all_serving_settings(self):
        """{model name: serving_settings dict} for every model with a non-empty
        override — the serve-overrides map in serve_overrides.json's shape. None
        when the DB is off/faulted (callers fall back to the JSON file)."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(ModelQueries.FETCH_ALL_SERVING_SETTINGS)
                    return {str(n): (dict(s) if isinstance(s, dict) else {}) for n, s in cur.fetchall()}
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "reading serving settings")
            return None

    def set_model_serving_settings(self, model_name: str, settings: dict) -> bool:
        """Write ONE model's serving_settings ({} clears). Used by
        serve/overrides._save for the entries a bulk rewrite (migrations) changed;
        set_override's own path goes through sync_worker_settings."""
        if not enabled() or not model_name:
            return False
        import json as _json
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    cur.execute(ModelQueries.UPSERT_MODEL_SETTINGS,
                                (str(model_name), _json.dumps(settings or {}, sort_keys=True)))
            return True
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "writing serving settings")
            return False

    def fetch_models_for_display(self, search: str = "", limit: int = 500):
        """Read canonical model rows and related records without running DDL."""
        if not enabled():
            return None
        needle = str(search or "").strip()
        pattern = f"%{needle}%"
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(
                        ModelQueries.FETCH_MODEL_DISPLAY,
                        (needle, pattern, pattern, pattern,
                         max(1, min(int(limit or 500), 1000))),
                    )
                    return [{
                        "id": r[0], "name": r[1], "attributes": r[2],
                        "serving_settings": r[3], "hub_id": r[4],
                        "framework": r[5], "source_updated_at": r[6],
                        "quants": r[7], "metrics": r[8],
                        "workers": r[9], "activity": r[10],
                    } for r in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "reading model display records")
            return None

    # ── per-pair operator knobs (DB-owned, 2026-10-01) ───────────────────
    def resolve_model_id(self, model_key):
        """models.id for a console model key (exact name, else the owner-
        stripped tail), or None when the DB is off / the model is unknown."""
        if not enabled() or model_key is None:
            return None
        if isinstance(model_key, int) or str(model_key).isdigit():
            return int(model_key)
        forms = name_forms(str(model_key))
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(PairKnobQueries.RESOLVE_MODEL_ID, (forms, forms[0]))
                    row = cur.fetchone()
                    return int(row[0]) if row else None
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "resolving a model id")
            return None

    def read_pair_knobs(self, model_id: int, worker_id: str):
        """user_settings of ONE pair row, or None (no row / DB off)."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(PairKnobQueries.FETCH_PAIR, (int(model_id), str(worker_id)))
                    row = cur.fetchone()
                    return dict(row[0] or {}) if row else None
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "reading pair knobs")
            return None

    def pair_knobs_by_worker(self, model_id: int, assigned_only: bool = False):
        """{worker_id: user_settings} for every pair row of a model (DB truth for
        per-worker knobs such as the quant pin). None when the DB is off."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute("SELECT worker_id, user_settings, assigned FROM model_workers WHERE model_id = %s",
                                (int(model_id),))
                    return {str(r[0]): dict(r[1] or {}) for r in cur.fetchall()
                            if (not assigned_only or r[2])}
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "reading pair knobs by worker")
            return None

    def write_pair_knobs(self, model_id: int, worker_id: str, to_set: dict, to_unset: list):
        """Write operator knobs on model_workers.user_settings for ONE pair.

        Contract (same as react/testshell/api.py — the rulings live in code):
          * only PAIR_KNOB_KEYS are accepted                → ("bad_knob", [...])
          * UPDATE-only: no row ⇒ the model is not assigned
            to / live on that worker; a knob there is residue → ("unassigned", None)
          * moe=true only where the selected quant's verdict
            says moe_offered                                 → ("moe_not_offered", why)
          * the WorkerStore is never touched.
        Returns ("ok", user_settings) on success, ("unavailable", err) when the
        DB is off or faulted.
        """
        if not enabled():
            return "unavailable", "model database is not enabled"
        to_set = dict(to_set or {})
        to_unset = [str(k) for k in (to_unset or [])]
        bad = [k for k in list(to_set) + to_unset if k not in PAIR_KNOB_KEYS]
        if bad:
            return "bad_knob", bad
        if "tuned_for" in to_set or "tuned_for" in to_unset:
            return "bad_knob", ["tuned_for is server-managed"]
        # KV cache rulings in code: the type must be a known llama.cpp cache
        # type; a quantized cache REQUIRES flash attention, so setting q8_0/q4_0
        # also sets flash_attn=true, and flash_attn=false while a quantized type
        # is (or stays) selected is rejected.
        if "kv_cache_type" in to_set:
            kct = str(to_set["kv_cache_type"] or "").strip().lower()
            if kct not in KV_CACHE_TYPES:
                return "bad_knob", [f"kv_cache_type={to_set['kv_cache_type']!r} (one of {sorted(KV_CACHE_TYPES)})"]
            to_set["kv_cache_type"] = kct
            if kct in KV_CACHE_TYPES_NEED_FLASH:
                to_set["flash_attn"] = True
        if "flash_attn" in to_set:
            if not isinstance(to_set["flash_attn"], bool):
                return "bad_knob", ["flash_attn must be true/false"]
        for _k in ("attention_gpu_layers", "experts_cpu_layers"):
            if _k in to_set:
                try:
                    _v = int(to_set[_k])
                except (TypeError, ValueError):
                    return "bad_knob", [f"{_k} must be a layer count (integer >= 0)"]
                if _v < 0:
                    return "bad_knob", [f"{_k} must be a layer count (integer >= 0)"]
                to_set[_k] = _v
        if "explicit_spill" in to_set:
            sp = str(to_set["explicit_spill"] or "").strip().lower()
            if sp not in EXPLICIT_SPILL_CHOICES:
                return "bad_knob", [f"explicit_spill={to_set['explicit_spill']!r} (one of {list(EXPLICIT_SPILL_CHOICES)})"]
            to_set["explicit_spill"] = sp
        if "quants" in to_set:
            q = to_set["quants"]
            if not isinstance(q, list) or not all(isinstance(x, str) and x.strip() for x in q):
                return "bad_knob", ["quants must be a list of GGUF basenames (preference order)"]
            seen = []
            for x in q:
                b = _os_path_basename(x)
                if b not in seen:
                    seen.append(b)
            if not seen:
                # an empty list = the verdict default: treat as unset (+ the derived pin)
                to_set.pop("quants", None)
                for k in ("quants", "gguf_file"):
                    if k not in to_unset:
                        to_unset.append(k)
            else:
                to_set["quants"] = seen
        if "quants" in to_unset and "gguf_file" not in to_unset and "gguf_file" not in to_set:
            to_unset.append("gguf_file")          # the list's derived pin goes with it
        if "gguf_file" in to_set and "quants" not in to_set and "quants" not in to_unset:
            to_unset.append("quants")             # a single pick replaces the list
        if "ctx_pct" in to_set:
            try:
                cp = int(to_set["ctx_pct"])
            except (TypeError, ValueError):
                return "bad_knob", ["ctx_pct must be an integer 1..100"]
            if not 1 <= cp <= 100:
                return "bad_knob", ["ctx_pct must be an integer 1..100"]
            to_set["ctx_pct"] = cp
        import json as _json
        import os as _os
        try:
            with self.db.lock:
                with self.db.transaction():
                    with self.db.cursor() as cur:
                        cur.execute(PairKnobQueries.FETCH_PAIR, (int(model_id), str(worker_id)))
                        pair = cur.fetchone()
                        if pair is None:
                            return "unassigned", None
                        if "quants" in to_set:
                            # QUANT LIST (operator 2026-10-02): every entry must be a known
                            # GGUF of this model; gguf_file is DERIVED = the first listed
                            # quant whose verdict fits at the pair's ctx target + KV type.
                            # None fits → the head stays (kept and greyed in the UI).
                            knobs_now = dict(pair[0] or {})
                            knobs_now.update(to_set)
                            for k in to_unset:
                                knobs_now.pop(k, None)
                            cur.execute("SELECT file, kv_cost->>'ctx_train' FROM model_quant_facts WHERE model_id=%s AND file ILIKE '%%.gguf'",
                                        (int(model_id),))
                            facts = {r[0]: (int(r[1]) if r[1] else None) for r in cur.fetchall()}
                            unknown = [f for f in to_set["quants"] if f not in facts]
                            if unknown:
                                return "bad_knob", [f"unknown quant(s) for this model: {unknown} (known: {sorted(facts)})"]
                            cur.execute(PairKnobQueries.FETCH_PAIR_VERDICTS, (int(model_id), str(worker_id)))
                            vmem = {r[0]: (r[3] or {}) for r in cur.fetchall()}
                            chosen, _reasons = quant_fit_walk(
                                to_set["quants"], vmem, ctx_pct=knobs_now.get("ctx_pct"), trained_by_file=facts,
                                kv_cache_type=knobs_now.get("kv_cache_type") or "f16",
                                bnb_on=knobs_now.get("bnb_4bit") is True)
                            to_set["gguf_file"] = chosen or to_set["quants"][0]
                            to_unset = [k for k in to_unset if k not in ("gguf_file", "quants")]
                            # placement knobs were tuned for another quant → auto for this one
                            tuned = knobs_now.get("tuned_for")
                            if tuned and tuned != to_set["gguf_file"]:
                                for k in PLACEMENT_KNOB_KEYS:
                                    to_set.pop(k, None)
                                    if k not in to_unset:
                                        to_unset.append(k)
                                if "tuned_for" not in to_unset:
                                    to_unset.append("tuned_for")
                        if any(k in to_set for k in PLACEMENT_KNOB_KEYS):
                            # stamp which quant these placement knobs are tuned for
                            _now = dict(pair[0] or {}); _now.update(to_set)
                            _gf = _now.get("gguf_file") or (pair[1] or "")
                            if _gf:
                                to_set["tuned_for"] = _os_path_basename(str(_gf))
                        elif "gguf_file" in to_set and "quants" not in to_set:
                            # a direct quant pick with knobs tuned for another file → auto
                            _tuned = (pair[0] or {}).get("tuned_for")
                            if _tuned and _tuned != _os_path_basename(str(to_set["gguf_file"])):
                                for k in PLACEMENT_KNOB_KEYS:
                                    if k not in to_unset:
                                        to_unset.append(k)
                                to_unset.append("tuned_for")
                        # ctx_pct ruling in code (2026-10-02): the target may not exceed
                        # the stored verdict's ctx ceiling for the selected mode and KV
                        # cache type (memory[mode].ctx_max[type]) — the same budgets as
                        # the quant verdict. No verdict yet -> accepted (nothing to check).
                        if "ctx_pct" in to_set or "kv_cache_type" in to_set or "bnb_4bit" in to_set or "moe" in to_set or "alloc_mode" in to_set:
                            knobs_now = dict(pair[0] or {})
                            knobs_now.update(to_set)
                            for k in to_unset:
                                knobs_now.pop(k, None)
                            cur.execute(PairKnobQueries.FETCH_PAIR_VERDICTS,
                                        (int(model_id), str(worker_id)))
                            rows = cur.fetchall()
                            mem_by = {r[0]: (r[3] or {}) for r in rows}
                            want = _os.path.basename(str(knobs_now.get("gguf_file") or ""))
                            mem = mem_by.get(want) or mem_by.get(pair[1] or "") or (
                                next(iter(mem_by.values())) if mem_by else {})
                            auto = pair[3] or {}
                            bnb_on = knobs_now.get("bnb_4bit") is True
                            moe_on = knobs_now.get("moe") is True
                            if bnb_on:
                                # the 4-bit verdict is the same shape under memory.bnb_4bit
                                mem = mem.get("bnb_4bit") or {}
                                mode = knobs_now.get("alloc_mode") or (
                                    ((auto.get("bnb_moe_explicit") or {}).get("mode") if moe_on else None)
                                    or (auto.get("bnb_4bit") or {}).get("mode"))
                            else:
                                mode = knobs_now.get("alloc_mode") or (
                                    ((auto.get("moe_explicit") or {}).get("mode") if moe_on else None)
                                    or (auto.get("standard") or {}).get("mode"))
                            kct = str(knobs_now.get("kv_cache_type") or "f16")
                            ktype = "f16" if kct == "bf16" else kct
                            cp = knobs_now.get("ctx_pct")
                            trained = None
                            try:
                                cur.execute("SELECT kv_cost->>'ctx_train' FROM model_quant_facts WHERE model_id=%s AND file=%s",
                                            (int(model_id), want or pair[1] or ""))
                                r = cur.fetchone()
                                trained = int(r[0]) if r and r[0] else None
                            except Exception:  # noqa: BLE001
                                trained = None
                            if mode and cp is not None and trained:
                                want_ctx = max(1024, (trained * int(cp) // 100) // 1024 * 1024)
                                cap = ((mem.get(mode) or {}).get("ctx_max") or {}).get(ktype)
                                if cap is not None and (int(cap) <= 0 or want_ctx > int(cap)):
                                    # AUTO mode is ctx-aware (operator 2026-10-02: MoE off on
                                    # Anko "defaulted to gpu-only and failed rather than max-ram"):
                                    # with no operator alloc_mode pin, fall through the mode
                                    # order to the first verdict mode whose ceiling holds the
                                    # target. Only a pinned mode, or no fitting mode, refuses.
                                    if not knobs_now.get("alloc_mode"):
                                        for alt in MODE_FALLBACK_ORDER:
                                            acap = ((mem.get(alt) or {}).get("ctx_max") or {}).get(ktype)
                                            if alt != mode and acap is not None and int(acap) >= want_ctx and (mem.get(alt) or {}).get("fits"):
                                                mode, cap = alt, acap
                                                break
                                    if int(cap) <= 0 or want_ctx > int(cap):
                                        return "ctx_exceeds", {"mode": mode, "kv_cache_type": kct, "ctx_max": int(cap),
                                                               "requested_ctx": want_ctx, "trained": trained,
                                                               "max_pct": int(int(cap) * 100 // trained)}
                        _PC = ("attention_gpu_layers", "experts_cpu_layers", "explicit_spill")
                        # `moe` rides inside explicit too (operator 2026-10-02 03:30: "experts
                        # may as well be within the explicit component as an optional
                        # setting if available"): flipping it re-derives n_cpu_moe.
                        if any(k in to_set for k in _PC + ("moe",)) or any(k in to_unset for k in _PC + ("moe",)):
                            # PER-CLASS explicit placement in LAYER COUNTS (operator
                            # 2026-10-02): attention_gpu_layers = the LAST a layers whose
                            # attention + KV sit on the GPU (--n-gpu-layers; KV follows the
                            # layer's device, so this boundary dictates where KV spills);
                            # experts_cpu_layers = the FIRST c layers whose experts stay in
                            # RAM (--n-cpu-moe), c >= L-a. Absent, c follows explicit_spill:
                            # prefer-gpu fills the GPU budget left beside the attention
                            # boundary, prefer-ram keeps every expert in RAM. Validated
                            # against the verdict band at the pair's ctx target + cache
                            # type, then alloc_mode=explicit / n_gpu_layers / n_cpu_moe
                            # are derived onto the same row. Rulings in code.
                            knobs_now = dict(pair[0] or {})
                            knobs_now.update(to_set)
                            for k in to_unset:
                                knobs_now.pop(k, None)
                            att = knobs_now.get("attention_gpu_layers")
                            if att is None:
                                if any(k in to_set for k in _PC):
                                    return "bad_knob", ["attention_gpu_layers is required for a per-class split (experts_cpu_layers / explicit_spill ride on it)"]
                                att = False      # the split is being unset: nothing to derive
                        else:
                            att = False
                        if att is not False:
                            cur.execute(PairKnobQueries.FETCH_PAIR_VERDICTS, (int(model_id), str(worker_id)))
                            rows = cur.fetchall()
                            mem_by = {r[0]: (r[3] or {}) for r in rows}
                            want = _os.path.basename(str(knobs_now.get("gguf_file") or ""))
                            mem = mem_by.get(want) or mem_by.get(pair[1] or "") or (next(iter(mem_by.values())) if mem_by else {})
                            band = ((mem.get("explicit") or {}).get("band")) or None
                            if not band:
                                return "explicit_infeasible", {"why": "no per-class band in the verdict (not a MoE GGUF, or no verdict yet)"}
                            L = int(band["layers"])
                            a = int(att)
                            if a > L:
                                return "bad_knob", [f"attention_gpu_layers={a} > {L} layers"]
                            cpu_exp = knobs_now.get("experts_cpu_layers")
                            spill = str(knobs_now.get("explicit_spill") or EXPLICIT_SPILL_CHOICES[0])
                            if knobs_now.get("moe") is False:
                                # expert split OFF inside explicit: experts have no placement
                                # of their own — they follow their layer's attention, i.e.
                                # exactly the L-a layers off the GPU keep their experts in RAM.
                                if cpu_exp is not None and int(cpu_exp) != L - a:
                                    return "bad_knob", [f"experts_cpu_layers={cpu_exp} with moe=false: the expert split is off, experts follow the attention boundary ({L - a} layers in RAM)"]
                                cpu_exp = L - a
                                spill = "off"
                            if cpu_exp is not None:
                                c = int(cpu_exp)
                                if c > L:
                                    return "bad_knob", [f"experts_cpu_layers={c} > {L} layers"]
                                if c < L - a:
                                    return "bad_knob", [f"experts_cpu_layers={c} < {L - a} (layers - attention_gpu_layers): a layer's experts can sit on the GPU only when its attention does"]
                            kct = str(knobs_now.get("kv_cache_type") or "f16")
                            scale = float(KV_CACHE_TYPES.get("f16" if kct == "bf16" else kct, 2.0)) / 2.0
                            trained = None
                            try:
                                cur.execute("SELECT kv_cost->>'ctx_train' FROM model_quant_facts WHERE model_id=%s AND file=%s",
                                            (int(model_id), want or pair[1] or ""))
                                r = cur.fetchone(); trained = int(r[0]) if r and r[0] else None
                            except Exception:  # noqa: BLE001
                                trained = None
                            cp = knobs_now.get("ctx_pct")
                            ctx = (max(1024, (trained * int(cp) // 100) // 1024 * 1024) if (trained and cp is not None)
                                   else int(band.get("kv_ctx") or 0))
                            kv_layer = (float(band["kv_total_bytes"]) / max(1, int(band["kv_ctx"] or 1)) * ctx / L) * scale if band.get("kv_ctx") else 0.0
                            per = [int(x) for x in band["expert_layer_bytes"]]
                            attn = float(band["attn_layer_bytes"])
                            # experts the GPU can still hold beside a attention layers + their KV
                            room = int(band["gpu_budget"]) - int(band["compute_bytes"]) - a * (attn + kv_layer)
                            e_max = -1
                            if room >= 0:
                                e_max = 0
                                for ee in range(a, -1, -1):
                                    if sum(per[L - ee:]) <= room:
                                        e_max = ee; break
                            if cpu_exp is not None:
                                e = L - int(cpu_exp)
                            else:
                                e = max(0, e_max) if spill == "prefer-gpu" else 0
                            gpu_need = a * (attn + kv_layer) + sum(per[L - e:]) + int(band["compute_bytes"])
                            ram_need = (L - a) * (attn + kv_layer) + sum(per[:L - e])
                            if gpu_need > int(band["gpu_budget"]) or ram_need > int(band["ram_budget"]):
                                return "explicit_infeasible", {"attention_gpu_layers": a, "experts_cpu_layers": L - e,
                                                               "explicit_spill": spill, "experts_cpu_layers_given": cpu_exp is not None,
                                                               "a": a, "e": e, "layers": L, "ctx": ctx, "kv_cache_type": kct,
                                                               "gpu_need": int(gpu_need), "gpu_budget": int(band["gpu_budget"]),
                                                               "ram_need": int(ram_need), "ram_budget": int(band["ram_budget"]),
                                                               "experts_cpu_layers_min": (L - e_max) if e_max >= 0 else None,
                                                               "attention_gpu_layers_min": int(band.get("a_min") or 0)}
                            to_set["alloc_mode"] = "explicit"
                            to_set["n_gpu_layers"] = -1 if a >= L else a
                            to_set["n_cpu_moe"] = 999 if e <= 0 else (L - e)
                        if to_set.get("flash_attn") is False:
                            knobs_now = dict(pair[0] or {})
                            knobs_now.update(to_set)
                            for k in to_unset:
                                knobs_now.pop(k, None)
                            if str(knobs_now.get("kv_cache_type") or "f16") in KV_CACHE_TYPES_NEED_FLASH:
                                return "bad_knob", [f"flash_attn=false is not allowed with kv_cache_type={knobs_now['kv_cache_type']} (quantized KV cache needs flash attention)"]
                        if to_set.get("moe") is True:
                            knobs_now = dict(pair[0] or {})
                            knobs_now.update(to_set)
                            for k in to_unset:
                                knobs_now.pop(k, None)
                            cur.execute(PairKnobQueries.FETCH_PAIR_VERDICTS,
                                        (int(model_id), str(worker_id)))
                            by = {r[0]: (bool(r[1]), r[2] or {}) for r in cur.fetchall()}
                            want = _os.path.basename(str(knobs_now.get("gguf_file") or ""))
                            sel = by.get(want) or by.get(pair[1] or "") or (
                                next(iter(by.values())) if by else None)
                            if sel is None:
                                return "moe_not_offered", "no verdict computed yet"
                            if not sel[0]:
                                return "moe_not_offered", (sel[1].get("why") if isinstance(sel[1], dict) else None) or "not offered"
                        cur.execute(PairKnobQueries.UPDATE_PAIR_KNOBS,
                                    (_json.dumps(to_set), to_unset, int(model_id), str(worker_id)))
                        row = cur.fetchone()
                        if row is None:
                            return "unassigned", None
                        return "ok", row[0] or {}
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "writing pair knobs")
            return "unavailable", f"{type(exc).__name__}: {exc}"

    def set_pair_assigned(self, model_id: int, worker_id: str, assigned: bool,
                          source: str = "operator", actor: str = "operator"):
        """DB-owned designation write for ONE pair (operator ruling 2026-10-02:
        the DB, not central's JSON, is the designation of record; central
        converges to it in the background — see hugpy_server designation_relay).
          assigned=True  -> upsert the pair row with assigned=true; `source` is
                            stamped on a NEW row only (provenance never retagged)
          assigned=False -> assigned=false AND pinned=false; the row is DELETED
                            unless the pair is live (loaded/loading/allocated).
        PINS (ruling 2026-10-02 "respect the pins"): an unassign by any actor
        other than "operator" is REFUSED while the pair is pinned →
        ("pinned", {...}). The operator's own unassign clears the pin (operator
        intent wins over the earlier operator intent).
        Returns ("ok", {"assigned", "row"}), ("pinned", detail) or ("unavailable", err)."""
        if not enabled():
            return "unavailable", "model database is not enabled"
        try:
            with self.db.lock:
                with self.db.transaction():
                    with self.db.cursor() as cur:
                        if assigned:
                            cur.execute(PairKnobQueries.INSERT_PAIR_ASSIGNED,
                                        (int(model_id), str(worker_id), (str(source or "").strip().lower() or None)))
                            cur.fetchone()
                            return "ok", {"assigned": True, "row": "kept"}
                        if str(actor or "operator") != "operator":
                            cur.execute(PairKnobQueries.FETCH_PAIR_PIN, (int(model_id), str(worker_id)))
                            pin = cur.fetchone()
                            if pin is not None and bool(pin[1]):
                                return "pinned", {"assigned": bool(pin[0]), "pinned": True, "source": pin[2],
                                                  "actor": actor, "reason": "pair is pinned by the operator; automation may not unassign it"}
                        cur.execute(PairKnobQueries.SET_PAIR_ASSIGNED, (False, False, False, False, int(model_id), str(worker_id)))
                        row = cur.fetchone()
                        if row is None:
                            return "ok", {"assigned": False, "row": "absent"}
                        # the row stays (assigned=false) until central no longer lists
                        # the pair; the heartbeat trigger drops it then (no residue)
                        return "ok", {"assigned": False, "row": "kept until central converges"}
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "writing pair designation")
            return "unavailable", f"{type(exc).__name__}: {exc}"

    # ── drive presence (detach step 2, 2026-10-02) ──────────────────────
    _presence_digest: dict = {}

    def record_worker_presence(self, worker_id: str, rows: list, source: str = "storage", complete: bool = True):
        """Snapshot what a worker reports ON ITS DRIVE into model_worker_presence.
        rows: [{model_key, bytes?, store?, protected?, detail?}] (storage survey)
        or [{model_key, name, hub_id, framework, filename, external_location /
        worker_location, api_url, …}] (discovered report; everything but
        model_key/bytes lands in detail). complete=True retires rows of the SAME
        source the report no longer lists. Change-driven: an identical report
        (per worker+source) is a no-op. Returns ("ok", {"rows", "changed"}),
        ("unavailable", err)."""
        if not enabled():
            return "unavailable", "model database is not enabled"
        import hashlib as _hashlib, json as _json
        clean = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            key = str(r.get("model_key") or r.get("key") or "").strip()
            if not key:
                continue
            detail = {k: v for k, v in r.items()
                      if k not in ("model_key", "key", "bytes", "size_bytes", "store", "protected") and v is not None
                      and k not in ("assigned", "pinned", "loaded", "loading", "provisioning", "granted",
                                    "counts_toward_budget", "last_picked", "tok_n", "tok_s_avg", "tok_s_last", "why")}
            b = r.get("bytes", r.get("size_bytes"))
            clean.append((key, int(b) if isinstance(b, (int, float)) and b >= 0 else None,
                          (str(r["store"]) if r.get("store") is not None else None),
                          (bool(r["protected"]) if r.get("protected") is not None else None),
                          _json.dumps(detail, sort_keys=True, default=str)))
        digest = _hashlib.sha1(_json.dumps(clean, sort_keys=True).encode()).hexdigest()
        dk = (str(worker_id), str(source))
        if self._presence_digest.get(dk) == digest:
            return "ok", {"rows": len(clean), "changed": False}
        try:
            with self.db.lock:
                with self.db.transaction():
                    with self.db.cursor() as cur:
                        for key, b, store, protected, detail in clean:
                            cur.execute(PresenceQueries.UPSERT,
                                        (str(worker_id), key, name_forms(key), key, str(source), b, store, protected, detail))
                        if complete:
                            if source == "storage":
                                cur.execute(PresenceQueries.DELETE_STALE, (str(worker_id), [k for k, *_ in clean]))
                            else:
                                cur.execute(PresenceQueries.DELETE_STALE + " AND source = %s",
                                            (str(worker_id), [k for k, *_ in clean], str(source)))
            self._presence_digest[dk] = digest
            return "ok", {"rows": len(clean), "changed": True}
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "writing worker presence")
            return "unavailable", f"{type(exc).__name__}: {exc}"

    def fetch_worker_presence(self, worker_id: str):
        """[{model_key, model_id, source, bytes, store, protected, detail, first_seen, seen_at}] or None."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(PresenceQueries.FETCH_BY_WORKER, (str(worker_id),))
                    return [{"model_key": r[0], "model_id": r[1], "source": r[2], "bytes": r[3], "store": r[4],
                             "protected": r[5], "detail": r[6] or {},
                             "first_seen": r[7].isoformat() if r[7] else None,
                             "seen_at": r[8].isoformat() if r[8] else None} for r in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "reading worker presence")
            return None

    def fetch_allocation_candidates(self, worker_id: str):
        """Present on the worker's drive but NOT an assigned pair: [{model_key,
        model_id, source, bytes, store, protected, detail, fits}] (fits None =
        no verdict yet). None when the DB is off."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(PresenceQueries.FETCH_CANDIDATES, (str(worker_id),))
                    return [{"model_key": r[0], "model_id": r[1], "source": r[2], "bytes": r[3], "store": r[4],
                             "protected": r[5], "detail": r[6] or {}, "fits": r[7]} for r in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "reading allocation candidates")
            return None

    def fetch_presence_catalog(self):
        """The discovered-models catalog in worker_model_catalog.json's shape —
        {worker_id: {"at": epoch, "models": {key: {…detail, model_key}}}} — so
        models_config._apply_worker_catalog can overlay worker-only models from
        the DB. None when the DB is off."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(PresenceQueries.FETCH_DISCOVERED)
                    out: dict = {}
                    for wid, key, detail, seen in cur.fetchall():
                        ent = out.setdefault(str(wid), {"at": 0.0, "models": {}})
                        row = dict(detail or {}); row["model_key"] = key
                        ent["models"][key] = row
                        ts = seen.timestamp() if seen else 0.0
                        if ts > ent["at"]:
                            ent["at"] = ts
                    return out
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "reading presence catalog")
            return None

    def set_pair_pinned(self, model_id: int, worker_id: str, pinned: bool, by: str = "operator"):
        """Operator LOCK on ONE pair (ruling 2026-10-02 06:05). Rides an ASSIGNED
        row only — pinning an unassigned pair is refused ("unassigned") because a
        pin is not a third state of assigned. pinned_by / pinned_at are their own
        columns (activity is heartbeat-owned and rewritten every beat). An unpin
        records who unpinned in pinned_by with pinned_at NULL.
        Returns ("ok", {"pinned","source","pinned_by","pinned_at"}), ("unassigned",
        None) or ("unavailable", err)."""
        if not enabled():
            return "unavailable", "model database is not enabled"
        try:
            with self.db.lock:
                with self.db.transaction():
                    with self.db.cursor() as cur:
                        cur.execute(PairKnobQueries.SET_PAIR_PINNED,
                                    (bool(pinned), str(by or "operator"), bool(pinned), bool(pinned),
                                     int(model_id), str(worker_id)))
                        row = cur.fetchone()
                        if row is None:
                            return "unassigned", None
                        return "ok", {"pinned": bool(row[0]), "source": row[1], "pinned_by": row[2],
                                      "pinned_at": (row[3].isoformat() if row[3] is not None else None)}
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "writing pair pin")
            return "unavailable", f"{type(exc).__name__}: {exc}"

    def fetch_pairs_by_worker(self):
        """{worker_id: {model name: {"assigned", "pinned", "source"}}} for every
        pair row, or None when the DB is off/faulted. Same absence semantics as
        fetch_assigned_by_worker (no entry = nothing known)."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(PairKnobQueries.FETCH_PAIRS_BY_WORKER)
                    out: dict = {}
                    for wid, name, flag, pinned, source, pinned_by in cur.fetchall():
                        out.setdefault(str(wid), {})[str(name)] = {
                            "assigned": bool(flag), "pinned": bool(pinned), "source": source, "pinned_by": pinned_by}
                    return out
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "reading pair designations")
            return None

    def fetch_assigned_by_worker(self):
        """{worker_id: {model name: assigned bool}} for every pair row, or None
        when the DB is off/faulted. A worker absent from the map has NO rows
        (nothing known) — callers must not treat that as "nothing assigned"."""
        if not enabled():
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(PairKnobQueries.FETCH_ASSIGNED_BY_WORKER)
                    out: dict = {}
                    for wid, name, flag in cur.fetchall():
                        out.setdefault(str(wid), {})[str(name)] = bool(flag)
                    return out
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "reading pair designations")
            return None

    def fetch_worker_settings(self, worker_id: str, model_names: list[str]):
        """Read settings for selected model/worker links from the canonical DB."""
        if not enabled() or not worker_id or not model_names:
            return None
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    cur.execute(ModelQueries.FETCH_WORKER_SETTINGS,
                                (str(worker_id), list(dict.fromkeys(map(str, model_names)))))
                    return {str(name): settings or {} for name, settings in cur.fetchall()}
        except Exception as exc:  # noqa: BLE001 — live registry remains fallback
            self.db.mark_unavailable(exc, "reading worker model settings")
            return None

    # ── metrics card + call ledger ────────────────────────────────────────
    def record_metric(self, model_name: str, worker: str, *, quant: str = "",
                      alloc_mode: str = "", tok_per_s=None, cold_load_s=None,
                      hot_load_s=None, upload_time_s=None, media_bytes=None,
                      temperature=None, task=None) -> bool:
        if not enabled() or not model_name or not worker:
            return False
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    self.metrics.upsert_sample(
                        cur, model_name=model_name, quant=quant,
                        alloc_mode=alloc_mode, worker=worker,
                        cold_load_s=cold_load_s, hot_load_s=hot_load_s,
                        tok_per_s=tok_per_s, upload_time_s=upload_time_s,
                        media_bytes=media_bytes, temperature=temperature,
                        task=task)
            return True
        except Exception as exc:  # noqa: BLE001 — never fail a serve
            self.db.mark_unavailable(exc, "recording a model metric")
            return False

    def record_grade(self, model_name: str, grade, *, suite=None,
                     detail=None, quant: str = "", worker: str = "") -> bool:
        """Persist an APTITUDE grade (0..100) for ``model_name`` onto its
        metrics rows (t147) — every alias form, every quant/alloc/worker row
        (or just ``quant`` when given); a stub row when it was never served.
        ``detail`` (dict/list -> JSON text) is the suite's per-task record.
        Fail-open like every write here: False on off/invalid/unreachable."""
        if not enabled() or not model_name:
            return False
        try:
            g = float(grade)
        except (TypeError, ValueError):
            return False
        if not (0.0 <= g <= 100.0):
            return False
        if detail is not None and not isinstance(detail, str):
            import json
            try:
                detail = json.dumps(detail, default=str)
            except Exception:  # noqa: BLE001
                detail = None
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    self.metrics.upsert_grade(
                        cur, name_forms=name_forms(model_name), grade=g,
                        suite=(str(suite) if suite else None),
                        detail=detail, quant=quant or "", worker=worker or "")
            return True
        except Exception as exc:  # noqa: BLE001 — never fail a grader
            self.db.mark_unavailable(exc, "recording a model grade")
            return False

    def record_cold_load(self, model_name: str, worker: str, cold_load_s,
                         *, quant: str = "") -> bool:
        """A measured COLD load (seconds, from the worker's calibration
        wire) onto ``model_name``'s rows on ``worker`` — the cold_load_s
        column the serve seam leaves NULL. Fail-open."""
        if not enabled() or not model_name or not worker:
            return False
        try:
            c = float(cold_load_s)
        except (TypeError, ValueError):
            return False
        if not (c > 0.0):
            return False
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    self.metrics.record_cold_load(
                        cur, name_forms=name_forms(model_name),
                        worker=str(worker), cold_load_s=c, quant=quant or "")
            return True
        except Exception as exc:  # noqa: BLE001 — never fail a heartbeat
            self.db.mark_unavailable(exc, "recording a cold load")
            return False

    def record_call(self, model_name: str, worker: str, *, quant: str = "",
                    alloc_mode: str = "", tok_per_s=None, prompt_tokens=None,
                    completion_tokens=None, elapsed_s=None, task=None,
                    request_id=None, state=None, prompt_s=None,
                    generation_s=None, gen_tokens=None) -> bool:
        """Append ONE call to model_calls. ``tok_per_s`` must be
        gen_tokens / generation_s of this call (see CallQueries.MIGRATIONS for
        the generation split); None when the call had no generation window."""
        if not enabled() or not model_name or not worker:
            return False
        try:
            with self.db.lock:
                with self.db.transaction(), self.db.cursor() as cur:
                    self.start(cur)
                    self.calls.insert_call(
                        cur, model_name=model_name, worker=worker,
                        quant=quant, alloc_mode=alloc_mode,
                        tok_per_s=tok_per_s, prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        elapsed_s=elapsed_s, task=task,
                        request_id=request_id, state=state,
                        prompt_s=prompt_s, generation_s=generation_s,
                        gen_tokens=gen_tokens)
            return True
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "recording a call")
            return False

    def fetch_model_metrics(self, model_name: str) -> list:
        if not enabled() or not model_name:
            return []
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    self.start(cur)
                    return self.metrics.fetch_by_model(
                        cur, name_forms(model_name))
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "fetching model metrics")
            return []

    def fetch_model_calls(self, model_name: str, limit: int = 200) -> list:
        if not enabled() or not model_name:
            return []
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    self.start(cur)
                    return self.calls.fetch_by_model(
                        cur, name_forms(model_name),
                        max(1, min(int(limit or 200), 2000)))
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "fetching model calls")
            return []

    def fetch_call_stats(self, model_name: str = "") -> list:
        """Throughput stats (Σtokens/Σgeneration-seconds over all recorded
        calls, n, p50/p90, min/max, first/last) per (model, worker, quant,
        alloc) cell plus a per-model rollup (worker None) nesting by_worker and
        the explicit ``unstamped`` bucket — see CallsRepository.call_stats.
        ``model_name`` '' = every model."""
        if not enabled():
            return []
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    self.start(cur)
                    return self.calls.call_stats(cur, name_forms(model_name) if model_name else None)
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "fetching call stats")
            return []

    def fetch_worker_averages(self, model_name: str) -> list:
        """Per-worker throughput of ``model_name`` over EVERY recorded call on
        that worker: Σ tokens-in-window / Σ generation seconds (``mean_tok_s``;
        ``avg_tok_s`` kept as its alias for older readers — it used to be the
        mean of per-call rates, which the metrics contract forbids), with
        n_calls / n_rated, spread, first/last and the explicit ``unstamped``
        bucket (calls without a quant/alloc stamp, included in the totals).
        Same numbers as call_stats' by_worker — one server-side source."""
        if not enabled() or not model_name:
            return []
        try:
            with self.db.lock:
                with self.db.cursor() as cur:
                    self.start(cur)
                    stats = self.calls.call_stats(cur, name_forms(model_name))
            out = []
            for st in stats:
                if st.get("level") != "model":
                    continue
                for w, ws in (st.get("by_worker") or {}).items():
                    out.append({**ws, "worker": w, "avg_tok_s": ws.get("mean_tok_s"),
                                "last_ts": ws.get("last_at")})
            out.sort(key=lambda r: -(r.get("n_calls") or 0))
            return out
        except Exception as exc:  # noqa: BLE001
            self.db.mark_unavailable(exc, "fetching worker averages")
            return []

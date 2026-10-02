-- model_full: ONE row per model. Spec and operator intent on the left,
-- heartbeat-driven mechanics isolated in `live` on the right.
--   weights            immutable weights facts, computed once per model file (compute.py)
--   workers[].knobs    operator intent per worker (user_settings) — the knobs
--   workers[].plan     pair summary (fits any quant, default variant); workers[].verdicts = one row per quant (model_worker_quants)
--   quants[]           model_quant_facts: per-file size / MoE structure / KV geometry (computed once)
--   workers[].pinned   what central currently has persisted for the pair (allocation)
--   live[]             loaded/loading/last_picked/worker status — the only heartbeat-driven part
CREATE OR REPLACE VIEW model_full AS
SELECT m.id, m.name, m.hub_id, m.framework,
       m.attributes        AS spec,
       m.weights           AS weights,
       m.serving_settings  AS serving,
       COALESCE((SELECT jsonb_agg(jsonb_build_object('file', f.file, 'quant', f.quant, 'path', f.path, 'bytes', f.size_bytes,
                    'shards', (SELECT q.shards FROM model_quants q WHERE q.model_id = m.id AND q.file = f.path OR (q.model_id = m.id AND q.file = f.file) LIMIT 1),
                    'is_moe', f.is_moe, 'expert_bytes', f.expert_bytes, 'non_expert_bytes', f.non_expert_bytes,
                    'expert_count', f.expert_count, 'expert_used_count', f.expert_used_count,
                    'kv_geo', f.kv_geo, 'kv_cost', f.kv_cost, 'bnb_4bit', f.bnb_4bit, 'error', f.error, 'computed_at', f.computed_at) ORDER BY f.file)
                 FROM model_quant_facts f WHERE f.model_id = m.id), '[]'::jsonb) AS quants,
       COALESCE((SELECT jsonb_agg(jsonb_build_object(
                    'worker_id', w.worker_id,
                    'worker', COALESCE(r.payload->>'name', w.worker_id),
                    'rank', w.allocation_rank,
                    'assigned', w.assigned, 'fits', w.fits,
                    'knobs', w.user_settings,
                    'pinned', w.allocation,
                    'plan', w.plan,
                    'verdicts', COALESCE((SELECT jsonb_agg(jsonb_build_object('file', v.file, 'fits', v.fits, 'modes', v.modes,
                                    'memory', v.memory, 'auto', v.auto, 'moe_offered', v.moe_offered, 'moe', v.moe,
                                    'budget_rev', v.budget_rev, 'kept', v.kept, 'computed_at', v.computed_at) ORDER BY v.file)
                                  FROM model_worker_quants v WHERE v.model_id = w.model_id AND v.worker_id = w.worker_id), '[]'::jsonb))
                  ORDER BY w.allocation_rank NULLS LAST, w.worker_id)
                 FROM model_workers w
                 LEFT JOIN hugpy_worker_registry r ON r.worker_id = w.worker_id OR r.payload->>'name' = w.worker_id
                 WHERE w.model_id = m.id), '[]'::jsonb) AS workers,
       COALESCE((SELECT jsonb_agg(jsonb_build_object(
                    'worker_id', w.worker_id,
                    'worker', COALESCE(r.payload->>'name', w.worker_id),
                    'loaded', w.activity->'loaded', 'loading', w.activity->'loading',
                    'last_picked', w.activity->'last_picked',
                    'worker_status', lv.e->>'status', 'worker_last_seen', lv.e->'last_seen',
                    'updated_at', w.updated_at)
                  ORDER BY w.worker_id)
                 FROM model_workers w
                 LEFT JOIN hugpy_worker_registry r ON r.worker_id = w.worker_id OR r.payload->>'name' = w.worker_id
                 LEFT JOIN LATERAL (SELECT e FROM hugpy_feed f CROSS JOIN LATERAL jsonb_array_elements(f.payload) e
                                    WHERE f.feed = 'liveness' AND e->>'id' = r.worker_id LIMIT 1) lv ON true
                 WHERE w.model_id = m.id), '[]'::jsonb) AS live,
       m.updated_at,
       m.hub                AS hub
FROM models m;
COMMENT ON VIEW model_full IS 'one row per model: spec/weights/serving/quants/workers(knobs+plan, intent) + live(mechanics). Built for react/testshell 2026-10-01.';

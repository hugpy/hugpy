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
       m.hub                AS hub,
       -- 2026-10-02: the model's calls (7 days; failures included since the
       -- call-log end hook) and its measured metrics + grades, per worker/quant.
       (SELECT jsonb_build_object(
                  'days', 7, 'total', count(*),
                  'ok', count(*) FILTER (WHERE COALESCE(c.state->'outcome'->>'ok', 'true') = 'true'),
                  'failed', count(*) FILTER (WHERE c.state->'outcome'->>'ok' = 'false'
                                             AND COALESCE(c.state->'outcome'->>'status', '') <> 'cancelled'),
                  'cancelled', count(*) FILTER (WHERE c.state->'outcome'->>'status' = 'cancelled'),
                  'median_tok_s', round((percentile_cont(0.5) WITHIN GROUP (ORDER BY c.tok_per_s))::numeric, 1),
                  'last_call', max(c.ts),
                  'top_errors', (SELECT COALESCE(jsonb_agg(jsonb_build_object('error', e.err, 'n', e.n) ORDER BY e.n DESC), '[]'::jsonb)
                                   FROM (SELECT left(c2.state->'outcome'->>'error', 120) AS err, count(*) AS n
                                           FROM model_calls c2
                                          WHERE c2.model_id = m.id AND c2.ts > now() - interval '7 days'
                                            AND c2.state->'outcome'->>'ok' = 'false'
                                          GROUP BY 1 ORDER BY 2 DESC LIMIT 3) e))
          FROM model_calls c WHERE c.model_id = m.id AND c.ts > now() - interval '7 days') AS calls,
       COALESCE((SELECT jsonb_agg(jsonb_build_object(
                    'worker', x.worker, 'quant', x.quant, 'alloc_mode', x.alloc_mode, 'task', x.task,
                    'tok_per_s', x.tok_per_s, 'tok_per_s_avg', x.tok_per_s_avg, 'n', x.n_samples,
                    'cold_load_s', x.cold_load_s, 'hot_load_s', x.hot_load_s, 'load_s', x.load_s,
                    'grade', x.grade, 'grade_suite', x.grade_suite, 'graded_at', x.graded_at,
                    'updated_at', x.updated_at) ORDER BY x.worker, x.quant, x.alloc_mode)
                 FROM model_metrics x WHERE x.model_id = m.id), '[]'::jsonb) AS metrics
FROM models m;
COMMENT ON VIEW model_full IS 'one row per model: spec/weights/serving/quants/workers(knobs+plan, intent) + live(mechanics) + calls(7d) + metrics(per worker/quant, grades). Built for react/testshell 2026-10-01.';

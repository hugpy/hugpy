"""QUERY REGISTRY — every SQL statement the model-index owns, named and in one
place (operator ruling 2026-09-10: "all db creation and calls should be
explicit", modeled on solcatcher's src/db/repositories/repos/*/query-registry).

No repository or service may inline SQL; they reference these constants. One
registry per repository, CREATE_TABLE + CREATE_INDEXES first, then the named
operations that table supports.
"""
from __future__ import annotations


class DiscoveryQueries:
    """discovery_models — the walk's report, one row per model."""

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS discovery_models (
        name        TEXT PRIMARY KEY,
        row         JSONB NOT NULL,
        hub_id      TEXT,
        framework   TEXT,
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    """

    CREATE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS discovery_models_hub_idx "
        "ON discovery_models (hub_id);",
    )

    DELETE_ABSENT = "DELETE FROM discovery_models WHERE NOT (name = ANY(%s))"

    UPSERT_ROW = """
    INSERT INTO discovery_models (name, row, hub_id, framework, updated_at)
    VALUES (%s, %s::jsonb, %s, %s, now())
    ON CONFLICT (name) DO UPDATE
    SET row = EXCLUDED.row, hub_id = EXCLUDED.hub_id,
        framework = EXCLUDED.framework, updated_at = now()
    """

    FETCH_ALL = "SELECT name, row FROM discovery_models"


class QuantQueries:
    """model_quants — one row per on-disk .gguf variant of a model."""

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS model_quants (
        model_name  TEXT NOT NULL,
        file        TEXT NOT NULL,
        quant       TEXT,
        bytes       BIGINT,
        shards      INT,
        PRIMARY KEY (model_name, file)
    );
    """

    CREATE_INDEXES = ()

    DELETE_ABSENT_MODELS = \
        "DELETE FROM model_quants WHERE NOT (model_name = ANY(%s))"

    DELETE_FOR_MODEL = "DELETE FROM model_quants WHERE model_name = %s"

    UPSERT_VARIANT = """
    INSERT INTO model_quants (model_name, file, quant, bytes, shards)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (model_name, file) DO UPDATE
    SET quant = EXCLUDED.quant, bytes = EXCLUDED.bytes,
        shards = EXCLUDED.shards
    """


class MetricsQueries:
    """model_metrics — the rollup card (model x quant x alloc_mode x worker),
    modeled on llm_storage/assets/metrics.xlsx_0.ods. tok_per_s is the LAST
    sample, tok_per_s_avg the running mean over n_samples; cold/hot load
    seconds and upload_time_s/media_bytes fill as those paths gain recording —
    absent stays NULL, never 0."""

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS model_metrics (
        model_name    TEXT NOT NULL,
        quant         TEXT NOT NULL DEFAULT '',
        alloc_mode    TEXT NOT NULL DEFAULT '',
        worker        TEXT NOT NULL,
        cold_load_s   DOUBLE PRECISION,
        hot_load_s    DOUBLE PRECISION,
        tok_per_s     DOUBLE PRECISION,
        tok_per_s_avg DOUBLE PRECISION,
        upload_time_s DOUBLE PRECISION,
        media_bytes   BIGINT,
        n_samples     INT NOT NULL DEFAULT 0,
        temperature   DOUBLE PRECISION,
        task          TEXT,
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        grade         DOUBLE PRECISION,
        grade_suite   TEXT,
        grade_detail  TEXT,
        graded_at     TIMESTAMPTZ,
        PRIMARY KEY (model_name, quant, alloc_mode, worker)
    );
    """

    CREATE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS model_metrics_model_idx "
        "ON model_metrics (model_name);",
    )

    # Additive column migrations for tables created before the column existed
    # (CREATE TABLE IF NOT EXISTS never alters). Idempotent; run with the
    # CREATE on every process start. grade (t147, 2026-09-15): the APTITUDE
    # grade 0..100 from an active graded suite (station model_grader /
    # studio-aptitude), grade_suite names the suite, grade_detail is its
    # per-task JSON, graded_at when it was last graded.
    MIGRATIONS = (
        "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS grade DOUBLE PRECISION;",
        "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS grade_suite TEXT;",
        "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS grade_detail TEXT;",
        "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS graded_at TIMESTAMPTZ;",
        # Recorded cold loads (2026-09-23, review_routes._persist_cold_split):
        # the central->worker transfer and load split of a measured cold seat,
        # read back so a benchmark never cold-resets a triple it already timed.
        "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS transfer_s DOUBLE PRECISION;",
        "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS transfer_bytes BIGINT;",
        "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS load_s DOUBLE PRECISION;",
        "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS bytes_per_s DOUBLE PRECISION;",
        "ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS cold_measured_at TIMESTAMPTZ;",
        # INTEGRITY IS NOT APTITUDE (2026-09-23). The integrity audit
        # (hugpy_ops.model_audit --record, suite 'integrity', grade 100/0 +
        # {"verdict": ...} detail) used to be stamped across EVERY serving row
        # of a model by UPDATE_GRADE; a later benchmark result that carried no
        # valid grade then overwrote only grade_suite (COALESCE in
        # review_routes._persist_benchmark_result), leaving an integrity 100
        # labelled as an aptitude suite (e.g. 'hugpy-vision-v1'). Relabel any
        # row whose detail is an integrity verdict. Idempotent.
        "UPDATE model_metrics SET grade_suite = 'integrity'"
        " WHERE grade_suite IS DISTINCT FROM 'integrity'"
        " AND left(grade_detail, 12) = '{\"verdict\": ';",
    )

    # A tok_per_s sample advances the running average and n_samples; every
    # other field overwrites only when supplied (COALESCE keeps prior values).
    UPSERT_SAMPLE = """
    INSERT INTO model_metrics
      (model_name, quant, alloc_mode, worker, cold_load_s, hot_load_s,
       tok_per_s, tok_per_s_avg, upload_time_s, media_bytes, n_samples,
       temperature, task, updated_at)
    VALUES (%(m)s, %(q)s, %(a)s, %(w)s, %(cold)s, %(hot)s, %(tok)s, %(tok)s,
            %(up)s, %(mb)s, CASE WHEN %(tok)s IS NULL THEN 0 ELSE 1 END,
            %(temp)s, %(task)s, now())
    ON CONFLICT (model_name, quant, alloc_mode, worker) DO UPDATE SET
      cold_load_s   = COALESCE(%(cold)s, model_metrics.cold_load_s),
      hot_load_s    = COALESCE(%(hot)s,  model_metrics.hot_load_s),
      upload_time_s = COALESCE(%(up)s,   model_metrics.upload_time_s),
      media_bytes   = COALESCE(%(mb)s,   model_metrics.media_bytes),
      temperature   = COALESCE(%(temp)s, model_metrics.temperature),
      task          = COALESCE(%(task)s, model_metrics.task),
      tok_per_s     = COALESCE(%(tok)s,  model_metrics.tok_per_s),
      tok_per_s_avg = CASE WHEN %(tok)s IS NULL
                           THEN model_metrics.tok_per_s_avg
                           ELSE (COALESCE(model_metrics.tok_per_s_avg, 0)
                                 * model_metrics.n_samples + %(tok)s)
                                / (model_metrics.n_samples + 1) END,
      n_samples     = model_metrics.n_samples
                      + CASE WHEN %(tok)s IS NULL THEN 0 ELSE 1 END,
      updated_at    = now()
    """

    FETCH_BY_MODEL = """
    SELECT quant, alloc_mode, worker, cold_load_s, hot_load_s, tok_per_s,
           tok_per_s_avg, upload_time_s, media_bytes, n_samples, temperature,
           task, extract(epoch from updated_at),
           grade, grade_suite, extract(epoch from graded_at)
    FROM model_metrics WHERE model_name = ANY(%s)
    ORDER BY updated_at DESC
    """

    # A GRADE is a property of the MODEL (one aptitude, however many
    # quant/alloc/worker rows serve it), so it is stamped across every row the
    # model already has — optionally narrowed to one quant — and, when the
    # model has no row yet (graded before it was ever served), seeded as a
    # stub row (n_samples 0, no tok/s) so the grade is visible immediately.
    UPDATE_GRADE = """
    UPDATE model_metrics
       SET grade = %(g)s, grade_suite = %(s)s, grade_detail = %(d)s,
           graded_at = now(), updated_at = now()
     WHERE model_name = ANY(%(names)s)
       AND (%(q)s = '' OR quant = %(q)s)
       AND alloc_mode <> 'integrity'
    """

    # The integrity verdict has ONE row per model, keyed apart from every
    # serving row (alloc_mode 'integrity', quant/worker ''), so it can never
    # overwrite an aptitude grade nor be overwritten by one. Readers select it
    # by grade_suite = 'integrity' exactly as before.
    UPSERT_INTEGRITY = """
    INSERT INTO model_metrics
      (model_name, quant, alloc_mode, worker, n_samples,
       grade, grade_suite, grade_detail, graded_at, updated_at)
    VALUES (%(m)s, '', 'integrity', '', 0, %(g)s, 'integrity', %(d)s, now(), now())
    ON CONFLICT (model_name, quant, alloc_mode, worker) DO UPDATE SET
      grade = EXCLUDED.grade, grade_suite = EXCLUDED.grade_suite,
      grade_detail = EXCLUDED.grade_detail, graded_at = now(),
      updated_at = now()
    """

    INSERT_GRADE_STUB = """
    INSERT INTO model_metrics
      (model_name, quant, alloc_mode, worker, n_samples,
       grade, grade_suite, grade_detail, graded_at, updated_at)
    VALUES (%(m)s, %(q)s, '', %(w)s, 0, %(g)s, %(s)s, %(d)s, now(), now())
    ON CONFLICT (model_name, quant, alloc_mode, worker) DO UPDATE SET
      grade = EXCLUDED.grade, grade_suite = EXCLUDED.grade_suite,
      grade_detail = EXCLUDED.grade_detail, graded_at = now(),
      updated_at = now()
    """

    # COLD LOAD feed (t146/t147): the calibration wire is the single
    # load_seconds producer (comms.model_metrics.record_loads_from_calibration).
    # A measured cold load lands on every row this model has on that worker
    # (the serve rows carry the real alloc_mode; the load sample does not know
    # it), else on a stub row via UPSERT_SAMPLE with alloc_mode ''.
    UPDATE_COLD_LOAD = """
    UPDATE model_metrics
       SET cold_load_s = %(cold)s, updated_at = now()
     WHERE model_name = ANY(%(names)s) AND worker = %(w)s
    """


class CallQueries:
    """model_calls — the append-only per-CALL ledger (operator: "maintains its
    record of every call and its metrics"). Rows are inserted, never updated;
    model_metrics is this stream's rollup. ``state`` is the configuration
    snapshot (co-residents, vram, budget) so a row explains its own number."""

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS model_calls (
        id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
        model_name    TEXT NOT NULL,
        worker        TEXT NOT NULL,
        quant         TEXT NOT NULL DEFAULT '',
        alloc_mode    TEXT NOT NULL DEFAULT '',
        tok_per_s     DOUBLE PRECISION,
        prompt_tokens INT,
        completion_tokens INT,
        elapsed_s     DOUBLE PRECISION,
        task          TEXT,
        request_id    TEXT,
        state         JSONB
    );
    """

    CREATE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS model_calls_model_ts_idx "
        "ON model_calls (model_name, ts DESC);",
        "CREATE INDEX IF NOT EXISTS model_calls_worker_idx "
        "ON model_calls (worker);",
    )

    # THE GENERATION SPLIT (2026-09-23). Additive, idempotent, run with the
    # CREATE on every process start (CREATE TABLE IF NOT EXISTS never alters):
    #   prompt_s     — prompt evaluation seconds (llama-server timings.prompt_ms)
    #   generation_s — the generation window (timings.predicted_ms; else
    #                  elapsed - prompt_s; else elapsed — see gen_basis in state)
    #   gen_tokens   — the tokens decoded INSIDE generation_s. llama-server
    #                  samples the first completion token from the prompt pass
    #                  (its time is in prompt_ms) and starts predicted_ms after
    #                  it, so an engine-timed window holds predicted_n - 1
    #                  tokens; a wall-clock window holds completion_tokens.
    # tok_per_s = gen_tokens / generation_s (NULL when either is <= 0: a
    # 1-token reply has no decode window and no rate).
    MIGRATIONS = (
        "ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS prompt_s DOUBLE PRECISION;",
        "ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS generation_s DOUBLE PRECISION;",
        "ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS gen_tokens INT;",
    )

    INSERT_CALL = """
    INSERT INTO model_calls
      (model_name, worker, quant, alloc_mode, tok_per_s, prompt_tokens,
       completion_tokens, elapsed_s, task, request_id, state,
       prompt_s, generation_s, gen_tokens)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
    """

    FETCH_BY_MODEL = """
    SELECT extract(epoch from ts), worker, quant, alloc_mode, tok_per_s,
           prompt_tokens, completion_tokens, elapsed_s, task, request_id, state,
           prompt_s, generation_s, gen_tokens
    FROM model_calls WHERE model_name = ANY(%s)
    ORDER BY ts DESC LIMIT %s
    """

    # THROUGHPUT = Σ tokens-in-window / Σ generation seconds over EVERY
    # recorded call (never an EMA, never "the last one"), with n, the spread
    # and first/last. Per-row effective window (``gs``) and its tokens (``gt``):
    #   generation_s column (rows written since 2026-09-23)  -> gen_tokens
    #   state.generation_s (benchmark rows: the grader's copy of the reply's
    #     timings, persisted in state)                      -> state.gen_tokens
    #   state.engine_gen_s (relay rows 2026-09-23 before the column existed:
    #     llama-server predicted_ms)                         -> completion - 1
    #   state.request.ttft_s (streamed relay rows: first->last token clock)
    #                                                        -> completion - 1
    #   else the wall clock (elapsed_s)                      -> completion
    # ``basis`` names which one each row used; the stats count rows per basis
    # so a mean never hides what it is made of.
    # MEASURE ONCE: a benchmark call is recorded twice — the relay's row (the
    # serving path's measurement) and the benchmark's own grade row
    # (state.benchmark / caller 'orchestrator'). The benchmark row is the
    # TWIN and is dropped from throughput when its relay row exists: same
    # request_id, or (older rows without one) same model/worker, relay row
    # written within the call's window before it, elapsed within 1 s.
    # UNSTAMPED rows (quant '' or alloc_mode '') are counted at the worker and
    # model level and reported as their own bucket (``u``), never dropped.
    # Levels (GROUPING SETS): cell (model, worker, quant, alloc), worker
    # (model, worker), model, and the unstamped bucket per worker / per model.
    CALL_STATS = """
    WITH b AS (
      SELECT c.id, {model_expr} AS model_name,
             regexp_replace(c.model_name, '^.*~', '') AS tail,
             c.worker, c.quant, c.alloc_mode, c.ts, c.request_id,
             c.completion_tokens AS ct, c.elapsed_s AS el,
             COALESCE(c.state ? 'benchmark' OR c.state->>'caller' = 'orchestrator', false) AS bench,
             c.generation_s AS gen_col, c.gen_tokens AS gt_col,
             CASE WHEN jsonb_typeof(c.state->'generation_s') = 'number'
                  THEN (c.state->>'generation_s')::float8 END AS gen_st,
             CASE WHEN jsonb_typeof(c.state->'gen_tokens') = 'number'
                  THEN (c.state->>'gen_tokens')::float8 END AS gt_st,
             CASE WHEN jsonb_typeof(c.state->'engine_gen_s') = 'number'
                  THEN (c.state->>'engine_gen_s')::float8 END AS eng_s,
             CASE WHEN jsonb_typeof(c.state->'request'->'ttft_s') = 'number'
                  THEN (c.state->'request'->>'ttft_s')::float8 END AS ttft,
             c.state->>'gen_basis' AS basis_st
      FROM model_calls c
      WHERE TRUE {model_filter}
    ), tw AS (
      SELECT DISTINCT x.id FROM b x JOIN b r
        ON r.tail = x.tail AND r.worker = x.worker AND r.id <> x.id
       AND x.bench AND NOT r.bench
       AND ((x.request_id IS NOT NULL AND r.request_id = x.request_id)
         OR (x.request_id IS NULL AND x.el IS NOT NULL AND r.el IS NOT NULL
             AND r.ts BETWEEN x.ts - make_interval(secs => x.el + 180) AND x.ts + interval '2 seconds'
             AND abs(r.el - x.el) < 1.0))
    ), e AS (
      SELECT b.*,
        CASE WHEN gen_col IS NOT NULL THEN gen_col
             WHEN gen_st IS NOT NULL THEN gen_st
             WHEN eng_s IS NOT NULL THEN eng_s
             WHEN ttft IS NOT NULL AND el > ttft THEN el - ttft
             ELSE el END AS gs,
        CASE WHEN gen_col IS NOT NULL THEN gt_col
             WHEN gen_st IS NOT NULL THEN gt_st
             WHEN eng_s IS NOT NULL OR (ttft IS NOT NULL AND el > ttft) THEN ct - 1
             ELSE ct END AS gt,
        COALESCE(basis_st,
                 CASE WHEN gen_col IS NOT NULL OR gen_st IS NOT NULL THEN 'reported'
                      WHEN eng_s IS NOT NULL THEN 'engine'
                      WHEN ttft IS NOT NULL AND el > ttft THEN 'stream'
                      ELSE 'wall' END) AS basis,
        (quant = '' OR alloc_mode = '') AS u,
        (b.id IN (SELECT id FROM tw)) AS twin
      FROM b
    ), d AS (
      SELECT e.*, (NOT twin AND gt > 0 AND gs > 0) AS rated FROM e
    )
    SELECT model_name, worker, quant, alloc_mode, u,
           GROUPING(worker) AS g_worker, GROUPING(quant) AS g_quant, GROUPING(u) AS g_u,
           count(*) FILTER (WHERE NOT twin) AS n_calls,
           count(*) FILTER (WHERE rated) AS n_rated,
           sum(gt) FILTER (WHERE rated) AS tokens,
           sum(gs) FILTER (WHERE rated) AS seconds,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY gt / gs) FILTER (WHERE rated) AS p50,
           percentile_cont(0.9) WITHIN GROUP (ORDER BY gt / gs) FILTER (WHERE rated) AS p90,
           min(gt / gs) FILTER (WHERE rated) AS mn,
           max(gt / gs) FILTER (WHERE rated) AS mx,
           min(extract(epoch from ts)) FILTER (WHERE NOT twin) AS first_at,
           max(extract(epoch from ts)) FILTER (WHERE NOT twin) AS last_at,
           sum(ct) FILTER (WHERE NOT twin AND ct > 0) AS completion_tokens,
           count(*) FILTER (WHERE NOT twin AND (ct IS NULL OR ct <= 0)) AS n_no_tokens,
           count(*) FILTER (WHERE NOT twin AND ct > 0 AND NOT rated) AS n_no_window,
           count(*) FILTER (WHERE rated AND basis IN ('engine', 'reported')) AS n_engine,
           count(*) FILTER (WHERE rated AND basis = 'stream') AS n_stream,
           count(*) FILTER (WHERE rated AND basis NOT IN ('engine', 'reported', 'stream')) AS n_wall,
           count(*) FILTER (WHERE NOT twin AND bench) AS n_bench,
           count(*) FILTER (WHERE twin) AS n_bench_twins,
           count(*) FILTER (WHERE NOT twin AND quant = '') AS n_no_quant,
           count(*) FILTER (WHERE NOT twin AND alloc_mode = '') AS n_no_alloc
    FROM d
    GROUP BY GROUPING SETS ((model_name, worker, quant, alloc_mode), (model_name, worker),
                            (model_name), (model_name, worker, u), (model_name, u))
    """

    WORKER_AVERAGES_BY_MODEL = """
    SELECT worker, avg(tok_per_s), count(*), max(extract(epoch from ts))
    FROM model_calls
    WHERE model_name = ANY(%s) AND tok_per_s IS NOT NULL
    GROUP BY worker ORDER BY avg(tok_per_s) DESC
    """

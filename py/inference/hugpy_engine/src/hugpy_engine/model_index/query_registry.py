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

    INSERT_CALL = """
    INSERT INTO model_calls
      (model_name, worker, quant, alloc_mode, tok_per_s, prompt_tokens,
       completion_tokens, elapsed_s, task, request_id, state)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
    """

    FETCH_BY_MODEL = """
    SELECT extract(epoch from ts), worker, quant, alloc_mode, tok_per_s,
           prompt_tokens, completion_tokens, elapsed_s, task, request_id
    FROM model_calls WHERE model_name = ANY(%s)
    ORDER BY ts DESC LIMIT %s
    """

    WORKER_AVERAGES_BY_MODEL = """
    SELECT worker, avg(tok_per_s), count(*), max(extract(epoch from ts))
    FROM model_calls
    WHERE model_name = ANY(%s) AND tok_per_s IS NOT NULL
    GROUP BY worker ORDER BY avg(tok_per_s) DESC
    """

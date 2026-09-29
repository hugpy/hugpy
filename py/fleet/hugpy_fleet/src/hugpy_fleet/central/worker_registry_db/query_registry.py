"""SQL for the authoritative central worker registry."""

CREATE_META = """
CREATE TABLE IF NOT EXISTS hugpy_worker_registry_meta (
    singleton SMALLINT PRIMARY KEY CHECK (singleton = 1),
    initialized BOOLEAN NOT NULL DEFAULT FALSE
)
"""

CREATE_WORKERS = """
CREATE TABLE IF NOT EXISTS hugpy_worker_registry (
    worker_id TEXT PRIMARY KEY,
    revision BIGINT NOT NULL DEFAULT 1,
    updated_at DOUBLE PRECISION NOT NULL,
    payload JSONB NOT NULL
)
"""

ENSURE_META = """
INSERT INTO hugpy_worker_registry_meta (singleton, initialized)
VALUES (1, FALSE) ON CONFLICT (singleton) DO NOTHING
"""

LOCK_META = "SELECT initialized FROM hugpy_worker_registry_meta WHERE singleton = 1 FOR UPDATE"
SET_INITIALIZED = "UPDATE hugpy_worker_registry_meta SET initialized = TRUE WHERE singleton = 1"
READ_ALL = "SELECT worker_id, payload FROM hugpy_worker_registry"

UPSERT = """
INSERT INTO hugpy_worker_registry (worker_id, revision, updated_at, payload)
VALUES (%s, 1, %s, %s)
ON CONFLICT (worker_id) DO UPDATE SET
    revision = hugpy_worker_registry.revision + 1,
    updated_at = EXCLUDED.updated_at,
    payload = EXCLUDED.payload
"""

DELETE = "DELETE FROM hugpy_worker_registry WHERE worker_id = %s"

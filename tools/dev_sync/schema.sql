-- dev_sync: content-addressed dev-module store for the hugpy fleet.
-- Convention, not invention: Git-style content addressing (also Nix / OCI / IPFS).
-- A package's VERSION is the sha256 of its file-tree manifest (tree_hash), never an
-- assigned counter. Files are deduped blobs; a package state is an immutable snapshot.
-- Additive & idempotent; lives in the dedicated `toolserver` Postgres.

-- Deduplicated file contents, addressed by content hash.
CREATE TABLE IF NOT EXISTS dev_blobs (
    hash       text PRIMARY KEY,          -- sha256 hex of `content`
    content    bytea NOT NULL,
    size       integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Immutable package snapshots. id = chronological cursor for the sync bus;
-- tree_hash = the content-addressed version (sha256 over sorted rel_path+blob_hash).
CREATE TABLE IF NOT EXISTS dev_states (
    id           bigserial PRIMARY KEY,
    package      text NOT NULL,
    tree_hash    text NOT NULL,           -- THE version identifier
    manifest     jsonb NOT NULL,          -- { rel_path: blob_hash, ... }
    parent_id    bigint REFERENCES dev_states(id),
    needs_build  boolean NOT NULL DEFAULT false, -- pyproject/native changed vs parent
    verified     boolean NOT NULL DEFAULT false, -- reconstruct + local install passed
    verified_at  timestamptz,
    verify_log   text,
    pypi_version text,                     -- semver mapped onto this tree at release
    author       text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (package, tree_hash)
);
CREATE INDEX IF NOT EXISTS dev_states_pkg_id ON dev_states (package, id);

-- Per-package pointers: HEAD (latest) and the latest verified save-state.
CREATE TABLE IF NOT EXISTS dev_modules (
    package          text PRIMARY KEY,
    rel_root         text NOT NULL,        -- repo-relative pkg root
    src_dir          text NOT NULL,        -- repo-relative src dir
    import_name      text NOT NULL,        -- importable name, e.g. hugpy_engine
    has_native       boolean NOT NULL DEFAULT false,
    head_id          bigint REFERENCES dev_states(id),  -- latest snapshot
    good_id          bigint REFERENCES dev_states(id),  -- latest verified snapshot
    released_version text,                 -- last PyPI semver
    bump_policy      text NOT NULL DEFAULT 'patch',
    updated_at       timestamptz NOT NULL DEFAULT now()
);

-- Per-worker replay position over dev_states.
CREATE TABLE IF NOT EXISTS dev_sync_cursor (
    worker          text PRIMARY KEY,
    last_applied_id bigint NOT NULL DEFAULT 0,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

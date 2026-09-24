#!/usr/bin/env python3
"""Apply tools/dev_sync/schema.sql to the toolserver Postgres (additive, idempotent).

Reuses abstract_toolserver.db._connect() so it picks up the exact same
SOLCATCHER_POSTGRESQL_* env / ~/.env the toolserver uses. Safe to re-run.
"""
import pathlib
import re
import sys

from abstract_toolserver import db


def _statements(sql: str):
    # Strip full-line -- comments, then split on ';'. Safe here: no function
    # bodies or dollar-quoting in schema.sql, so no ';' lives inside a statement.
    no_comments = "\n".join(
        line for line in sql.splitlines() if not line.strip().startswith("--")
    )
    for stmt in no_comments.split(";"):
        stmt = stmt.strip()
        if stmt:
            yield stmt


def main() -> int:
    sql = pathlib.Path(__file__).with_name("schema.sql").read_text()
    stmts = list(_statements(sql))
    with db._connect() as conn:
        with conn.cursor() as cur:
            for stmt in stmts:
                cur.execute(stmt)
        conn.commit()
    with db._connect(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'dev\\_%' ESCAPE '\\' ORDER BY 1"
            )
            tables = [r["table_name"] for r in cur.fetchall()]
    print(f"applied {len(stmts)} statements; dev_* tables now: {tables}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""pkg_drift — report-only: does the running fleet match a pkg_src configuration?

Compares, at the DISTRIBUTION level, what each worker runs against the member
versions of a configuration (default: the latest known-good one). No DB writes,
no fleet changes: it reads ``pkg_src`` and central's HTTP surfaces only.

What a member version deploys as. Wheels come from release-tag checkouts
(build_wheels.py, setuptools-scm), so a version is deployable exactly when its
tree is the tree of some release tag: it is a member of a config imported from
``tag:vX`` (or was first recorded from one). Such a version is "released as" X;
an identical tree carried into a later tag is released as that too (same source,
different wheel version). A ``+mN`` tree that no tag contains has no wheel —
reported as not deployable.

Fleet data (same sources hugpy-drift-check section C reads, via its helpers):
  GET {central}/api/llm/workers/required-version  -> central's pin
  GET {central}/api/llm/workers                   -> rows incl. pkg_version +
                                                     environment_digest.build
  GET {central}/api/llm/workers/constraints.txt   -> the lockstep pins
Heartbeats report ONE distribution (hugpy-fleet: pkg_version / build). The
siblings are installed by the lockstep converge under constraints.txt
(``name==<required>`` for every workspace dist), so their installed version is
inferred from hugpy-fleet's and flagged ``reported: false``.
Offline workers are ``info``, never drift (drift-check's rule).

    pkg_drift.py [--config NAME] [--central URL] [--worker NAME] [--dsn DSN]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.request
from pathlib import Path

import psycopg
from psycopg import sql

SCHEMA = "pkg_src"
PY_ROOT = Path(__file__).resolve().parent.parent          # .../hugpy/py
DEFAULT_DSN = os.environ.get("PKG_SRC_DSN", "dbname=hugpy")
DEFAULT_CENTRAL = "http://127.0.0.1:7002"
REPORTED_DIST = "hugpy-fleet"                              # the one dist a heartbeat carries
OFFLINE = ("offline", "gone", "stale")                     # drift-check's not-in-fleet states

# Reuse drift-check's HTTP + build helpers (installed, else from the workspace tree).
try:
    from hugpy_ops import drift as _drift
except ImportError:
    sys.path.insert(0, str(PY_ROOT / "operations" / "hugpy_ops" / "src"))
    try:
        from hugpy_ops import drift as _drift
    except ImportError:                                    # stdlib fallback, same shape
        _drift = None


def _fetch_json(url: str, timeout: float = 60.0):
    if _drift:
        return _drift.fetch_json(url, _drift.central_token(), timeout=timeout)
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "pkg-drift"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_text(url: str, timeout: float = 30.0) -> str:
    if _drift:
        return _drift.fetch_text(url, _drift.central_token(), timeout=timeout)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _err(exc: BaseException) -> str:
    return _drift._exc(exc) if _drift else f"{type(exc).__name__}: {exc}"


def _build_desc(build: dict | None) -> str | None:
    return _drift._build_desc(build) if (_drift and build) else (build or {}).get("version")


def norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# --------------------------------------------------------------------------- workspace

def lockstep_dists() -> dict[str, str]:
    """{package dir name: dist} for every [[package]] in partition.toml — the set
    released in lockstep and pinned by central's constraints.txt."""
    manifest = tomllib.loads((PY_ROOT / "partition.toml").read_text(encoding="utf-8"))
    by_dist = {norm_dist(p["distribution"]) for p in manifest.get("package", [])}
    out = {}
    for pp in sorted(PY_ROOT.glob("*/*/pyproject.toml")):
        m = re.search(r'^\s*name\s*=\s*"([^"]+)"', pp.read_text(), re.M)
        if m and norm_dist(m.group(1)) in by_dist:
            out[pp.parent.name] = norm_dist(m.group(1))
    return out


def _vkey(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else -1 for x in re.split(r"[.+]", v))


def _strip_v(tag: str) -> str:
    return tag[1:] if tag[:1] in "vV" else tag


# --------------------------------------------------------------------------- DB (read-only)

def pick_config(cur, name: str | None) -> tuple[int, str, bool, str]:
    """(id, name, known_good, source) — explicit name, else latest known-good."""
    s = sql.Identifier(SCHEMA)
    if name:
        cur.execute(sql.SQL("SELECT id, name, known_good, source FROM {s}.configs "
                            "WHERE name = %s").format(s=s), (name,))
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"no configuration {name!r} (see: config ls)")
        return row
    cur.execute(sql.SQL("SELECT id, name, known_good, source FROM {s}.configs WHERE known_good "
                        "ORDER BY good_at DESC NULLS LAST, id DESC LIMIT 1").format(s=s))
    row = cur.fetchone()
    if row is None:
        cur.execute(sql.SQL("SELECT string_agg(name, ', ' ORDER BY id) FROM {s}.configs")
                    .format(s=s))
        have = cur.fetchone()[0] or "none"
        raise SystemExit("no known-good configuration yet (none passed verify + config good); "
                         f"pass config= explicitly — configs: {have}")
    return row


def config_rows(cur, cid: int) -> list[tuple]:
    """(package, version_id, label, source, released_as[]) per member; released_as =
    every release tag whose tree is exactly this version."""
    cur.execute(sql.SQL("""
        SELECT m.package, v.id, v.label, v.source,
               coalesce(array_agg(DISTINCT substr(c2.source, 5))
                        FILTER (WHERE c2.source LIKE 'tag:%%'), '{{}}')
        FROM {s}.config_members m JOIN {s}.versions v ON v.id = m.version_id
        LEFT JOIN {s}.config_members m2 ON m2.version_id = v.id
        LEFT JOIN {s}.configs c2 ON c2.id = m2.config_id
        WHERE m.config_id = %s
        GROUP BY m.package, v.id, v.label, v.source ORDER BY m.package""")
                .format(s=sql.Identifier(SCHEMA)), (cid,))
    out = []
    for pkg, vid, label, source, tags in cur.fetchall():
        rel = {_strip_v(t) for t in tags}
        if source and source.startswith("tag:"):
            rel.add(_strip_v(source[4:]))
        out.append((pkg, vid, label, source, sorted(rel, key=_vkey)))
    return out


# --------------------------------------------------------------------------- the report

def expected_members(cur, cfg_source: str, cid: int, dists: dict[str, str]) -> tuple[list, list]:
    """(deployable lockstep members, members that never reach a worker)."""
    tag_release = _strip_v(cfg_source[4:]) if cfg_source.startswith("tag:") else None
    members, skipped = [], []
    for pkg, vid, label, source, released in config_rows(cur, cid):
        if pkg not in dists:
            skipped.append({"package": pkg, "label": label,
                            "why": "not a lockstep dist (not in partition.toml) — never on workers"})
            continue
        expected = tag_release if tag_release in released else (released[-1] if released else None)
        members.append({"package": pkg, "dist": dists[pkg], "version_id": vid,
                        "expected_label": label, "expected_version": expected,
                        "released_as": released, "deployable": bool(released)})
    return members, skipped


def compare_worker(w: dict, members: list[dict]) -> dict:
    name = w.get("name") or w.get("id") or "?"
    status = str(w.get("status") or "?").lower()
    digest = w.get("environment_digest") if isinstance(w.get("environment_digest"), dict) else {}
    build = digest.get("build") if isinstance(digest.get("build"), dict) else None
    installed = (build or {}).get("version") or w.get("pkg_version")
    online = status not in OFFLINE and not w.get("unreachable")
    notes = []
    if build and w.get("pkg_version") and build.get("version") != w.get("pkg_version"):
        notes.append(f"build {build.get('version')} != pkg_version {w.get('pkg_version')}")
    if build and build.get("dirty"):
        notes.append("build is dirty")
    pkgs = []
    for m in members:
        reported = m["dist"] == REPORTED_DIST
        if not m["deployable"]:
            match = None
        elif installed == m["expected_version"]:
            match = "exact"
        elif installed in m["released_as"]:
            match = "same_source"                      # different wheel version, identical tree
        else:
            match = None
        row = {"package": m["package"], "dist": m["dist"], "expected_label": m["expected_label"],
               "expected_version": m["expected_version"] or "unreleased (no wheel; not deployable)",
               "installed_version": installed, "reported": reported,
               "match": match, "ok": match is not None}
        if reported and build:
            row["build"] = _build_desc(build)
        pkgs.append(row)
    bad = [p for p in pkgs if not p["ok"]]
    undeployable = [m["package"] for m in members if not m["deployable"]]
    if undeployable:
        notes.append(f"config not deployable: {', '.join(undeployable)} have no released wheel")
    if not online:
        verdict = "info"
        notes.insert(0, f"{status}{', unreachable' if w.get('unreachable') else ''}: last known "
                        f"{REPORTED_DIST} {installed or 'none'} — not compared")
    elif not installed:
        verdict = "unknown"
        notes.append("heartbeat carries no pkg_version/build — cannot compare")
    else:
        verdict = "drift" if bad else "in_sync"
    if online and installed and any(not p["reported"] for p in pkgs):
        notes.append(f"only {REPORTED_DIST} is reported; siblings inferred from the "
                     "lockstep converge (constraints.txt pins)")
    return {"worker": name, "id": w.get("id"), "status": status, "online": online,
            "verdict": verdict, "drifted": [p["dist"] for p in bad] if online else [],
            "packages": pkgs, "notes": notes}


def api_drift(conn, config: str | None = None, central: str = DEFAULT_CENTRAL,
              worker: str | None = None) -> dict:
    """Fleet vs configuration ``config`` (default: latest known-good). Read-only."""
    central = (central or DEFAULT_CENTRAL).rstrip("/")
    dists = lockstep_dists()
    with conn.cursor() as cur:
        cid, name, good, cfg_source = pick_config(cur, config)
        members, skipped = expected_members(cur, cfg_source, cid, dists)
    out = {"config": name, "config_known_good": good, "config_source": cfg_source,
           "not_on_workers": skipped, "central": {"url": central}, "workers": [], "notes": []}
    if not good:
        out["notes"].append(f"config {name} is NOT known-good (verify + config good pending)")
    # A release every deployable member is (source-identical to) — what central should pin.
    common = set.intersection(*(set(m["released_as"]) for m in members)) if members else set()
    out["config_release"] = sorted(common)

    c = out["central"]
    try:
        req = _fetch_json(f"{central}/api/llm/workers/required-version")
        c["required_pkg_version"] = req.get("required_pkg_version")
        c["pkg_index_url"] = req.get("pkg_index_url")
        c["matches_config"] = c["required_pkg_version"] in common
    except Exception as exc:  # noqa: BLE001 — unreachable is a finding
        c.update(required_pkg_version=None, matches_config=None, error=_err(exc))
    try:
        pins = [ln.strip() for ln in _fetch_text(f"{central}/api/llm/workers/constraints.txt")
                .splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        want = {m["dist"]: m["expected_version"] for m in members}
        off = [p for p in pins if "==" in p and norm_dist(p.split("==")[0]) in want
               and p.split("==", 1)[1] != want[norm_dist(p.split("==")[0])]]
        c["constraints_pins"] = len(pins)
        c["constraints_mismatch"] = off
    except Exception as exc:  # noqa: BLE001
        c["constraints_error"] = _err(exc)

    try:
        rows = _fetch_json(f"{central}/api/llm/workers")
        if isinstance(rows, dict):
            rows = rows.get("workers") or rows.get("data") or []
    except Exception as exc:  # noqa: BLE001
        out["notes"].append(f"{central}/api/llm/workers unreachable: {_err(exc)}")
        rows = []
    for w in rows if isinstance(rows, list) else []:
        if not isinstance(w, dict):
            continue
        if worker and worker not in (w.get("name"), w.get("id")):
            continue
        out["workers"].append(compare_worker(w, members))
    if worker and not out["workers"]:
        out["notes"].append(f"no worker {worker!r} at central")

    verdicts = [w["verdict"] for w in out["workers"]]
    out["summary"] = {v: verdicts.count(v) for v in ("in_sync", "drift", "info", "unknown")}
    out["summary"]["verdict"] = ("drift" if "drift" in verdicts or c.get("matches_config") is False
                                 else "unknown" if "unknown" in verdicts or not verdicts
                                 or c.get("matches_config") is None else "in_sync")
    return out


def format_report(r: dict) -> str:
    c = r["central"]
    lines = [f"config {r['config']} ({r['config_source']}, "
             f"{'known-good' if r['config_known_good'] else 'NOT known-good'}); "
             f"releases it equals: {', '.join(r['config_release']) or 'none'}",
             f"central {c['url']}: pins {c.get('required_pkg_version')} — "
             f"{'matches' if c.get('matches_config') else 'DOES NOT match' if c.get('matches_config') is False else 'unknown vs'} config"]
    if c.get("constraints_mismatch"):
        lines.append(f"  constraints.txt pins differing from config: {', '.join(c['constraints_mismatch'])}")
    for w in r["workers"]:
        lines.append(f"  {w['worker']:12} {w['status']:8} {w['verdict'].upper()}"
                     + (f"  drifted: {', '.join(w['drifted'])}" if w["drifted"] else ""))
        for p in w["packages"]:
            if not p["ok"] and w["online"]:
                lines.append(f"      {p['dist']:16} want {p['expected_version']} "
                             f"({p['expected_label']}) have {p['installed_version']}"
                             f"{'' if p['reported'] else ' (inferred)'}")
        lines += [f"      - {n}" for n in w["notes"]]
    lines += [f"note: {n}" for n in r["notes"]]
    s = r["summary"]
    lines.append(f"summary: {s['verdict'].upper()}  in_sync={s['in_sync']} drift={s['drift']} "
                 f"info={s['info']} unknown={s['unknown']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="configuration name (default: latest known-good)")
    ap.add_argument("--central", default=DEFAULT_CENTRAL)
    ap.add_argument("--worker")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    with psycopg.connect(a.dsn) as conn:
        r = api_drift(conn, a.config, a.central, a.worker)
    print(json.dumps(r, indent=2) if a.json else format_report(r))
    return {"in_sync": 0, "drift": 1}.get(r["summary"]["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())

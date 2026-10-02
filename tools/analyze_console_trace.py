#!/usr/bin/env python3
"""Analyze a persisted console trace.

Examples:
  python tools/analyze_console_trace.py TRACE_ID
  python tools/analyze_console_trace.py TRACE_ID --db /path/to/hugpy-comms.db --json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys


def analyze(trace_id: str, db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    request = con.execute("SELECT * FROM console_trace_requests WHERE id=?", (trace_id,)).fetchone()
    spans = [dict(r) for r in con.execute(
        "SELECT seq,event,depth,function,file,line,at,duration_ms,error "
        "FROM console_trace_spans WHERE trace_id=? ORDER BY seq", (trace_id,))]
    con.close()
    if request is None:
        raise KeyError(f"trace not found: {trace_id}")
    calls = [s for s in spans if s["event"] == "call"]
    exceptions = [s for s in spans if s["event"] == "exception"]
    return {"request": dict(request), "spans": spans,
            "summary": {"calls": len(calls), "events": len(spans),
                         "exceptions": len(exceptions),
                         "python_ms": sum(float(s["duration_ms"] or 0) for s in spans if s["event"] == "return")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_id")
    ap.add_argument("--db", default=os.environ.get("HUGPY_COMMS_DB"), help="shared console DB")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)
    db = args.db or os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), f"hugpy-comms-{os.getuid()}.db")
    try:
        result = analyze(args.trace_id, db)
    except (OSError, sqlite3.Error, KeyError) as exc:
        print(f"analyze-console-trace: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    req = result["request"]
    print(f"{req['method']} {req['path']} -> {req['status']} in {req['duration_ms'] or 0:.1f} ms")
    print(f"{result['summary']['calls']} calls, {result['summary']['exceptions']} exceptions")
    for span in result["spans"]:
        indent = "  " * min(int(span["depth"]), 30)
        extra = f" ({span['duration_ms']:.2f} ms)" if span["duration_ms"] is not None else ""
        error = f" :: {span['error']}" if span["error"] else ""
        print(f"{indent}{span['event']} {span['function']} [{span['file']}:{span['line']}]" + extra + error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

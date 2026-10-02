#!/usr/bin/env python3
"""Capture the complete API payloads that feed the Hugpy Models/Workers UI.

Read-only. Saves complete response bodies (without selecting model fields) and
per-call status/timing metadata. Calls are isolated: a timeout from one endpoint
does not prevent the remaining endpoints from being captured.

Auth, when needed, is read from HUGPY_OPERATOR_TOKEN (Bearer) or HUGPY_COOKIE
(raw Cookie header value). Nothing credential-like is accepted as a CLI arg.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request


def find_catalog_row(models: list[dict], key: str) -> dict | None:
    """Mirror WorkersPanel/catalogRow.js; return the full matching catalog row."""
    if not key or not models:
        return None

    def key_of(row: dict) -> str | None:
        return row.get("model_key", row.get("key"))

    exact = next((row for row in models if key_of(row) == key), None)
    if exact:
        return exact
    if "~" in key:
        owner, bare = key.split("~", 1)
        hub = next((row for row in models
                    if row.get("hub_id") == f"{owner}/{bare}"), None)
        if hub:
            return hub
        matches = [row for row in models if key_of(row) == bare]
        return matches[0] if len(matches) == 1 else None
    matches = [row for row in models
               if str(key_of(row) or "").endswith("~" + key)
               or str(row.get("hub_id") or "").endswith("/" + key)]
    return matches[0] if len(matches) == 1 else None


def request_body(url: str, timeout: float, headers: dict[str, str],
                 accept: str = "application/json") -> tuple[int, dict, bytes, float]:
    req = urllib.request.Request(url, headers={"Accept": accept, **headers})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            return response.status, dict(response.headers.items()), body, time.monotonic() - started
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return exc.code, dict(exc.headers.items()), body, time.monotonic() - started


def write_capture(out_dir: pathlib.Path, name: str, url: str,
                  timeout: float, headers: dict[str, str],
                  accept: str = "application/json") -> dict:
    """Save one complete HTTP response or a full error description."""
    started = time.monotonic()
    meta = {"name": name, "url": url, "accept": accept, "timeout_s": timeout}
    try:
        status, response_headers, body, elapsed = request_body(
            url, timeout, headers, accept)
        raw_path = out_dir / f"{name}.body"
        raw_path.write_bytes(body)
        meta.update({"http_status": status, "elapsed_s": round(elapsed, 3),
                     "content_type": response_headers.get("Content-Type"),
                     "body_bytes": len(body), "raw_body_file": raw_path.name})
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            meta["json"] = False
        else:
            json_path = out_dir / f"{name}.json"
            json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False,
                                            default=str) + "\n", encoding="utf-8")
            meta.update({"json": True, "pretty_json_file": json_path.name})
        meta["ok"] = 200 <= status < 300
    except Exception as exc:  # capture each call independently, including timeouts
        elapsed = time.monotonic() - started
        error_path = out_dir / f"{name}.error.txt"
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        meta.update({"ok": False, "elapsed_s": round(elapsed, 3),
                     "error": f"{type(exc).__name__}: {exc}",
                     "error_file": error_path.name})
    print(f"{name}: {'HTTP ' + str(meta['http_status']) if 'http_status' in meta else 'FAILED'} "
          f"in {meta['elapsed_s']:.3f}s — "
          f"{meta.get('raw_body_file') or meta.get('error_file')}")
    return meta


def capture_event_snapshot(out_dir: pathlib.Path, url: str, timeout: float,
                           headers: dict[str, str]) -> dict:
    """Capture the initial SSE snapshot event the UI receives on connect."""
    started = time.monotonic()
    meta = {"name": "events_initial_snapshot", "url": url,
            "accept": "text/event-stream", "timeout_s": timeout}
    req = urllib.request.Request(
        url, headers={"Accept": "text/event-stream", "Cache-Control": "no-cache", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            lines: list[bytes] = []
            event_name = ""
            data_lines: list[str] = []
            while True:
                line = response.readline()
                if not line:
                    raise RuntimeError("SSE ended before an initial snapshot event")
                lines.append(line)
                decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if decoded.startswith("event:"):
                    event_name = decoded.partition(":")[2].strip()
                elif decoded.startswith("data:"):
                    data_lines.append(decoded.partition(":")[2].lstrip())
                elif decoded == "" and event_name:
                    if event_name == "snapshot":
                        raw_sse = out_dir / "events_initial_snapshot.sse"
                        raw_sse.write_bytes(b"".join(lines))
                        payload = json.loads("\n".join(data_lines))
                        json_path = out_dir / "events_initial_snapshot.json"
                        json_path.write_text(json.dumps(
                            payload, indent=2, ensure_ascii=False, default=str) + "\n",
                            encoding="utf-8")
                        elapsed = time.monotonic() - started
                        meta.update({"ok": True, "http_status": response.status,
                                     "elapsed_s": round(elapsed, 3),
                                     "content_type": response.headers.get("Content-Type"),
                                     "raw_sse_file": raw_sse.name,
                                     "pretty_json_file": json_path.name,
                                     "body_bytes": sum(map(len, lines))})
                        break
                    event_name, data_lines = "", []
    except Exception as exc:
        elapsed = time.monotonic() - started
        error_path = out_dir / "events_initial_snapshot.error.txt"
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        meta.update({"ok": False, "elapsed_s": round(elapsed, 3),
                     "error": f"{type(exc).__name__}: {exc}",
                     "error_file": error_path.name})
    print(f"events_initial_snapshot: "
          f"{'HTTP ' + str(meta['http_status']) if 'http_status' in meta else 'FAILED'} "
          f"in {meta['elapsed_s']:.3f}s — "
          f"{meta.get('pretty_json_file') or meta.get('error_file')}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get(
        "HUGPY_API_BASE_URL", "http://127.0.0.1:7002"),
        help="Hugpy API origin (default: HUGPY_API_BASE_URL or localhost:7002)")
    parser.add_argument("--out-dir", type=pathlib.Path,
        help="output directory (default: timestamped hugpy-api-capture-* here)")
    parser.add_argument("--worker",
        help="also save the full catalog rows matched to every model designated on this worker")
    parser.add_argument("--timeout", type=float, default=120.0,
        help="timeout per API call in seconds (default: 120; /api/models can be slow)")
    parser.add_argument("--skip-events", action="store_true",
        help="skip the live SSE initial snapshot; capture the JSON endpoints only")
    args = parser.parse_args()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or pathlib.Path(f"hugpy-api-capture-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.base_url.rstrip("/")
    headers: dict[str, str] = {}
    if token := os.environ.get("HUGPY_OPERATOR_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    if cookie := os.environ.get("HUGPY_COOKIE"):
        headers["Cookie"] = cookie

    # These are the actual UI inputs and the serving-context diagnostic: the
    # complete catalog, worker roster, serving rows (ctx_size / ctx_source),
    # snapshot fallback, and initial EventSource snapshot.
    captures = [
        write_capture(out_dir, "models", base + "/api/models", args.timeout, headers),
        write_capture(out_dir, "workers", base + "/api/llm/workers", args.timeout, headers),
        write_capture(out_dir, "serving", base + "/api/llm/serving", args.timeout, headers),
        write_capture(out_dir, "snapshot", base + "/api/llm/snapshot", args.timeout, headers),
    ]
    if not args.skip_events:
        captures.append(capture_event_snapshot(
            out_dir, base + "/api/llm/events", args.timeout, headers))

    if args.worker:
        try:
            workers = json.loads((out_dir / "workers.body").read_bytes())
            models = json.loads((out_dir / "models.body").read_bytes())
            if not isinstance(workers, list) or not isinstance(models, list):
                raise ValueError("worker/catalog response was not a JSON list")
            worker = next((row for row in workers
                           if args.worker in (row.get("id"), row.get("name"))), None)
            if worker is None:
                raise ValueError(f"worker {args.worker!r} not found in captured roster")
            designated = worker.get("models") or []
            joined = {
                "worker_selector": args.worker,
                "worker": worker,
                "designated_models": [
                    {"designation_key": key,
                     "catalog_row": find_catalog_row(models, key)}
                    for key in designated
                ],
            }
            joined_path = out_dir / "worker-models.json"
            joined_path.write_text(json.dumps(
                joined, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8")
            print(f"worker-models: saved full worker row and all {len(designated)} "
                  f"matched catalog rows — {joined_path.name}")
        except Exception as exc:
            (out_dir / "worker-models.error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            print(f"worker-models: FAILED — {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    manifest = {"base_url": base, "created_utc": stamp,
                "captures": captures}
    (out_dir / "calls.json").write_text(json.dumps(
        manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nComplete response captures: {out_dir.resolve()}")
    print("The .body/.sse files preserve raw response bytes; .json files are "
          "pretty-printed complete JSON payloads.")
    return 0 if any(row.get("ok") for row in captures) else 2


if __name__ == "__main__":
    raise SystemExit(main())

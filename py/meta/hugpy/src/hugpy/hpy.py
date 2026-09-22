#!/usr/bin/env python3
"""hpy — lightweight fleet browser + caller (stdlib only, no deps).

Endpoints used (hugpy central):
  GET  /v1/models        -> model catalog (id, task, serveable)
  GET  /llm/workers      -> workers, what each has loaded/local/servable, tasks
  POST /v1/chat/completions

Config via env:
  HUGPY_URL   base URL (default http://127.0.0.1:7002; use http://192.168.1.100:7002 off-box)
  HUGPY_KEY   bearer key (hp_...) — required for /v1 calls, not for listing

Usage:
  hpy                         overview: workers + warm models
  hpy workers                 workers, status, warm/local model counts, tasks
  hpy models [-t TASK] [-w WORKER] [--warm] [--local]
  hpy where MODEL             which workers have MODEL warm / local / servable
  hpy call MODEL "prompt" [-w WORKER] [-s SYSTEM] [-n MAXTOK] [--temp T] [--raw]
                              (prompt may also come from stdin)
  hpy ask "prompt"            emergency inference: warm -> on-disk -> catalog

Installed by the ``hugpy`` meta distribution; also runs straight from the
source tree (``python hpy.py``) because it imports nothing outside the stdlib.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("HUGPY_URL", "http://127.0.0.1:7002").rstrip("/")
KEY  = os.environ.get("HUGPY_KEY", "")
WARM = "●"   # ● loaded/warm
LOC  = "○"   # ○ on disk (no download)


def _get(path):
    req = urllib.request.Request(BASE + path)
    if KEY:
        req.add_header("Authorization", "Bearer " + KEY)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def _post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if KEY:
        req.add_header("Authorization", "Bearer " + KEY)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def catalog():
    """id -> {task, tasks, serveable, blocked, ctx}"""
    out = {}
    for m in _get("/v1/models").get("data", []):
        out[m["id"]] = {
            "task": m.get("task") or "?",
            "tasks": m.get("tasks") or [],
            "serveable": m.get("serveable"),
            "blocked": m.get("blocked"),
            "ctx": m.get("context_length"),
        }
    return out


def workers():
    """list of {name,status,role,unreachable,loaded,local,servable,tasks,alloc}"""
    out = []
    for w in _get("/llm/workers"):
        alloc = {a.get("model_key"): a for a in w.get("allocations", []) if a.get("model_key")}
        out.append({
            "name": w.get("name"),
            "status": w.get("status"),
            "role": w.get("role"),
            "unreachable": w.get("unreachable"),
            "loaded": list(w.get("loaded_models") or []),
            "local": list(w.get("models_local") or []),
            "servable": list(w.get("models") or []),
            "tasks": sorted((w.get("task_capabilities") or {}).keys()),
            "alloc": alloc,
        })
    return out


def cmd_workers(args):
    for w in workers():
        flag = "" if not w["unreachable"] else "  [UNREACHABLE]"
        print(f"\n{w['name']}  ({w['status']}, {w['role']}){flag}")
        warm = w["loaded"]
        print(f"  {WARM} warm ({len(warm)}): " + (", ".join(warm) if warm else "-"))
        print(f"  {LOC} local ({len(w['local'])}): " + (", ".join(w['local'][:12]) + (" …" if len(w['local']) > 12 else "") if w['local'] else "-"))
        print(f"    servable: {len(w['servable'])}   tasks: {', '.join(w['tasks']) or '-'}")


def cmd_models(args):
    cat = catalog()
    ws = workers()
    # model -> sets of worker names
    warm, local, serv = {}, {}, {}
    for w in ws:
        for m in w["loaded"]:
            warm.setdefault(m, []).append(w["name"])
        for m in w["local"]:
            local.setdefault(m, []).append(w["name"])
        for m in w["servable"]:
            serv.setdefault(m, []).append(w["name"])
    ids = set(cat) | set(warm) | set(local) | set(serv)
    rows = []
    for m in sorted(ids, key=str.lower):
        task = cat.get(m, {}).get("task", "?")
        if args.task and args.task.lower() not in task.lower():
            continue
        if args.worker and args.worker not in (set(warm.get(m, [])) | set(local.get(m, [])) | set(serv.get(m, []))):
            continue
        if args.warm and m not in warm:
            continue
        if args.local and m not in local and m not in warm:
            continue
        rows.append((m, task, warm.get(m, []), local.get(m, [])))
    w1 = max([len(r[0]) for r in rows] + [5])
    w2 = max([len(r[1]) for r in rows] + [4])
    print(f"{'MODEL':<{w1}}  {'TASK':<{w2}}  {WARM}warm / {LOC}local (workers)")
    for m, task, wm, lo in rows:
        marks = []
        if wm:
            marks.append(f"{WARM}{','.join(wm)}")
        if lo:
            marks.append(f"{LOC}{','.join(lo)}")
        print(f"{m:<{w1}}  {task:<{w2}}  {'  '.join(marks) or '(catalog / cold-download)'}")
    print(f"\n{len(rows)} models. {WARM}=warm (instant)  {LOC}=on disk (no download)  else cold-download.")


def cmd_where(args):
    m = args.model
    for w in workers():
        tags = []
        if m in w["loaded"]:
            tags.append(WARM + "warm")
        if m in w["local"]:
            tags.append(LOC + "local")
        if m in w["servable"]:
            tags.append("servable")
        if tags:
            a = w["alloc"].get(m, {})
            extra = f"  ctx={a.get('ctx')} @{a.get('endpoint')}" if a else ""
            print(f"  {w['name']:<10} {' '.join(tags)}{extra}")
    print(f"\nPin with:  hpy call {m} \"...\" -w <worker>")


def cmd_call(args):
    if not KEY:
        sys.exit("error: HUGPY_KEY not set (need a hp_... bearer key for /v1). export HUGPY_KEY=...")
    prompt = args.prompt or sys.stdin.read()
    if not prompt.strip():
        sys.exit("error: empty prompt (pass as arg or via stdin)")
    msgs = []
    if args.system:
        msgs.append({"role": "system", "content": args.system})
    msgs.append({"role": "user", "content": prompt})
    body = {"model": args.model, "messages": msgs}
    if args.max_tokens:
        body["max_tokens"] = args.max_tokens
    if args.temp is not None:
        body["temperature"] = args.temp
    if args.worker:
        body["alloc"] = {"worker": args.worker}
    try:
        resp = _post("/v1/chat/completions", body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        sys.exit(f"HTTP {e.code}: {detail}\n(500 'cannot serve' = worker lacks model; 503 cold_load_capacity = retry)")
    if args.raw:
        print(json.dumps(resp, indent=2))
        return
    print(resp["choices"][0]["message"]["content"])


def _is_textgen(task, tasks):
    t = (task or "").lower()
    allt = " ".join(tasks).lower()
    return ("text-generation" in t or "text2text" in t or "summar" in t
            or "text-generation" in allt or "text2text" in allt or t == "?")


def _plan(cat, ws, force_model=None, force_worker=None):
    """Return ordered list of (model, worker_or_None, why). Warm first, then
    on-disk, then cold-download catalog — so emergency calls land fastest."""
    warm, local = {}, {}
    for w in ws:
        for m in w["loaded"]:
            warm.setdefault(m, w["name"])
        for m in w["local"]:
            local.setdefault(m, w["name"])
    plan, seen = [], set()

    def add(m, worker, why):
        key = (m, worker)
        if key not in seen:
            seen.add(key)
            plan.append((m, worker, why))

    if force_model:
        # try the forced model on: its warm worker, then forced/any worker (central routes+cold-loads)
        if force_worker:
            add(force_model, force_worker, "forced model+worker")
        if force_model in warm:
            add(force_model, warm[force_model], "warm")
        add(force_model, None, "central route (may cold-load)")
        return plan
    # auto emergency ranking: warm text-gen -> on-disk text-gen -> catalog serveable text-gen
    for m, wk in warm.items():
        info = cat.get(m, {})
        if _is_textgen(info.get("task"), info.get("tasks", [])):
            add(m, wk, "warm")
    for m, wk in local.items():
        info = cat.get(m, {})
        if _is_textgen(info.get("task"), info.get("tasks", [])):
            add(m, wk, "on-disk (loads, no download)")
    for m, info in cat.items():
        if info.get("serveable") and not info.get("blocked") and _is_textgen(info.get("task"), info.get("tasks", [])):
            add(m, None, "catalog (cold-download)")
    return plan


def cmd_ask(args):
    cat = catalog()
    ws = workers()
    plan = _plan(cat, ws, args.model, args.worker)
    if not plan:
        sys.exit("no candidate text-generation model found on the fleet")
    if args.plan:
        for m, wk, why in plan:
            print(f"  {m}  @{wk or 'central'}  ({why})")
        return
    if not KEY:
        sys.exit("error: HUGPY_KEY not set (need a hp_... bearer key). export HUGPY_KEY=...")
    prompt = args.prompt or sys.stdin.read()
    if not prompt.strip():
        sys.exit("error: empty prompt")
    msgs = []
    if args.system:
        msgs.append({"role": "system", "content": args.system})
    msgs.append({"role": "user", "content": prompt})
    last = ""
    for m, wk, why in plan:
        body = {"model": m, "messages": msgs}
        if args.max_tokens:
            body["max_tokens"] = args.max_tokens
        if args.temp is not None:
            body["temperature"] = args.temp
        if wk:
            body["alloc"] = {"worker": wk}
        for attempt in range(args.retries + 1):
            try:
                resp = _post("/v1/chat/completions", body)
                sys.stderr.write(f"[answered via {m} @ {wk or 'central'} — {why}]\n")
                print(resp["choices"][0]["message"]["content"])
                return
            except urllib.error.HTTPError as e:
                last = e.read().decode(errors="replace")[:200]
                if e.code == 503 and attempt < args.retries:
                    sys.stderr.write(f"[503 cold-load busy on {m}, retrying…]\n")
                    continue
                sys.stderr.write(f"[skip {m}@{wk or 'central'}: HTTP {e.code} {last}]\n")
                break
            except urllib.error.URLError as e:
                sys.stderr.write(f"[skip {m}: {e}]\n")
                break
    sys.exit(f"emergency inference failed — all {len(plan)} candidates exhausted. last: {last}")


def cmd_overview(args):
    print(f"hugpy @ {BASE}   (key: {'set' if KEY else 'MISSING'})")
    cmd_workers(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="hpy", description="lightweight hugpy fleet browser + caller "
                                "(stdlib only; HUGPY_URL / HUGPY_KEY select the central)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("workers", help="list workers + warm/local models + tasks")
    pm = sub.add_parser("models", help="list models across the fleet")
    pm.add_argument("-t", "--task", help="filter by task substring")
    pm.add_argument("-w", "--worker", help="only models present on this worker")
    pm.add_argument("--warm", action="store_true", help="only warm (instant) models")
    pm.add_argument("--local", action="store_true", help="only warm or on-disk (no download)")
    pw = sub.add_parser("where", help="which workers hold a model")
    pw.add_argument("model")
    pc = sub.add_parser("call", help="chat with a model")
    pc.add_argument("model")
    pc.add_argument("prompt", nargs="?", help="prompt text (or pipe via stdin)")
    pc.add_argument("-w", "--worker", help="pin to a worker (hard filter)")
    pc.add_argument("-s", "--system", help="system prompt")
    pc.add_argument("-n", "--max-tokens", type=int)
    pc.add_argument("--temp", type=float)
    pc.add_argument("--raw", action="store_true", help="dump full JSON response")
    pa = sub.add_parser("ask", help="emergency inference: auto-pick fastest model & fall back")
    pa.add_argument("prompt", nargs="?", help="prompt (or pipe via stdin)")
    pa.add_argument("-m", "--model", help="force a specific model (still falls back warm->central)")
    pa.add_argument("-w", "--worker", help="prefer a specific worker")
    pa.add_argument("-s", "--system", help="system prompt")
    pa.add_argument("-n", "--max-tokens", type=int, default=512)
    pa.add_argument("--temp", type=float)
    pa.add_argument("--retries", type=int, default=2, help="503 cold-load retries per candidate")
    pa.add_argument("--plan", action="store_true", help="show the fallback plan, don't call")
    args = p.parse_args(argv)
    fn = {None: cmd_overview, "workers": cmd_workers, "models": cmd_models,
          "where": cmd_where, "call": cmd_call, "ask": cmd_ask}[args.cmd]
    try:
        fn(args)
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {BASE}: {e}")


if __name__ == "__main__":
    sys.exit(main())

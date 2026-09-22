"""model_sync — pull whole model directories from a central abstractgpt node.

The "list models -> resolve where each goes -> pull the entire directory" flow,
wired onto central's existing read-only endpoints:

    GET /api/llm/models                      the model list
    GET /api/llm/models/<key>/manifest       a model's files + sizes
    GET /api/llm/models/<key>/file?path=..   one file (Range/resumable)
    GET /api/llm/models/<key>/archive        the whole directory as one tar

It reuses the worker provisioning core (``ensure_model_present``), so every pull
is archive-first, falls back to a verified/resumable per-file transfer, is
single-flight, and is complete-or-raise — a partial directory is never left
looking done. Use it to sync a box's model storage from central without the GPU
worker agent.

    # list what central has
    hugpy-storage sync --central https://hugpy.ai --list

    # pull one model (or several), or everything
    hugpy-storage sync --central https://hugpy.ai --model DAN-Qwen3-1.7B
    hugpy-storage sync --central https://hugpy.ai --all

As a library:

    from hugpy_storage.model_sync import list_models, pull_model, pull_models
    pull_models("https://hugpy.ai", ["DAN-Qwen3-1.7B"])
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from hugpy_storage.provision import ensure_model_present, list_central_models

logger = logging.getLogger("hugpy_storage.model_sync")


def list_models(central_url: str) -> list[dict]:
    """Central's model rows (each carries model_key/key + hub_id + routing meta)."""
    return list_central_models(central_url)


def _model_key_of(row: dict) -> str | None:
    return row.get("model_key") or row.get("key")


def resolve_local_path(model: dict, root: str | None = None) -> str:
    """Where this model's directory should live locally (central's layout)."""
    from hugpy_storage.model_paths import route_destination

    return route_destination(model, root) if root else route_destination(model)


def pull_model(central_url: str, model_key: str, root: str | None = None,
               progress=None) -> bool:
    """Pull one model's entire directory from central. Returns True on success.

    Delegates to the worker provisioning core (archive-first, verified per-file
    fallback, single-flight, complete-or-raise).
    """
    return ensure_model_present(model_key, central_url, progress=progress)


def pull_models(central_url: str, model_keys: list[str] | None = None,
                root: str | None = None, progress=None) -> dict[str, bool]:
    """Pull several models (or all of central's) into local storage.

    ``model_keys=None`` pulls everything central lists. Returns
    ``{model_key: ok}``.
    """
    rows = list_models(central_url)
    by_key = {k: r for r in rows if (k := _model_key_of(r))}

    keys = model_keys if model_keys is not None else list(by_key)
    if not keys:
        logger.warning("no models to pull (central listed none and none given)")
        return {}

    results: dict[str, bool] = {}
    for key in keys:
        model = by_key.get(key)
        if model is not None:
            try:
                logger.info("pulling %s -> %s", key, resolve_local_path(model, root))
            except Exception:
                logger.info("pulling %s", key)
        else:
            logger.info("pulling %s (not in central's list; trying anyway)", key)

        try:
            ok = pull_model(central_url, key, root=root, progress=progress)
        except Exception as exc:  # noqa: BLE001 — one model's failure shouldn't stop the rest
            logger.error("pull of %s failed: %s", key, exc)
            ok = False
        results[key] = ok
        logger.info("pull %s: %s", key, "ok" if ok else "FAILED")
    return results


def _stderr_progress():
    """A throttled progress printer for CLI use."""
    state = {"last": 0.0}

    def report(done, total, name):
        now = time.time()
        if now - state["last"] < 0.2 and not (total and done >= total):
            return
        state["last"] = now
        pct = (100.0 * done / total) if total else 0.0
        sys.stderr.write(f"\r  {str(name or ''):<24} {pct:5.1f}%  {done}/{total}      ")
        sys.stderr.flush()
        if total and done >= total:
            sys.stderr.write("\n")

    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="hugpy_storage.model_sync")
    parser.add_argument("--central", required=True,
                        help="central base URL, e.g. https://hugpy.ai")
    parser.add_argument("--model", action="append", dest="models",
                        help="model_key to pull (repeatable)")
    parser.add_argument("--all", action="store_true",
                        help="pull every model central lists")
    parser.add_argument("--root", default=None,
                        help="local storage root override (default: this box's)")
    parser.add_argument("--list", action="store_true",
                        help="just list central's models and exit")
    args = parser.parse_args(argv)

    if args.list:
        rows = list_models(args.central)
        if not rows:
            print("central listed no models (or no list endpoint answered)",
                  file=sys.stderr)
            return 1
        for row in rows:
            print(f"{_model_key_of(row)}\t{row.get('hub_id')}\t{row.get('framework')}")
        return 0

    if not args.models and not args.all:
        print("specify --model KEY (repeatable), --all, or --list", file=sys.stderr)
        return 2

    keys = None if args.all else args.models
    results = pull_models(args.central, keys, root=args.root,
                          progress=_stderr_progress())
    failed = [k for k, ok in results.items() if not ok]
    ok_count = len(results) - len(failed)
    print(f"\npulled {ok_count}/{len(results)} models"
          + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

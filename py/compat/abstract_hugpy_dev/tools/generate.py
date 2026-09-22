"""Regenerate the transitional ``abstract_hugpy_dev`` compatibility surface.

Inputs (all under ``tools/inputs/`` now that the monolith tree is gone)
  * the last monolith wheel (every module that ever shipped),
  * the monolith's generated ``_relocations.json`` (old -> new module map),
  * ``surface.json``, the recorded surface probe (see below),
  * ``py/partition.toml`` (ownership; fills map entries the map missed),
  * the installed ``hugpy_*`` packages (which targets really exist),
  * a transitional interpreter run (``PYTHONPATH=<monolith>/src
    HUGPY_ALLOW_MONOLITH=1``) that records which public names
    ``import abstract_hugpy_dev`` used to expose and where each now lives.

Outputs (all under ``src/abstract_hugpy_dev/`` unless noted)
  * ``_relocations.json``  corrected map, only entries whose target exists
  * ``_namespace.json``    retired aggregators (import statements) + retired modules
  * ``__init__.py``        explicit re-export list, grouped per new module
  * ``../../RELOCATIONS.md``  the human-readable table

Usage (workspace root):
  .venv/bin/python py/compat/abstract_hugpy_dev/tools/generate.py
  .venv/bin/python py/compat/abstract_hugpy_dev/tools/generate.py --surface-probe  # internal
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
SRC = PKG_ROOT / "src" / "abstract_hugpy_dev"
PY_ROOT = PKG_ROOT.parents[1]
WORKSPACE = PY_ROOT.parent
MONOLITH = WORKSPACE / "abstract_hugpy_dev"  # gone after the partition; inputs/ replaces it
INPUTS = HERE / "inputs"  # last monolith wheel, its relocation map, the recorded surface probe
ROOT_NAME = "abstract_hugpy_dev"
LAYERS = [
    "hugpy_platform", "hugpy_control", "hugpy_storage", "hugpy_engine", "hugpy_media",
    "hugpy_video", "hugpy_oracle", "hugpy_fleet", "hugpy_curation", "hugpy_ops",
    "hugpy_discord", "hugpy_server", "hugpy",
]
VERSION_FALLBACK = "0.1.267.dev0"

sys.path.insert(0, str(PY_ROOT / "tooling"))
from partition_lib import load_manifest  # noqa: E402


# ---------------------------------------------------------------------------
# wheel enumeration + aggregator parsing


def wheel_modules(wheel: Path) -> "OrderedDict[str, tuple[str, str]]":
    """old dotted module -> (monolith-relative path, source)."""
    out: OrderedDict[str, tuple[str, str]] = OrderedDict()
    with zipfile.ZipFile(wheel) as zf:
        for name in sorted(zf.namelist()):
            if not name.startswith(ROOT_NAME + "/") or not name.endswith(".py"):
                continue
            rel = name[len(ROOT_NAME) + 1:]
            parts = rel[:-3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            old = ".".join([ROOT_NAME, *parts])
            out[old] = (rel, zf.read(name).decode("utf-8", "replace"))
    return out


def _resolve_relative(old: str, is_package: bool, level: int, module: str | None) -> str:
    package = old if is_package else old.rsplit(".", 1)[0]
    if level > 1:
        package = ".".join(package.split(".")[: -(level - 1)])
    return package + ("." + module if module else "") if package else (module or "")


def aggregator_statements(old: str, rel: str, source: str, lenient: bool = False) -> list[dict] | None:
    """Import statements of a pure re-export module, or None if it has real code.

    ``lenient`` keeps the import statements and ignores everything else (used
    for the old package root, whose ``__init__`` also set ``__version__``).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    is_package = rel.endswith("__init__.py")
    stmts: list[dict] = []

    def visit(node) -> bool:
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            return True
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                return True
            target = _resolve_relative(old, is_package, node.level, node.module) if node.level else node.module
            if any(a.name == "*" for a in node.names):
                stmts.append({"kind": "star", "module": target})
            else:
                stmts.append({"kind": "from", "module": target,
                              "names": [[a.name, a.asname or a.name] for a in node.names]})
            return True
        if isinstance(node, ast.Import):
            for a in node.names:
                stmts.append({"kind": "import", "module": a.name, **({"as": a.asname} if a.asname else {})})
            return True
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "__all__":
                return True
            if isinstance(node.value, ast.Name):
                stmts.append({"kind": "alias", "name": node.targets[0].id, "target": node.value.id})
                return True
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and not call.keywords
                    and all(isinstance(a, ast.Name) and a.id == "__name__" for a in call.args)):
                # ``logger = get_logFile(__name__)`` idiom of the aggregator modules
                stmts.append({"kind": "call", "name": node.targets[0].id, "func": call.func.id,
                              "argc": len(call.args)})
                return True
            return False
        if isinstance(node, ast.Try):
            return all(visit(n) for n in node.body) and all(
                all(visit(n) for n in h.body) for h in node.handlers
            )
        if isinstance(node, ast.Pass):
            return True
        return False

    for node in tree.body:
        if not visit(node) and not lenient:
            return None
    return stmts


def target_exists(new: str) -> bool:
    try:
        return importlib.util.find_spec(new) is not None
    except (ImportError, AttributeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# surface probe (runs inside the transitional interpreter)


def surface_probe() -> None:
    import inspect
    import types

    import abstract_hugpy_dev  # noqa: F401  (the old namespace, monolith on PYTHONPATH)

    def ga(obj, name):
        try:
            return getattr(obj, name, None)
        except Exception:
            return None

    def layer(mod: str) -> int:
        top = mod.split(".")[0]
        return LAYERS.index(top) if top in LAYERS else 999

    names = [n for n in dir(abstract_hugpy_dev) if not n.startswith("_")]
    hug_modules = sorted(
        (k for k, v in list(sys.modules.items()) if v is not None and k.split(".")[0] in LAYERS),
        key=lambda k: (layer(k), k.count("."), k),
    )
    stdlib = getattr(sys, "stdlib_module_names", set())
    out: dict[str, dict] = {}
    for n in names:
        obj = getattr(abstract_hugpy_dev, n)
        rec: dict = {"kind": type(obj).__name__}
        if isinstance(obj, types.ModuleType):
            top = obj.__name__.split(".")[0]
            rec.update(kind="module", module=obj.__name__)
            rec["source"] = "monolith" if top == ROOT_NAME else ("hugpy" if top in LAYERS else "external")
            out[n] = rec
            continue
        defmod = getattr(obj, "__module__", None) if (inspect.isroutine(obj) or inspect.isclass(obj)) else None
        if isinstance(defmod, str) and defmod.split(".")[0] not in LAYERS and defmod.split(".")[0] != ROOT_NAME:
            top = defmod.split(".")[0]
            pick = None
            public = [top.lstrip("_")] if top.startswith("_") else []
            for cand in (*public, top, defmod):
                try:
                    mod = importlib.import_module(cand)
                except Exception:
                    continue
                if ga(mod, n) is obj:
                    pick = cand
                    break
            if pick is not None:
                rec.update(source="external", module=pick, stdlib=top in stdlib)
                out[n] = rec
                continue
        cands = [k for k in hug_modules if ga(sys.modules[k], n) is obj]
        if isinstance(defmod, str) and defmod in cands:
            rec.update(source="hugpy", module=defmod)
        elif cands:
            rec.update(source="hugpy", module=cands[0])
        else:
            others = sorted(
                (k for k, v in list(sys.modules.items())
                 if v is not None and not k.startswith(ROOT_NAME) and ga(v, n) is obj),
                key=lambda k: (any(part.startswith("_") for part in k.split(".")), k.count("."), k),
            )
            if others:
                rec.update(source="external", module=others[0], stdlib=others[0].split(".")[0] in stdlib)
            else:
                rec.update(source="monolith", module=defmod)
        out[n] = rec
    json.dump(out, sys.stdout, indent=1)


def run_surface_probe(monolith_src: Path, python: str) -> dict[str, dict]:
    env = dict(os.environ, PYTHONPATH=str(monolith_src), HUGPY_ALLOW_MONOLITH="1")
    proc = subprocess.run(
        [python, str(Path(__file__).resolve()), "--surface-probe"],
        env=env, capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit("surface probe failed")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# emitters


def emit_init(surface: dict[str, dict]) -> str:
    stdlib_mods: dict[str, list[str]] = defaultdict(list)
    third_mods: dict[str, list[str]] = defaultdict(list)
    hugpy_mods: dict[str, list[str]] = defaultdict(list)
    module_binds: list[tuple[str, str, str]] = []  # (name, module, source)
    not_relocated: list[str] = []
    for name, rec in sorted(surface.items()):
        src = rec["source"]
        if src == "monolith":
            not_relocated.append(name)
            continue
        if rec["kind"] == "module":
            module_binds.append((name, rec["module"], src))
            continue
        if src == "hugpy":
            hugpy_mods[rec["module"]].append(name)
        elif rec.get("stdlib"):
            stdlib_mods[rec["module"]].append(name)
        else:
            third_mods[rec["module"]].append(name)

    def layer(mod: str) -> tuple:
        top = mod.split(".")[0]
        return (LAYERS.index(top) if top in LAYERS else 999, mod)

    lines: list[str] = []
    w = lines.append
    w('"""abstract_hugpy_dev - transitional alias over the hugpy-* distributions.')
    w("")
    w("GENERATED by tools/generate.py; do not edit by hand.")
    w("")
    w("This package contains no implementation. Importing it installs a meta-path")
    w("finder that aliases every old ``abstract_hugpy_dev.<module>`` path to the same")
    w("module object in its new ``hugpy_*`` home (see RELOCATIONS.md), and re-exports")
    w("below every public name ``import abstract_hugpy_dev`` used to expose at top")
    w("level (monolith 0.1.266). Each group is independent: a missing optional")
    w("distribution only removes its own names.")
    w('"""')
    w("")
    w("from __future__ import annotations")
    w("")
    w("from ._relocations import install as _install_relocations")
    w("from ._relocations import aggregator_getattr as _aggregator_getattr")
    w("")
    w("_install_relocations()")
    w("")
    w("try:")
    w("    from importlib.metadata import version as _pkg_version")
    w("")
    w('    __version__ = _pkg_version("abstract_hugpy_dev")')
    w("except Exception:  # uninstalled source run")
    w(f'    __version__ = "{VERSION_FALLBACK}"')
    w("")
    w("# The old namespace installed the pure-Python pydantic shim before anything")
    w("# imported pydantic (phones without pydantic_core). Keep that contract.")
    w("try:")
    w("    from hugpy_platform.compat_pydantic import ensure_pydantic as _ensure_pydantic")
    w("")
    w("    _ensure_pydantic()")
    w("except ImportError:  # hugpy-platform missing: nothing below will import either")
    w("    pass")
    w("")
    if module_binds:
        w("# --- modules the old namespace exposed as attributes -----------------------")
        stdlib = getattr(sys, "stdlib_module_names", set())
        for name, module, src in sorted(module_binds, key=lambda t: (t[2] != "external", t[1])):
            if src == "external" and module.split(".")[0] in stdlib:
                w(f"import {module} as {name}" if module != name else f"import {module}")
        for name, module, src in sorted(module_binds, key=lambda t: (t[2] != "external", t[1])):
            if src == "external" and module.split(".")[0] not in stdlib:
                w("try:")
                w(f"    import {module} as {name}" if module != name else f"    import {module}")
                w(f"except ImportError:  # {module.split('.')[0]} not installed")
                w("    pass")
        w("")
        hug_binds = [(n, m) for n, m, s in module_binds if s == "hugpy"]
        if hug_binds:
            w("try:")
            for name, module in sorted(hug_binds, key=lambda t: layer(t[1])):
                w(f"    import {module} as {name}")
            w("except ImportError:")
            w("    pass")
            w("")
    if stdlib_mods:
        w("# --- standard library names the old namespace re-exported ----------------")
        w("# Bound one by one: a name this interpreter lacks (added or removed in another")
        w("# Python release) is skipped and listed in _SURFACE_UNAVAILABLE instead of")
        w("# breaking the whole import.")
        w("from ._relocations import reexport_stdlib as _reexport_stdlib")
        w("")
        w("_SURFACE_UNAVAILABLE: list[str] = []")
        for module in sorted(stdlib_mods, key=_public_stdlib_module):
            names = sorted(set(stdlib_mods[module]))
            w(_reexport_lines(_public_stdlib_module(module), names))
        w("")
    if third_mods:
        w("# --- third-party names the old namespace re-exported ---------------------")
        for module in sorted(third_mods):
            w("try:")
            w(_from_line(module, third_mods[module], indent="    "))
            w(f"except ImportError:  # {module.split('.')[0]} not installed")
            w("    pass")
        w("")
    current_layer = None
    for module in sorted(hugpy_mods, key=layer):
        top = module.split(".")[0]
        if top != current_layer:
            current_layer = top
            dist = top.replace("_", "-")
            w(f"# --- {dist} " + "-" * max(1, 74 - len(dist)))
        w("try:")
        w(_from_line(module, hugpy_mods[module], indent="    "))
        w(f"except ImportError:  # {top.replace('_', '-')} missing or its optional dependency absent")
        w("    pass")
    w("")
    if not_relocated:
        w("# not relocated: " + ", ".join(not_relocated))
        w("#   These were the retired aggregator packages themselves (imports/, managers/,")
        w("#   utils/ ...). They resolve lazily through __getattr__ below as namespace")
        w("#   modules, e.g. ``abstract_hugpy_dev.managers.serve`` -> hugpy_engine.serve.")
        w("")
    all_names = sorted(surface)
    w("_SURFACE = (")
    for chunk in _chunks(all_names, 6):
        w("    " + ", ".join(repr(n) for n in chunk) + ",")
    w(")")
    w("")
    w("__all__ = [_n for _n in _SURFACE if _n in globals()]")
    w("")
    w("")
    w("def __getattr__(name: str):")
    w('    """Lazy fallback: anything ``from .imports/.managers/.utils import *`` bound."""')
    w("    return _aggregator_getattr(__name__, name)")
    w("")
    return "\n".join(lines)


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _public_stdlib_module(module: str) -> str:
    """``_typing`` -> ``typing``: the probe may record a C accelerator module as
    the origin of an object; import it from the public module when that exposes it."""
    top = module.split(".")[0]
    if not top.startswith("_"):
        return module
    public = top.lstrip("_")
    try:
        importlib.import_module(public)
    except ImportError:
        return module
    return public


def _reexport_lines(module: str, names: list[str]) -> str:
    head = f"_SURFACE_UNAVAILABLE += _reexport_stdlib(globals(), {module!r}, ["
    one = head + ", ".join(repr(n) for n in names) + "])"
    if len(one) <= 96:
        return one
    body = ",\n".join(f"    {n!r}" for n in names)
    return f"{head}\n{body},\n])"


def _from_line(module: str, names: list[str], indent: str) -> str:
    names = sorted(set(names))
    one = f"{indent}from {module} import " + ", ".join(names)
    if len(one) <= 96:
        return one
    body = ",\n".join(f"{indent}    {n}" for n in names)
    return f"{indent}from {module} import (\n{body},\n{indent})"


def emit_relocations_md(relocations: dict[str, str], aggregators: dict[str, list[dict]],
                        retired: dict[str, str], wheel: Path, surface_counts: dict[str, int]) -> str:
    by_pkg: dict[str, int] = defaultdict(int)
    for new in relocations.values():
        by_pkg[new.split(".")[0]] += 1
    lines = [
        "# abstract_hugpy_dev relocations",
        "",
        f"Generated from `{wheel.name}` (the last monolith release) and the installed",
        "`hugpy-*` packages by `tools/generate.py`. `import abstract_hugpy_dev` installs",
        "a finder that makes every **old module** below resolve to the **new module**",
        "object (same object, not a copy). Retired aggregators are synthesised as lazy",
        "namespace modules; retired code has no alias and must be replaced.",
        "",
        "| Kind | Count |",
        "|---|---|",
        f"| Relocated modules (aliased) | {len(relocations)} |",
        f"| Retired aggregators (lazy namespaces) | {len(aggregators)} |",
        f"| Retired modules (no alias) | {len(retired)} |",
        f"| Top-level names re-exported | {surface_counts.get('exported', 0)} |",
        f"| Top-level names not relocated | {surface_counts.get('not_relocated', 0)} |",
        "",
        "## Relocated modules per package",
        "",
        "| Package | Modules |",
        "|---|---|",
    ]
    for pkg in sorted(by_pkg, key=lambda p: LAYERS.index(p) if p in LAYERS else 99):
        lines.append(f"| `{pkg}` | {by_pkg[pkg]} |")
    lines += ["", "## Old module -> new module", "", "| Old module | New module |", "|---|---|"]
    for old, new in sorted(relocations.items()):
        lines.append(f"| `{old}` | `{new}` |")
    lines += ["", "## Retired aggregators (lazy namespaces)", "",
              "Attribute lookups replay the aggregator's old import statements against the",
              "new packages, so `from abstract_hugpy_dev.comms import job_store` still works.", "",
              "| Old module | Re-exported from |", "|---|---|"]
    for old, stmts in sorted(aggregators.items()):
        srcs = []
        for s in stmts:
            m = s.get("module")
            if m and m not in srcs and s["kind"] in ("star", "from"):
                srcs.append(m)
        lines.append(f"| `{old}` | " + ", ".join(f"`{m}`" for m in srcs[:8]) + (" ..." if len(srcs) > 8 else "") + " |")
    lines += ["", "## Retired modules (no alias)", "", "| Old module | Note |", "|---|---|"]
    for old, note in sorted(retired.items()):
        lines.append(f"| `{old}` | {note} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface-probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--wheel", type=Path, default=None,
                    help="last monolith wheel (default: newest inputs/*.whl, else abstract_hugpy_dev/dist/*.whl)")
    ap.add_argument("--map", type=Path, default=INPUTS / "monolith_relocations.json",
                    help="the monolith's generated _relocations.json (default: inputs/monolith_relocations.json)")
    ap.add_argument("--monolith-src", type=Path, default=MONOLITH / "src",
                    help="monolith source tree, only needed to re-record the surface probe")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--surface-json", type=Path, default=INPUTS / "surface.json",
                    help="saved surface probe (default: inputs/surface.json); re-recorded from "
                         "--monolith-src only when the file is missing")
    ap.add_argument("--skip-surface", action="store_true", help="only regenerate the map/namespace files")
    args = ap.parse_args(argv)

    if args.surface_probe:
        surface_probe()
        return 0

    wheel = args.wheel
    if wheel is None:
        wheels = sorted(INPUTS.glob("abstract_hugpy_dev-*.whl")) or sorted(
            (MONOLITH / "dist").glob("abstract_hugpy_dev-*.whl"))
        if not wheels:
            raise SystemExit("no monolith wheel found; pass --wheel")
        wheel = wheels[-1]
    manifest = load_manifest()
    old_map: dict[str, str] = json.loads(args.map.read_text(encoding="utf-8")) if args.map.exists() else {}
    modules = wheel_modules(wheel)

    relocations: dict[str, str] = {}
    aggregators: dict[str, list[dict]] = {}
    retired: dict[str, str] = {}
    unwired = PY_ROOT / "unwired"

    def note_retired(old: str, rel: str, why: str) -> None:
        stem = Path(rel).name
        copies = [p.relative_to(PY_ROOT).as_posix() for p in unwired.rglob(stem)] if unwired.exists() else []
        retired[old] = why + (f"; copy at py/{copies[0]}" if copies else "")

    for old, (rel, source) in modules.items():
        new = old_map.get(old)
        pid, root, dest = manifest.owner_of(rel)
        if new is None and pid not in ("retire", "unowned"):
            new = manifest.new_module(rel)
        if new is not None and target_exists(new):
            relocations[old] = new
            continue
        if old == ROOT_NAME:
            aggregators[old] = aggregator_statements(old, rel, source, lenient=True) or []
            continue
        stmts = aggregator_statements(old, rel, source)
        if stmts is not None:
            aggregators[old] = stmts
            continue
        if new is not None:
            note_retired(old, rel, f"target `{new}` no longer exists in its package")
        elif pid == "retire":
            note_retired(old, rel, "retired by partition.toml (dead or superseded code)")
        else:
            note_retired(old, rel, "not owned by any package")
    # entries the monolith map knows about that never shipped in the wheel
    # (modules created during the partition) are kept when they exist.
    for old, new in old_map.items():
        if old not in relocations and old not in aggregators and old not in retired and target_exists(new):
            relocations[old] = new

    SRC.mkdir(parents=True, exist_ok=True)
    (SRC / "_relocations.json").write_text(json.dumps(relocations, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (SRC / "_namespace.json").write_text(
        json.dumps({"aggregators": aggregators, "retired": retired}, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"relocated={len(relocations)} aggregators={len(aggregators)} retired={len(retired)}")

    surface_counts = {}
    if not args.skip_surface:
        if args.surface_json and args.surface_json.exists():
            surface = json.loads(args.surface_json.read_text(encoding="utf-8"))
        else:
            if not (args.monolith_src / ROOT_NAME / "__init__.py").exists():
                raise SystemExit(f"no saved surface probe at {args.surface_json} and no monolith "
                                 f"source at {args.monolith_src}; restore inputs/surface.json")
            surface = run_surface_probe(args.monolith_src, args.python)
            if args.surface_json:
                args.surface_json.write_text(json.dumps(surface, indent=1), encoding="utf-8")
        (SRC / "__init__.py").write_text(emit_init(surface), encoding="utf-8")
        surface_counts = {
            "exported": sum(1 for r in surface.values() if r["source"] != "monolith"),
            "not_relocated": sum(1 for r in surface.values() if r["source"] == "monolith"),
        }
        print(f"surface names={len(surface)} exported={surface_counts['exported']} "
              f"not_relocated={surface_counts['not_relocated']}")
    (PKG_ROOT / "RELOCATIONS.md").write_text(
        emit_relocations_md(relocations, aggregators, retired, wheel, surface_counts), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

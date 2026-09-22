"""Shared helpers for the monolith-to-distributions partition.

Everything here is derived from ``py/partition.toml``: which package owns a
monolith source path, where a moved module lives now, and which import edges
are allowed between packages. Both the validator and the extraction tools use
this module so they can never disagree about ownership.
"""

from __future__ import annotations

import ast
import json
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

PY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PY_ROOT / "partition.toml"
RETIRE_ID = "retire"


@dataclass
class Package:
    id: str
    distribution: str
    import_name: str
    destination: str  # relative to PY_ROOT
    status: str
    depends: list[str]
    optional_depends: list[str]
    moves: dict[str, str]  # monolith-relative source path -> package-relative dest

    @property
    def root(self) -> Path:
        return PY_ROOT / self.destination

    @property
    def src_dir(self) -> Path:
        return self.root / "src" / self.import_name

    @property
    def allowed(self) -> set[str]:
        return set(self.depends) | set(self.optional_depends)


@dataclass
class Manifest:
    source_root: Path
    monolith_import: str
    packages: dict[str, Package]
    order: list[str]
    retired: list[str]
    external: list[dict]
    _owner_index: list[tuple[str, str, str]] = field(default_factory=list)

    # -- ownership ---------------------------------------------------------
    def owner_of(self, rel: str) -> tuple[str, str | None, str | None]:
        """Return (package_id | 'retire' | 'unowned', matched_root, dest)."""
        rel = rel.replace("\\", "/")
        best: tuple[str, str, str | None] | None = None
        for root, pid, dest in self._owner_index:
            if rel == root or rel.startswith(root.rstrip("/") + "/"):
                if best is None or len(root) > len(best[1]):
                    best = (pid, root, dest)
        if best is None:
            return "unowned", None, None
        return best

    def dest_path(self, rel: str) -> tuple[str, Path] | None:
        """Where a monolith-relative source path lands: (package_id, path)."""
        pid, root, dest = self.owner_of(rel)
        if pid in (RETIRE_ID, "unowned") or root is None or dest is None:
            return None
        pkg = self.packages[pid]
        if rel == root:
            tail = ""
        else:
            tail = rel[len(root.rstrip("/")) + 1 :]
        base = pkg.src_dir if dest in (".", "") else pkg.src_dir / dest
        return pid, (base / tail if tail else base)

    # -- module names ------------------------------------------------------
    def old_module(self, rel: str) -> str | None:
        if not rel.endswith(".py"):
            return None
        parts = rel[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join([self.monolith_import, *parts]) if parts else self.monolith_import

    def new_module(self, rel: str) -> str | None:
        """Module name for a monolith-relative .py path after its move."""
        if not rel.endswith(".py"):
            return None
        landing = self.dest_path(rel)
        if landing is None:
            return None
        pid, path = landing
        pkg = self.packages[pid]
        sub = path.relative_to(pkg.src_dir).with_suffix("")
        parts = list(sub.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join([pkg.import_name, *parts])

    def module_map(self, only_existing: bool = False) -> dict[str, str]:
        """old dotted module -> new dotted module for every owned .py file."""
        out: dict[str, str] = {}
        for rel in self.iter_source_files(existing=only_existing):
            old = self.old_module(rel)
            new = self.new_module(rel)
            if old and new:
                out[old] = new
        return out

    # -- source enumeration ------------------------------------------------
    def iter_source_files(self, existing: bool = True) -> Iterator[str]:
        """Every file under source_root (relative posix paths)."""
        if not self.source_root.exists():
            return
        for path in sorted(self.source_root.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            yield path.relative_to(self.source_root).as_posix()

    def package_for_module(self, module: str) -> str | None:
        """Package id (or 'monolith') for a dotted module name."""
        top = module.split(".")[0]
        if top == self.monolith_import:
            return "monolith"
        for pid, pkg in self.packages.items():
            if top == pkg.import_name:
                return pid
        return None


def load_manifest(path: Path = MANIFEST) -> Manifest:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    here = path.parent
    packages: dict[str, Package] = {}
    order: list[str] = []
    for row in data["package"]:
        pkg = Package(
            id=row["id"],
            distribution=row["distribution"],
            import_name=row["import_name"],
            destination=row["destination"],
            status=row.get("status", "new"),
            depends=list(row.get("depends", [])),
            optional_depends=list(row.get("optional_depends", [])),
            moves=dict(row.get("moves", {})),
        )
        packages[pkg.id] = pkg
        order.append(pkg.id)
    retired: list[str] = []
    for row in data.get("retire", []):
        retired.extend(row.get("source_roots", []))
    manifest = Manifest(
        source_root=(here / data["source_root"]).resolve(),
        monolith_import=data.get("monolith_import", "abstract_hugpy_dev"),
        packages=packages,
        order=order,
        retired=retired,
        external=list(data.get("external_package", [])),
    )
    index: list[tuple[str, str, str]] = []
    for pkg in packages.values():
        for src, dest in pkg.moves.items():
            index.append((src, pkg.id, dest))
    for src in retired:
        index.append((src, RETIRE_ID, ""))
    manifest._owner_index = index
    return manifest


# ---------------------------------------------------------------------------
# Import graph analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImportEdge:
    src_file: str  # path relative to the workspace (py/ or monolith)
    src_pkg: str
    dst_module: str
    dst_pkg: str
    lineno: int


def _module_parts_for(rel: str, top: str) -> list[str]:
    parts = [top, *rel[:-3].split("/")]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def imported_modules(path: Path, rel: str, top: str) -> Iterator[tuple[str, int]]:
    """Yield (absolute dotted module, lineno) for every import in ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return
    modparts = _module_parts_for(rel, top)
    is_pkg = rel.endswith("__init__.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # package __init__ at level 1 refers to itself
                cut = len(modparts) - node.level + (1 if is_pkg else 0)
                base = modparts[:cut]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            yield mod, node.lineno
            for alias in node.names:
                if alias.name != "*":
                    yield f"{mod}.{alias.name}", node.lineno


_DOTTED_RE = None


def referenced_modules(path: Path, rel: str, top: str) -> Iterator[tuple[str, int]]:
    """Static imports plus dotted module strings (import_module, patch targets)."""
    import re

    global _DOTTED_RE
    yield from imported_modules(path, rel, top)
    if _DOTTED_RE is None:
        _DOTTED_RE = re.compile(r"^(?:abstract_hugpy_dev|hugpy_[a-z_]+|hugpy)(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?::[A-Za-z_][A-Za-z0-9_]*)?$")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _DOTTED_RE.match(node.value):
            yield node.value.split(":")[0], node.lineno


def workspace_edges(manifest: Manifest) -> list[ImportEdge]:
    """Cross-package import edges across the monolith and extracted packages."""
    edges: list[ImportEdge] = []

    def scan(base: Path, top: str, label_of):
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(base).as_posix()
            src_pkg = label_of(rel)
            for mod, lineno in imported_modules(path, rel, top):
                dst = manifest.package_for_module(mod)
                if dst is None:
                    continue
                if dst == "monolith":
                    dst_rel = _monolith_rel_for(manifest, mod)
                    dst = manifest.owner_of(dst_rel)[0] if dst_rel else "monolith"
                if dst == src_pkg:
                    continue
                edges.append(
                    ImportEdge(
                        src_file=str(path.relative_to(PY_ROOT.parent)),
                        src_pkg=src_pkg,
                        dst_module=mod,
                        dst_pkg=dst,
                        lineno=lineno,
                    )
                )

    if manifest.source_root.exists():
        scan(
            manifest.source_root,
            manifest.monolith_import,
            lambda rel: manifest.owner_of(rel)[0],
        )
    for pkg in manifest.packages.values():
        if pkg.src_dir.exists():
            scan(pkg.src_dir, pkg.import_name, lambda rel, pid=pkg.id: pid)
    return edges


def _monolith_rel_for(manifest: Manifest, module: str) -> str | None:
    """Best-effort map of a dotted monolith module to a source-relative path."""
    parts = module.split(".")[1:]
    root = manifest.source_root
    while parts:
        cand = root.joinpath(*parts)
        if cand.is_dir():
            return "/".join(parts) + "/__init__.py"
        if cand.with_suffix(".py").exists():
            return "/".join(parts) + ".py"
        parts = parts[:-1]
    return "__init__.py"


def forbidden_edges(manifest: Manifest, edges: Iterable[ImportEdge]) -> list[ImportEdge]:
    out = []
    for edge in edges:
        if edge.src_pkg in (RETIRE_ID, "unowned", "monolith"):
            continue
        pkg = manifest.packages.get(edge.src_pkg)
        if pkg is None:
            continue
        if edge.dst_pkg in ("monolith", RETIRE_ID, "unowned"):
            out.append(edge)
        elif edge.dst_pkg not in pkg.allowed:
            out.append(edge)
    return out


def summarize_edges(edges: Iterable[ImportEdge]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for edge in edges:
        summary[edge.src_pkg][edge.dst_pkg] += 1
    return {k: dict(v) for k, v in summary.items()}


def dump_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")

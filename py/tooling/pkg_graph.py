"""pkg_graph — symbol index, duplicate detection and dependency graph for pkg_src.

Every .py blob in ``pkg_src.blobs`` is parsed ONCE (blobs are content-addressed,
so an unchanged file in a new micro version costs nothing) into:

    pkg_src.py_parsed    one row per parsed blob (ok / syntax error)
    pkg_src.py_symbols   functions, methods, classes, module/class variables
    pkg_src.py_imports   every import, with the scope it binds in
    pkg_src.py_calls     every call site: caller qualname -> callee as written

Symbols carry two body hashes: ``body_sha`` (exact source, docstring stripped)
and ``norm_sha`` (AST with argument/local names canonicalised), so copies and
renamed copies are both found.

A *graph* resolves those raw rows for one member set ({package: version_id}):
module names, imports (following re-exports through __init__), ``self.``/``cls.``
methods up base classes, and classifies each edge internal / cross-package /
external / missing. Graphs are cached by member set:

    pkg_src.graphs          one per distinct member set
    pkg_src.graph_modules   module -> package, version, file
    pkg_src.graph_edges     caller -> callee (aggregated call sites)
    pkg_src.graph_imports   module -> module/package import edges

``pkg_src.test_trace`` holds, per verify job, which workspace functions each
test actually executed (coverage contexts), so a tested function's trace is
explicit: test -> functions run, and function -> tests that ran it.
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import re
from collections import defaultdict

from psycopg import sql
from psycopg.types.json import Jsonb

SCHEMA = "pkg_src"
MIN_DUP_NODES = 12            # ignore trivial bodies (pass / return x) as duplicates
_BUILTINS = set(dir(builtins))


def S():
    return sql.Identifier(SCHEMA)


def ensure_graph_schema(cur) -> None:
    # Already there → no DDL (read/cache-only roles such as the toolserver's lack CREATE).
    cur.execute("SELECT to_regclass('pkg_src.test_trace') IS NOT NULL")
    if cur.fetchone()[0]:
        return
    cur.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {s}.py_parsed (
            sha256      text PRIMARY KEY REFERENCES {s}.blobs(sha256),
            ok          boolean NOT NULL,
            error       text,
            n_symbols   integer NOT NULL DEFAULT 0,
            parsed_at   timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS {s}.py_symbols (
            sha256      text NOT NULL REFERENCES {s}.blobs(sha256),
            qualname    text NOT NULL,
            kind        text NOT NULL,     -- function|async_function|method|class|variable|class_variable
            name        text NOT NULL,
            parent      text,              -- enclosing qualname ('' = module)
            lineno      integer NOT NULL,
            end_lineno  integer,
            args        text,
            decorators  text[],
            bases       text[],
            doc         text,
            body_sha    text,
            norm_sha    text,
            node_count  integer,
            PRIMARY KEY (sha256, qualname, lineno)
        );
        CREATE INDEX IF NOT EXISTS py_symbols_norm ON {s}.py_symbols (norm_sha);
        CREATE INDEX IF NOT EXISTS py_symbols_name ON {s}.py_symbols (name);
        CREATE TABLE IF NOT EXISTS {s}.py_imports (
            sha256      text NOT NULL REFERENCES {s}.blobs(sha256),
            scope       text NOT NULL,     -- '' = module level, else function qualname
            lineno      integer NOT NULL,
            level       integer NOT NULL,  -- relative-import dots
            module      text,              -- as written ('' for "from . import x")
            name        text,              -- NULL for "import module"
            asname      text
        );
        CREATE INDEX IF NOT EXISTS py_imports_sha ON {s}.py_imports (sha256);
        CREATE TABLE IF NOT EXISTS {s}.py_calls (
            sha256      text NOT NULL REFERENCES {s}.blobs(sha256),
            caller      text NOT NULL,     -- '' = module level
            callee      text NOT NULL,     -- dotted, as written: foo / mod.foo / self.bar
            lineno      integer NOT NULL
        );
        CREATE INDEX IF NOT EXISTS py_calls_sha ON {s}.py_calls (sha256);
        CREATE TABLE IF NOT EXISTS {s}.graphs (
            id          bigserial PRIMARY KEY,
            members_key text NOT NULL UNIQUE,
            members     jsonb NOT NULL,
            stats       jsonb,
            built_at    timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS {s}.graph_modules (
            graph_id    bigint NOT NULL REFERENCES {s}.graphs(id) ON DELETE CASCADE,
            module      text NOT NULL,
            package     text NOT NULL,
            version_id  bigint NOT NULL,
            rel_path    text NOT NULL,
            sha256      text NOT NULL,
            is_test     boolean NOT NULL,
            PRIMARY KEY (graph_id, module)
        );
        CREATE TABLE IF NOT EXISTS {s}.graph_edges (
            graph_id    bigint NOT NULL REFERENCES {s}.graphs(id) ON DELETE CASCADE,
            src_module  text NOT NULL,
            src_qualname text NOT NULL,
            dst_module  text NOT NULL,     -- external: top-level distribution/module
            dst_qualname text NOT NULL,
            kind        text NOT NULL,     -- internal|cross|external|dynamic|missing
            sites       integer NOT NULL,
            first_line  integer NOT NULL
        );
        CREATE INDEX IF NOT EXISTS graph_edges_src ON {s}.graph_edges (graph_id, src_module, src_qualname);
        CREATE INDEX IF NOT EXISTS graph_edges_dst ON {s}.graph_edges (graph_id, dst_module, dst_qualname);
        CREATE TABLE IF NOT EXISTS {s}.graph_imports (
            graph_id    bigint NOT NULL REFERENCES {s}.graphs(id) ON DELETE CASCADE,
            src_module  text NOT NULL,
            dst_module  text NOT NULL,
            dst_package text,              -- NULL = external
            kind        text NOT NULL,     -- internal|cross|external|missing
            lineno      integer NOT NULL
        );
        CREATE TABLE IF NOT EXISTS {s}.test_trace (
            job_id      bigint NOT NULL REFERENCES {s}.jobs(id) ON DELETE CASCADE,
            package     text NOT NULL,     -- package whose suite ran the test
            test        text NOT NULL,     -- pytest node id
            module      text NOT NULL,
            qualname    text NOT NULL,
            lines       integer NOT NULL,  -- lines of that function the test executed
            PRIMARY KEY (job_id, test, module, qualname)
        );
        CREATE INDEX IF NOT EXISTS test_trace_fn ON {s}.test_trace (module, qualname);
    """).format(s=S()))


# ------------------------------------------------------------------------ parse

def _dotted(node) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _strip_doc(body):
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


class _Canon(ast.NodeTransformer):
    """Rename args and locally-bound names to v0, v1 … in order of appearance."""

    def __init__(self, local: set[str]):
        self.local, self.map = local, {}

    def _n(self, name):
        if name not in self.local:
            return name
        return self.map.setdefault(name, f"v{len(self.map)}")

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self._n(node.id), ctx=node.ctx), node)

    def visit_arg(self, node):
        node.arg = self._n(node.arg)
        node.annotation = None
        return node


def _hashes(fn) -> tuple[str, str, int]:
    body = _strip_doc(fn.body)
    exact = hashlib.sha256("\n".join(ast.unparse(b) for b in body).encode()).hexdigest()
    local = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            local.add(n.id)
    clone = copy.deepcopy(fn)
    clone.name, clone.decorator_list, clone.returns = "_", [], None
    clone.body = _strip_doc(clone.body) or [ast.Pass()]
    canon = _Canon(local).visit(clone)
    norm = hashlib.sha256(ast.dump(canon, annotate_fields=False).encode()).hexdigest()
    return exact, norm, sum(1 for _ in ast.walk(ast.Module(body=body, type_ignores=[])))


def parse_source(text: str):
    """(symbols, imports, calls) for one file; raises SyntaxError."""
    tree = ast.parse(text)
    symbols, imports, calls = [], [], []

    def visit(nodes, parent: str, in_class: bool, fn_scope: str):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{parent}.{node.name}" if parent else node.name
                exact, norm, size = _hashes(node)
                kind = "method" if in_class else (
                    "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")
                symbols.append(dict(qualname=q, kind=kind, name=node.name, parent=parent,
                                    lineno=node.lineno, end_lineno=node.end_lineno,
                                    args=ast.unparse(node.args),
                                    decorators=[ast.unparse(d) for d in node.decorator_list],
                                    bases=None, doc=(ast.get_docstring(node) or "").split("\n")[0][:300] or None,
                                    body_sha=exact, norm_sha=norm, node_count=size))
                scan(node.body, q)
                visit(node.body, q, False, q)
            elif isinstance(node, ast.ClassDef):
                q = f"{parent}.{node.name}" if parent else node.name
                symbols.append(dict(qualname=q, kind="class", name=node.name, parent=parent,
                                    lineno=node.lineno, end_lineno=node.end_lineno, args=None,
                                    decorators=[ast.unparse(d) for d in node.decorator_list],
                                    bases=[b for b in (_dotted(x) for x in node.bases) if b],
                                    doc=(ast.get_docstring(node) or "").split("\n")[0][:300] or None,
                                    body_sha=None, norm_sha=None, node_count=None))
                scan([n for n in node.body if not isinstance(
                    n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], fn_scope)
                visit(node.body, q, True, fn_scope)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and not fn_scope:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            symbols.append(dict(qualname=f"{parent}.{n.id}" if parent else n.id,
                                                kind="class_variable" if in_class else "variable",
                                                name=n.id, parent=parent, lineno=node.lineno,
                                                end_lineno=node.end_lineno, args=None,
                                                decorators=None, bases=None, doc=None,
                                                body_sha=None, norm_sha=None, node_count=None))
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)) and not fn_scope:
                # module-level conditional defs/imports (try: import x / if TYPE_CHECKING)
                inner = [c for f in ("body", "orelse", "finalbody") for c in getattr(node, f, [])]
                for h in getattr(node, "handlers", []):
                    inner += h.body
                visit(inner, parent, in_class, fn_scope)

    def scan(nodes, scope: str):
        """Imports + calls directly inside ``nodes`` (not inside nested defs)."""
        stack = list(nodes)
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)) \
                    and n not in nodes:
                if isinstance(n, ast.Lambda):
                    stack.extend(ast.iter_child_nodes(n))
                continue
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.extend(n.decorator_list)       # decorators run in the enclosing scope
                continue
            if isinstance(n, ast.Import):
                for a in n.names:
                    imports.append(dict(scope=scope, lineno=n.lineno, level=0, module=a.name,
                                        name=None, asname=a.asname))
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    imports.append(dict(scope=scope, lineno=n.lineno, level=n.level or 0,
                                        module=n.module or "", name=a.name, asname=a.asname))
            elif isinstance(n, ast.Call):
                d = _dotted(n.func)
                if d:
                    calls.append(dict(caller=scope, callee=d, lineno=n.lineno))
            stack.extend(ast.iter_child_nodes(n))

    scan(tree.body, "")
    visit(tree.body, "", False, "")
    return symbols, imports, calls


def index_blobs(conn, log=lambda *_: None) -> int:
    """Parse every .py blob referenced by any version that isn't parsed yet."""
    with conn.cursor() as cur:
        ensure_graph_schema(cur)
        cur.execute(sql.SQL("""
            SELECT DISTINCT b.sha256, b.content_text FROM {s}.version_files f
            JOIN {s}.blobs b USING (sha256)
            LEFT JOIN {s}.py_parsed p ON p.sha256 = b.sha256
            WHERE f.rel_path LIKE '%%.py' AND p.sha256 IS NULL AND b.content_text IS NOT NULL
        """).format(s=S()))
        todo = cur.fetchall()
        for sha, text in todo:
            try:
                syms, imps, calls = parse_source(text)
                ok, err = True, None
            except (SyntaxError, ValueError, RecursionError) as e:
                syms, imps, calls, ok, err = [], [], [], False, f"{type(e).__name__}: {e}"
            cur.executemany(sql.SQL(
                "INSERT INTO {s}.py_symbols (sha256, qualname, kind, name, parent, lineno, end_lineno, "
                "args, decorators, bases, doc, body_sha, norm_sha, node_count) VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING").format(s=S()),
                [(sha, x["qualname"], x["kind"], x["name"], x["parent"], x["lineno"], x["end_lineno"],
                  x["args"], x["decorators"], x["bases"], x["doc"], x["body_sha"], x["norm_sha"],
                  x["node_count"]) for x in syms])
            cur.executemany(sql.SQL(
                "INSERT INTO {s}.py_imports (sha256, scope, lineno, level, module, name, asname) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)").format(s=S()),
                [(sha, x["scope"], x["lineno"], x["level"], x["module"], x["name"], x["asname"])
                 for x in imps])
            cur.executemany(sql.SQL(
                "INSERT INTO {s}.py_calls (sha256, caller, callee, lineno) VALUES (%s,%s,%s,%s)")
                .format(s=S()), [(sha, x["caller"], x["callee"], x["lineno"]) for x in calls])
            cur.execute(sql.SQL("INSERT INTO {s}.py_parsed (sha256, ok, error, n_symbols) "
                                "VALUES (%s,%s,%s,%s)").format(s=S()), (sha, ok, err, len(syms)))
    conn.commit()
    if todo:
        log(f"indexed {len(todo)} new .py blobs")
    return len(todo)


# ------------------------------------------------------------------------ graph

def module_name(package: str, rel: str) -> tuple[str, bool]:
    """(dotted module, is_test) for a package-relative .py path."""
    p = rel[:-3]
    parts = p.split("/")
    is_test = "tests" in parts[:-1] or parts[-1].startswith("test_") or parts[-1] == "conftest"
    if parts[0] == "src":
        parts = parts[1:]
    elif parts[0] in ("tests", "test"):
        parts = [f"{package}[tests]"] + parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or package, is_test


class Graph:
    """In-memory resolver for one member set."""

    def __init__(self, cur, members: dict[str, int]):
        self.members = members
        cur.execute(sql.SQL("""
            SELECT v.package, f.version_id, f.rel_path, f.sha256
            FROM {s}.version_files f JOIN {s}.versions v ON v.id = f.version_id
            WHERE f.version_id = ANY(%s) AND f.rel_path LIKE '%%.py'""").format(s=S()),
            (list(members.values()),))
        self.modules = {}                      # module -> (package, version_id, rel, sha, is_test)
        for pkg, vid, rel, sha in cur.fetchall():
            mod, is_test = module_name(pkg, rel)
            self.modules[mod] = (pkg, vid, rel, sha, is_test)
        shas = list({m[3] for m in self.modules.values()})
        self.syms = defaultdict(dict)          # sha -> qualname -> row
        cur.execute(sql.SQL("SELECT sha256, qualname, kind, bases, lineno, end_lineno FROM {s}.py_symbols "
                            "WHERE sha256 = ANY(%s)").format(s=S()), (shas,))
        for sha, q, kind, bases, lo, hi in cur.fetchall():
            self.syms[sha][q] = (kind, bases or [], lo, hi)
        self.imps = defaultdict(lambda: defaultdict(dict))   # sha -> scope -> bound name -> target
        cur.execute(sql.SQL("SELECT sha256, scope, level, module, name, asname, lineno FROM {s}.py_imports "
                            "WHERE sha256 = ANY(%s)").format(s=S()), (shas,))
        self.raw_imports = defaultdict(list)
        for sha, scope, level, module, name, asname, lineno in cur.fetchall():
            self.raw_imports[sha].append((scope, level, module, name, asname, lineno))
        self.calls = defaultdict(list)
        cur.execute(sql.SQL("SELECT sha256, caller, callee, lineno FROM {s}.py_calls WHERE sha256 = ANY(%s)")
                    .format(s=S()), (shas,))
        for sha, caller, callee, lineno in cur.fetchall():
            self.calls[sha].append((caller, callee, lineno))
        self.roots = {m.split(".")[0] for m in self.modules}
        for mod, (_, _, _, sha, _) in self.modules.items():
            for scope, level, module, name, asname, _ in self.raw_imports[sha]:
                absmod = self.absolute(mod, level, module)
                if name is None:               # import a.b.c [as x]
                    bound = asname or module.split(".")[0]
                    target = module if asname else module.split(".")[0]
                    self.imps[sha][scope][bound] = ("module", target)
                elif name != "*":
                    self.imps[sha][scope][asname or name] = ("from", absmod, name)

    def absolute(self, mod: str, level: int, module: str) -> str:
        if not level:
            return module
        is_pkg = self.modules.get(mod, ("", 0, ""))[2].endswith("__init__.py")
        base = mod.split(".")
        base = base if is_pkg else base[:-1]
        base = base[:len(base) - (level - 1)] if level > 1 else base
        return ".".join(base + ([module] if module else []))

    # -- resolution: returns (module, qualname) inside the workspace, or
    #    ("!ext", top-level name) / ("!missing", dotted) / None (unknown)
    def lookup(self, mod: str, name: str, depth: int = 0):
        """What ``name`` means at module scope of ``mod``."""
        if depth > 8 or mod not in self.modules:
            return None
        sha = self.modules[mod][3]
        if name in self.syms[sha]:
            return (mod, name)
        b = self.imps[sha][""].get(name)
        if b:
            return self.follow(b, depth + 1)
        sub = f"{mod}.{name}"
        if sub in self.modules:
            return (sub, "")
        return None

    def absent(self, mod: str, name: str):
        """A name a workspace module doesn't define: lazily served (PEP 562) or missing."""
        if mod in self.modules and "__getattr__" in self.syms[self.modules[mod][3]]:
            return ("!dynamic", f"{mod}.{name}")
        return ("!missing", f"{mod}.{name}")

    def follow(self, b, depth):
        if b[0] == "module":
            return self.module_ref(b[1])
        _, absmod, name = b
        if absmod in self.modules:
            r = self.lookup(absmod, name, depth)
            if r:
                return r
            if f"{absmod}.{name}" in self.modules:
                return (f"{absmod}.{name}", "")
            return self.absent(absmod, name)
        if f"{absmod}.{name}" in self.modules:
            return (f"{absmod}.{name}", "")
        return self.module_ref(absmod, name)

    def module_ref(self, dotted: str, attr: str | None = None):
        full = f"{dotted}.{attr}" if attr else dotted
        if full in self.modules:
            return (full, "")
        if dotted.split(".")[0] in self.roots:
            if dotted in self.modules and attr:
                return ("!missing", full)
            return ("!missing", full) if dotted not in self.modules else (dotted, "")
        return ("!ext", dotted.split(".")[0])

    def attr(self, target, rest: list[str], depth=0):
        """Walk ``.a.b`` from a resolved target."""
        for i, part in enumerate(rest):
            if target is None or target[0].startswith("!"):
                return target
            mod, q = target
            if q == "":                        # module: look up member
                target = self.lookup(mod, part, depth) or (
                    self.absent(mod, part) if mod in self.modules else None)
            else:
                kind = self.syms[self.modules[mod][3]].get(q, ("",))[0]
                if kind == "class":
                    target = self.method(mod, q, part, depth)
                else:
                    return None                # attribute of a value: not statically known
        return target

    def method(self, mod, cls, name, depth=0):
        if depth > 8:
            return None
        sha = self.modules[mod][3]
        if f"{cls}.{name}" in self.syms[sha]:
            return (mod, f"{cls}.{name}")
        for base in self.syms[sha].get(cls, ("", []))[1]:
            b = self.resolve_name(mod, "", base, depth + 1)
            if b and not b[0].startswith("!") and b[1]:
                r = self.method(b[0], b[1], name, depth + 1)
                if r:
                    return r
        return None

    def resolve_name(self, mod, scope, dotted, depth=0):
        sha = self.modules[mod][3]
        head, *rest = dotted.split(".")
        # function-local imports, innermost scope outward
        s = scope
        while s:
            b = self.imps[sha][s].get(head)
            if b:
                return self.attr(self.follow(b, depth + 1), rest, depth)
            s = s.rsplit(".", 1)[0] if "." in s else ""
        if head in ("self", "cls") and scope:
            cls = self._enclosing_class(sha, scope)
            if cls and rest:
                return self.attr(self.method(mod, cls, rest[0], depth), rest[1:], depth)
            return None
        t = self.lookup(mod, head, depth)
        if t is None:
            return ("!builtin", head) if head in _BUILTINS and not rest else None
        return self.attr(t, rest, depth)

    def _enclosing_class(self, sha, scope):
        parts = scope.split(".")
        for i in range(len(parts) - 1, 0, -1):
            q = ".".join(parts[:i])
            if self.syms[sha].get(q, ("",))[0] == "class":
                return q
        return None

    def edges(self):
        out = defaultdict(lambda: [0, 10 ** 9])
        for mod, (pkg, _, _, sha, _) in self.modules.items():
            for caller, callee, lineno in self.calls[sha]:
                t = self.resolve_name(mod, caller, callee)
                if t is None or t[0] == "!builtin":
                    continue
                if t[0] == "!ext":
                    key = (mod, caller, t[1], callee.split(".", 1)[-1] if "." in callee else callee, "external")
                elif t[0] in ("!missing", "!dynamic"):
                    key = (mod, caller, t[1], "", t[0][1:])
                else:
                    dpkg = self.modules[t[0]][0]
                    key = (mod, caller, t[0], t[1], "internal" if dpkg == pkg else "cross")
                e = out[key]
                e[0] += 1
                e[1] = min(e[1], lineno)
        return out

    def import_edges(self):
        rows = []
        for mod, (pkg, _, _, sha, _) in self.modules.items():
            for scope, level, module, name, asname, lineno in self.raw_imports[sha]:
                absmod = self.absolute(mod, level, module)
                cand = f"{absmod}.{name}" if name and f"{absmod}.{name}" in self.modules else absmod
                if cand in self.modules:
                    dpkg = self.modules[cand][0]
                    rows.append((mod, cand, dpkg, "internal" if dpkg == pkg else "cross", lineno))
                elif absmod.split(".")[0] in self.roots:
                    rows.append((mod, cand, None, "missing", lineno))
                else:
                    rows.append((mod, absmod.split(".")[0], None, "external", lineno))
        return rows


def members_key(members: dict[str, int]) -> str:
    return hashlib.sha256(json.dumps(sorted(members.items())).encode()).hexdigest()


def build_graph(conn, members: dict[str, int], log=lambda *_: None) -> int:
    """Resolve (or reuse) the graph for ``members``; returns graph id."""
    members = {k: int(v) for k, v in members.items()}
    key = members_key(members)
    with conn.cursor() as cur:
        ensure_graph_schema(cur)
        cur.execute(sql.SQL("SELECT id FROM {s}.graphs WHERE members_key = %s").format(s=S()), (key,))
        row = cur.fetchone()
        if row:
            return row[0]
    index_blobs(conn, log)
    with conn.cursor() as cur:
        g = Graph(cur, members)
        edges = g.edges()
        imports = g.import_edges()
        stats = {"modules": len(g.modules), "edges": len(edges),
                 "edge_kinds": dict(_count(k[4] for k in edges)),
                 "import_kinds": dict(_count(r[3] for r in imports))}
        cur.execute(sql.SQL("INSERT INTO {s}.graphs (members_key, members, stats) VALUES (%s,%s,%s) "
                            "RETURNING id").format(s=S()), (key, Jsonb(members), Jsonb(stats)))
        gid = cur.fetchone()[0]
        cur.executemany(sql.SQL("INSERT INTO {s}.graph_modules VALUES (%s,%s,%s,%s,%s,%s,%s)").format(s=S()),
                        [(gid, m, p, v, r, sh, t) for m, (p, v, r, sh, t) in g.modules.items()])
        cur.executemany(sql.SQL("INSERT INTO {s}.graph_edges VALUES (%s,%s,%s,%s,%s,%s,%s,%s)").format(s=S()),
                        [(gid, *k[:5], n, first) for k, (n, first) in edges.items()])
        cur.executemany(sql.SQL("INSERT INTO {s}.graph_imports VALUES (%s,%s,%s,%s,%s,%s)").format(s=S()),
                        [(gid, *r) for r in imports])
    conn.commit()
    log(f"graph {gid}: {stats}")
    return gid


def _count(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return sorted(d.items())


# --------------------------------------------------------------------- analyses

def duplicates(cur, gid: int, min_nodes: int = MIN_DUP_NODES, limit: int = 200) -> list[dict]:
    """Groups of functions/methods with the same canonical body across the graph."""
    cur.execute(sql.SQL("""
        SELECT s.norm_sha, max(s.node_count),
               count(DISTINCT s.body_sha) AS variants,
               json_agg(json_build_object('package', m.package, 'module', m.module,
                        'qualname', s.qualname, 'line', s.lineno, 'test', m.is_test,
                        'exact', s.body_sha) ORDER BY m.module, s.qualname)
        FROM {s}.graph_modules m JOIN {s}.py_symbols s ON s.sha256 = m.sha256
        WHERE m.graph_id = %s AND s.norm_sha IS NOT NULL AND s.node_count >= %s AND NOT m.is_test
        GROUP BY s.norm_sha HAVING count(*) > 1
        ORDER BY count(*) DESC, max(s.node_count) DESC LIMIT %s""").format(s=S()),
        (gid, min_nodes, limit))
    out = []
    for norm, nodes, variants, copies in cur.fetchall():
        out.append({"norm_sha": norm[:12], "nodes": nodes, "copies": len(copies),
                    "renamed_variants": variants > 1,
                    "packages": sorted({c["package"] for c in copies}),
                    "locations": [f"{c['module']}:{c['qualname']}:{c['line']}" for c in copies]})
    return out


def name_collisions(cur, gid: int, limit: int = 100) -> list[dict]:
    """Same top-level function name defined in >1 package with DIFFERENT bodies."""
    cur.execute(sql.SQL("""
        SELECT s.name, count(DISTINCT s.norm_sha), array_agg(DISTINCT m.package),
               array_agg(m.module || ':' || s.qualname ORDER BY m.module)
        FROM {s}.graph_modules m JOIN {s}.py_symbols s ON s.sha256 = m.sha256
        WHERE m.graph_id = %s AND s.kind IN ('function', 'async_function') AND s.parent = ''
          AND NOT m.is_test AND s.name NOT LIKE '\\_%%' AND s.name NOT IN ('main', 'get_toolset')
        GROUP BY s.name HAVING count(DISTINCT m.package) > 1 AND count(DISTINCT s.norm_sha) > 1
        ORDER BY count(*) DESC LIMIT %s""").format(s=S()), (gid, limit))
    return [{"name": n, "bodies": b, "packages": p, "locations": l} for n, b, p, l in cur.fetchall()]


def duplicate_calls(cur, gid: int, limit: int = 200) -> list[dict]:
    """Call sites that land on one copy of a duplicated function — consolidation map:
    for each duplicate group, which callers use which copy."""
    cur.execute(sql.SQL("""
        WITH dup AS (
            SELECT s.norm_sha, m.module, s.qualname
            FROM {s}.graph_modules m JOIN {s}.py_symbols s ON s.sha256 = m.sha256
            WHERE m.graph_id = %s AND s.norm_sha IS NOT NULL AND s.node_count >= %s AND NOT m.is_test
              AND s.norm_sha IN (
                SELECT s2.norm_sha FROM {s}.graph_modules m2 JOIN {s}.py_symbols s2 ON s2.sha256 = m2.sha256
                WHERE m2.graph_id = %s AND s2.node_count >= %s AND NOT m2.is_test
                GROUP BY s2.norm_sha HAVING count(*) > 1))
        SELECT d.norm_sha, d.module || ':' || d.qualname,
               coalesce(json_agg(e.src_module || ':' || e.src_qualname || ' (' || e.sites || ')')
                        FILTER (WHERE e.src_module IS NOT NULL), '[]')
        FROM dup d LEFT JOIN {s}.graph_edges e
          ON e.graph_id = %s AND e.dst_module = d.module AND e.dst_qualname = d.qualname
        GROUP BY d.norm_sha, d.module, d.qualname
        ORDER BY d.norm_sha, d.module LIMIT %s""").format(s=S()),
        (gid, MIN_DUP_NODES, gid, MIN_DUP_NODES, gid, limit))
    groups = defaultdict(list)
    for norm, loc, callers in cur.fetchall():
        groups[norm[:12]].append({"copy": loc, "callers": callers})
    return [{"norm_sha": k, "copies": v} for k, v in groups.items()]


def package_matrix(cur, gid: int) -> dict:
    """{package: {depends_on_package: import_count}} from resolved imports (non-test)."""
    cur.execute(sql.SQL("""
        SELECT sm.package, i.dst_package, count(*) FROM {s}.graph_imports i
        JOIN {s}.graph_modules sm ON sm.graph_id = i.graph_id AND sm.module = i.src_module
        WHERE i.graph_id = %s AND i.kind = 'cross' AND NOT sm.is_test
        GROUP BY 1, 2 ORDER BY 1, 2""").format(s=S()), (gid,))
    out = defaultdict(dict)
    for a, b, n in cur.fetchall():
        out[a][b] = n
    return dict(out)


def declared_deps(cur, members: dict[str, int]) -> dict[str, set[str]]:
    """{package: {sibling packages its pyproject lists}} (dist names -> dir names)."""
    cur.execute(sql.SQL("""
        SELECT v.package, b.content_text FROM {s}.version_files f
        JOIN {s}.versions v ON v.id = f.version_id JOIN {s}.blobs b USING (sha256)
        WHERE f.version_id = ANY(%s) AND f.rel_path = 'pyproject.toml'""").format(s=S()),
        (list(members.values()),))
    texts = dict(cur.fetchall())
    dist = {}
    for pkg, t in texts.items():
        m = re.search(r'^\s*name\s*=\s*"([^"]+)"', t or "", re.M)
        if m:
            dist[re.sub(r"[-_.]+", "-", m.group(1).lower())] = pkg
    out = {}
    for pkg, t in texts.items():
        deps = set()
        for m in re.finditer(r'"([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(?:[<>=!~;].*?)?"', t or ""):
            d = dist.get(re.sub(r"[-_.]+", "-", m.group(1).lower()))
            if d and d != pkg:
                deps.add(d)
        out[pkg] = deps
    return out


def dependency_problems(cur, gid: int, members: dict[str, int]) -> dict:
    s = S()
    cur.execute(sql.SQL("""
        SELECT i.src_module, i.dst_module, min(i.lineno) FROM {s}.graph_imports i
        WHERE i.graph_id = %s AND i.kind = 'missing' GROUP BY 1, 2 ORDER BY 1""").format(s=s), (gid,))
    missing_imports = [f"{a} imports {b} (line {n}) — not in the workspace" for a, b, n in cur.fetchall()]
    cur.execute(sql.SQL("""
        SELECT src_module || ':' || src_qualname, dst_module, min(first_line) FROM {s}.graph_edges
        WHERE graph_id = %s AND kind = 'missing' GROUP BY 1, 2 ORDER BY 1""").format(s=s), (gid,))
    missing_calls = [f"{a} calls {b} (line {n}) — no such symbol" for a, b, n in cur.fetchall()]
    matrix = package_matrix(cur, gid)
    declared = declared_deps(cur, members)
    undeclared = {p: sorted(set(d) - declared.get(p, set()) - {p}) for p, d in matrix.items()}
    undeclared = {p: d for p, d in undeclared.items() if d}
    # package-level cycles
    cycles, seen = [], set()
    def dfs(node, path):
        for nxt in matrix.get(node, {}):
            if nxt in path:
                cyc = path[path.index(nxt):] + [nxt]
                k = frozenset(cyc)
                if k not in seen:
                    seen.add(k); cycles.append(" -> ".join(cyc))
            elif len(path) < 12:
                dfs(nxt, path + [nxt])
    for p in matrix:
        dfs(p, [p])
    return {"missing_imports": missing_imports, "missing_calls": missing_calls,
            "undeclared_package_deps": undeclared, "package_cycles": cycles,
            "package_matrix": matrix}


def unreferenced(cur, gid: int, limit: int = 300) -> list[str]:
    """Top-level functions nothing in the workspace calls or imports by name (dead-code candidates)."""
    cur.execute(sql.SQL("""
        SELECT m.module || ':' || s.qualname FROM {s}.graph_modules m
        JOIN {s}.py_symbols s ON s.sha256 = m.sha256
        WHERE m.graph_id = %s AND NOT m.is_test AND s.kind IN ('function', 'async_function')
          AND s.parent = '' AND s.name NOT LIKE '\\_%%' AND s.name NOT IN ('main', 'get_toolset')
          AND coalesce(array_length(s.decorators, 1), 0) = 0
          AND NOT EXISTS (SELECT 1 FROM {s}.graph_edges e WHERE e.graph_id = m.graph_id
                          AND e.dst_module = m.module AND e.dst_qualname = s.qualname)
          AND NOT EXISTS (SELECT 1 FROM {s}.py_imports i JOIN {s}.graph_modules m2
                          ON m2.sha256 = i.sha256 AND m2.graph_id = m.graph_id WHERE i.name = s.name)
        ORDER BY 1 LIMIT %s""").format(s=S()), (gid, limit))
    return [r[0] for r in cur.fetchall()]


def callees(cur, gid: int, module: str, qualname: str, depth: int = 6) -> list[dict]:
    """Static transitive call chain from one function (workspace edges only)."""
    cur.execute(sql.SQL("""
        WITH RECURSIVE walk(module, qualname, depth, path) AS (
            SELECT %s::text, %s::text, 0, ARRAY[%s || ':' || %s]
          UNION
            SELECT e.dst_module, e.dst_qualname, w.depth + 1, w.path || (e.dst_module || ':' || e.dst_qualname)
            FROM walk w JOIN {s}.graph_edges e ON e.graph_id = %s AND e.src_module = w.module
                 AND e.src_qualname = w.qualname AND e.kind IN ('internal', 'cross')
            WHERE w.depth < %s AND NOT (e.dst_module || ':' || e.dst_qualname) = ANY(w.path))
        SELECT DISTINCT ON (module, qualname) module, qualname, depth, path FROM walk
        WHERE depth > 0 ORDER BY module, qualname, depth""").format(s=S()),
        (module, qualname, module, qualname, gid, depth))
    return [{"function": f"{m}:{q}", "depth": d, "via": p} for m, q, d, p in cur.fetchall()]


def callers(cur, gid: int, module: str, qualname: str, depth: int = 6) -> list[dict]:
    """Static transitive callers of one function."""
    cur.execute(sql.SQL("""
        WITH RECURSIVE walk(module, qualname, depth) AS (
            SELECT %s::text, %s::text, 0
          UNION
            SELECT e.src_module, e.src_qualname, w.depth + 1
            FROM walk w JOIN {s}.graph_edges e ON e.graph_id = %s AND e.dst_module = w.module
                 AND e.dst_qualname = w.qualname AND e.kind IN ('internal', 'cross')
            WHERE w.depth < %s)
        SELECT module, qualname, min(depth) FROM walk WHERE depth > 0
        GROUP BY 1, 2 ORDER BY 3, 1, 2""").format(s=S()), (module, qualname, gid, depth))
    return [{"function": f"{m}:{q}", "depth": d} for m, q, d in cur.fetchall()]


# ------------------------------------------------------------------ test trace

def load_coverage_trace(conn, job_id: int, package: str, cov_file: str, gid: int) -> int:
    """Map a coverage.py data file recorded with per-test contexts onto workspace
    functions; insert pkg_src.test_trace rows. Returns rows inserted."""
    import sqlite3
    db = sqlite3.connect(cov_file)
    files = dict(db.execute("SELECT id, path FROM file"))
    contexts = dict(db.execute("SELECT id, context FROM context"))
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT module, sha256 FROM {s}.graph_modules WHERE graph_id = %s")
                    .format(s=S()), (gid,))
        mods = dict(cur.fetchall())
        cur.execute(sql.SQL("""SELECT s.sha256, s.qualname, s.lineno, s.end_lineno FROM {s}.py_symbols s
                               WHERE s.sha256 = ANY(%s) AND s.kind IN ('function','async_function','method')""")
                    .format(s=S()), (list(set(mods.values())),))
        spans = defaultdict(list)
        for sha, q, lo, hi in cur.fetchall():
            spans[sha].append((lo, hi or lo, q))
    # path -> module: match the longest dotted suffix that is a workspace module
    def to_module(path: str) -> str | None:
        p = path[:-3] if path.endswith(".py") else path
        parts = p.replace("\\", "/").split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for i in range(len(parts)):
            m = ".".join(parts[i:])
            if m in mods:
                return m
        return None
    def numbits_to_nums(numbits: bytes):          # coverage.py's line bitmap (no import needed)
        return [i * 8 + b for i, byte in enumerate(numbits) for b in range(8) if byte & (1 << b)]
    agg = defaultdict(int)
    for file_id, ctx_id, numbits in db.execute("SELECT file_id, context_id, numbits FROM line_bits"):
        test = contexts.get(ctx_id) or ""
        if not test or "::" not in test:
            continue
        test = test.rsplit("|", 1)[0]
        mod = to_module(files[file_id])
        if mod is None:
            continue
        fns = sorted(spans[mods[mod]], key=lambda t: t[1] - t[0])   # innermost first
        for line in numbits_to_nums(numbits):
            for lo, hi, q in fns:
                if lo <= line <= hi:
                    agg[(test, mod, q)] += 1
                    break
    with conn.cursor() as cur:
        cur.executemany(sql.SQL("INSERT INTO {s}.test_trace VALUES (%s,%s,%s,%s,%s,%s) "
                                "ON CONFLICT DO NOTHING").format(s=S()),
                        [(job_id, package, t, m, q, n) for (t, m, q), n in agg.items()])
    conn.commit()
    return len(agg)


def trace_report(cur, job_id: int, gid: int, tested: list[str]) -> dict:
    """What the job's tests exercised: per-package function coverage, duplicated
    functions hit by tests, untested functions."""
    cur.execute(sql.SQL("""
        WITH fns AS (
            SELECT m.package, m.module, s.qualname FROM {s}.graph_modules m
            JOIN {s}.py_symbols s ON s.sha256 = m.sha256
            WHERE m.graph_id = %s AND NOT m.is_test AND m.package = ANY(%s)
              AND s.kind IN ('function', 'async_function', 'method')),
        hit AS (SELECT DISTINCT module, qualname FROM {s}.test_trace WHERE job_id = %s)
        SELECT f.package, count(*), count(h.module) FROM fns f
        LEFT JOIN hit h USING (module, qualname) GROUP BY 1 ORDER BY 1""").format(s=S()),
        (gid, tested, job_id))
    coverage = {p: {"functions": n, "executed_by_tests": h, "pct": round(100 * h / n, 1) if n else None}
                for p, n, h in cur.fetchall()}
    cur.execute(sql.SQL("""
        SELECT t.module || ':' || t.qualname, count(DISTINCT t.test), s.norm_sha
        FROM {s}.test_trace t
        JOIN {s}.graph_modules m ON m.graph_id = %s AND m.module = t.module
        JOIN {s}.py_symbols s ON s.sha256 = m.sha256 AND s.qualname = t.qualname
        WHERE t.job_id = %s AND s.node_count >= %s AND s.norm_sha IN (
            SELECT s2.norm_sha FROM {s}.graph_modules m2 JOIN {s}.py_symbols s2 ON s2.sha256 = m2.sha256
            WHERE m2.graph_id = %s AND NOT m2.is_test AND s2.node_count >= %s
            GROUP BY 1 HAVING count(*) > 1)
        GROUP BY 1, 3 ORDER BY 3, 1""").format(s=S()),
        (gid, job_id, MIN_DUP_NODES, gid, MIN_DUP_NODES))
    dup_hits = [{"function": f, "tests": n, "dup_group": d[:12]} for f, n, d in cur.fetchall()]
    return {"function_coverage": coverage, "tested_duplicates": dup_hits}


def function_tests(cur, module: str, qualname: str, job_id: int | None = None) -> list[dict]:
    """Which tests executed this function (latest job that traced it, or ``job_id``)."""
    cur.execute(sql.SQL("""
        SELECT job_id, test, lines FROM {s}.test_trace
        WHERE module = %s AND qualname = %s AND job_id = coalesce(%s,
              (SELECT max(job_id) FROM {s}.test_trace WHERE module = %s AND qualname = %s))
        ORDER BY test""").format(s=S()), (module, qualname, job_id, module, qualname))
    return [{"job": j, "test": t, "lines": n} for j, t, n in cur.fetchall()]


def test_functions(cur, job_id: int, test: str) -> list[dict]:
    """Which workspace functions one test executed, in module order."""
    cur.execute(sql.SQL("SELECT module, qualname, lines FROM {s}.test_trace WHERE job_id = %s AND test = %s "
                        "ORDER BY module, qualname").format(s=S()), (job_id, test))
    return [{"function": f"{m}:{q}", "lines": n} for m, q, n in cur.fetchall()]

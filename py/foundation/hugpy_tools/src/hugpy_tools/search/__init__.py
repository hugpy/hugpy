"""hugpy_tools.search — a fast, self-contained local content search.

Stdlib-only. Imports NOTHING from the rest of hugpy (or any hugpy_* package);
this subpackage can be lifted out as its own distribution unchanged.

SIBLING COPY: ``abstract_search.engine`` (AbstractEndeavors) is a deliberately
kept-in-step copy of this engine. abstract_* packages never depend on hugpy_*,
so the code is duplicated, not shared — change matching/filter semantics here and
mirror them there (and vice-versa).

NO IMPLICIT FILTERING (operator ruling 2026-09-24)
--------------------------------------------------
A search-specific engine allows/excludes NOTHING silently. With NO user filters,
``iter_files`` / ``get_files_and_dirs`` / ``find_content`` return EVERYTHING
under the roots — dotfiles, ``node_modules``, ``.venv``, binaries. There is no
default extension whitelist, no default excluded dirs, no default hidden-file
skipping.

Opt into the old conveniences EXPLICITLY with named presets: ``preset="code"``
(``CODE_TEXT_EXTS`` + ``EXCLUDE_NOISE_DIRS``, skip hidden) or ``preset="noise"``
(``EXCLUDE_NOISE_DIRS`` + ``BINARY_EXTS``, skip hidden), or pass the exported sets
as filter kwargs.

PRESET MERGE SEMANTICS = UNION (so you are never surprised): presets and your
kwargs are UNIONed field-by-field. Good for excludes (more excluded); but the
allow-whitelist is unioned too, so a whitelist preset BROADENS —
``preset="code"`` + ``allowed_exts=[".rs"]`` matches code/text AND ``.rs``, not
just ``.rs``. To NARROW by extension use ``preset="noise"`` (excludes only) +
your ``allowed_exts``, or no preset. Everything applied shows in the report.

Nothing is dropped silently: every ``search`` / ``collect`` / ``find_content`` /
``get_files_and_dirs`` records :func:`last_report` — the exact ``effective_filters``
(incl. presets) and ``skipped`` counts by reason (binary / unreadable /
too_large / symlink_loop). Content search still skips binary files (a NUL-byte
match is meaningless) but reports them. Pass ``report=True`` for ``(result,
report)`` inline.

Include/exclude semantics: exclude always wins; include narrows; directory
excludes are exact path SEGMENTS pruned at walk time; globs match the
root-relative path AND the basename; names/globs/exts are case-insensitive;
literal content matching is case-insensitive unless ``case_sensitive=True``.
Symlinks are not followed by default (least surprising); ``max_depth`` /
``max_bytes`` are explicit (defaults: unbounded, no cap).
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence, Tuple

__all__ = [
    "Filters", "make_filters", "iter_files", "collect", "read_text",
    "read_any_file", "search", "search_content", "find_content", "findContent",
    "get_file_filters", "get_files_and_dirs", "getPaths", "find_lines",
    "stringInContent", "last_report", "effective_filters",
    "provider", "LocalSearchProvider",
    "PRESETS", "CODE_TEXT_EXTS", "EXCLUDE_NOISE_DIRS", "BINARY_EXTS",
]

# --------------------------------------------------------------------------
# Opt-in presets / named sets
# --------------------------------------------------------------------------
EXCLUDE_NOISE_DIRS: frozenset = frozenset({
    ".git", ".hg", ".svn", ".bzr",
    "node_modules", "bower_components",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
    ".tox", ".nox", ".eggs", "*.egg-info",
    ".venv", "venv", "env", "virtualenv", "site-packages",
    "dist", "build", ".next", ".nuxt", ".svelte-kit", "out",
    ".idea", ".vscode", ".gradle", "target",
    "htmlcov",
})

BINARY_EXTS: frozenset = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svgz",
    ".tif", ".tiff", ".heic", ".heif", ".psd", ".raw", ".jp2", ".jxl",
    ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v", ".mpg",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz", ".whl",
    ".jar", ".war", ".egg", ".dmg", ".iso", ".img",
    ".bin", ".exe", ".dll", ".so", ".o", ".a", ".dylib", ".class",
    ".pyc", ".pyo", ".pdb",
    ".pt", ".pth", ".safetensors", ".gguf", ".onnx", ".npy", ".npz",
    ".h5", ".hdf5", ".pickle", ".pkl",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".xlsb", ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
    ".shp", ".shx", ".dbf", ".sbn", ".sbx", ".db", ".sqlite", ".sqlite3",
})

CODE_TEXT_EXTS: frozenset = frozenset({
    ".py", ".pyw", ".pyi",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".html", ".htm", ".xml", ".svg",
    ".css", ".scss", ".sass", ".less",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".md", ".markdown", ".rst", ".txt",
    ".sh", ".bash", ".zsh", ".c", ".h", ".cpp", ".hpp", ".go", ".rs",
    ".java", ".kt", ".rb", ".php", ".sql", ".lua", ".vue", ".svelte",
})

PRESETS: dict = {
    "code":  {"allowed_exts": CODE_TEXT_EXTS, "exclude_dirs": EXCLUDE_NOISE_DIRS,
              "include_hidden": False},
    "noise": {"exclude_dirs": EXCLUDE_NOISE_DIRS, "exclude_exts": BINARY_EXTS,
              "include_hidden": False},
    "text":  {"exclude_exts": BINARY_EXTS, "exclude_dirs": EXCLUDE_NOISE_DIRS,
              "include_hidden": False},
}

_BINARY_SNIFF_BYTES = 8192

_STOP = threading.Event()


def request_stop() -> None:
    _STOP.set()


def reset_stop() -> None:
    _STOP.clear()


def should_stop() -> bool:
    return _STOP.is_set()


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
_LAST_REPORT: dict = {}


def last_report() -> dict:
    """Report from the most recent search/collect: ``effective_filters``,
    ``scanned``, ``matched`` (search) and ``skipped`` counts by reason. Works
    for the fixed-shape legacy APIs; ``report=True`` returns it inline."""
    return dict(_LAST_REPORT)


def _blank_skipped() -> dict:
    return {"binary": 0, "unreadable": 0, "too_large": 0, "symlink_loop": 0}


def _set_report(rep: dict) -> dict:
    global _LAST_REPORT
    _LAST_REPORT = rep
    return rep


# --------------------------------------------------------------------------
# Filters — every field defaults to "no filtering"
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Filters:
    exts: Optional[frozenset] = None
    exclude_exts: frozenset = frozenset()
    dir_names: Optional[frozenset] = None
    exclude_dirs: frozenset = frozenset()
    include_globs: tuple = ()
    exclude_globs: tuple = ()
    max_bytes: Optional[int] = None
    include_hidden: bool = True
    follow_symlinks: bool = False
    max_depth: Optional[int] = None
    presets: tuple = ()

    def dir_ok(self, name: str, rel: str) -> bool:
        if not self.include_hidden and name.startswith("."):
            return False
        low = name.lower()
        for pat in self.exclude_dirs:
            if _seg_match(low, pat):
                return False
        for g in self.exclude_globs:
            if _glob_match(g, rel, name):
                return False
        return True

    def file_ok(self, name: str, rel: str, ext: str) -> bool:
        if not self.include_hidden and name.startswith("."):
            return False
        if ext in self.exclude_exts:
            return False
        for g in self.exclude_globs:
            if _glob_match(g, rel, name):
                return False
        if self.exts is not None and ext not in self.exts:
            return False
        if self.include_globs and not any(
                _glob_match(g, rel, name) for g in self.include_globs):
            return False
        if self.dir_names is not None:
            parts = {p.lower() for p in rel.split(os.sep)[:-1]}
            if not (parts & self.dir_names):
                return False
        return True


def _seg_match(seg_low: str, pat: str) -> bool:
    p = pat.lower()
    if any(ch in p for ch in "*?["):
        return fnmatch.fnmatch(seg_low, p)
    return seg_low == p


def _glob_match(pattern: str, rel: str, name: str) -> bool:
    p = pattern.lower()
    if p.startswith("**/"):
        p = "*" + p[3:]
    rl = rel.lower().replace(os.sep, "/")
    nl = name.lower()
    return fnmatch.fnmatch(rl, p) or fnmatch.fnmatch(nl, p)


def _as_list(val) -> list:
    if val is None or val is True or val is False:
        return []
    if isinstance(val, (list, tuple, set, frozenset)):
        return list(val)
    if isinstance(val, str):
        return [v for v in (s.strip() for s in val.split(",")) if v]
    return [val]


def _norm_exts(val) -> set:
    out = set()
    for e in _as_list(val):
        e = str(e).strip().lower()
        if e:
            out.add(e if e.startswith(".") else "." + e)
    return out


def _norm_names(val) -> set:
    return {str(x).strip().lower() for x in _as_list(val) if str(x).strip()}


def _norm_globs(val) -> list:
    return [str(x).strip() for x in _as_list(val) if str(x).strip()]


def make_filters(preset=None, **kw) -> Filters:
    """Build :class:`Filters` from ONLY the caller's filters + any named presets.
    Nothing is injected silently. Presets merge (sets union). Unknown preset
    raises ValueError."""
    names = [str(p).lower() for p in _as_list(preset) if str(p).strip()]
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise ValueError(f"unknown preset(s) {unknown}; known: {sorted(PRESETS)}")

    exts, xexts, dirnames, xdirs = set(), set(), set(), set()
    iglobs, xglobs = [], []
    inc_hidden, follow, max_depth, max_bytes = True, False, None, None
    hidden_forced = False

    def _merge(src, is_user):
        nonlocal inc_hidden, follow, max_depth, max_bytes, hidden_forced
        exts.update(_norm_exts(src.get("exts", src.get("allowed_exts"))))
        xexts.update(_norm_exts(src.get("exclude_exts")))
        dirnames.update(_norm_names(src.get("dir_names", src.get("allowed_dirs"))))
        xdirs.update(_norm_names(src.get("exclude_dirs")))
        iglobs.extend(_norm_globs(src.get("include_globs", src.get("allowed_patterns"))))
        xglobs.extend(_norm_globs(src.get("exclude_globs", src.get("exclude_patterns"))))
        if "include_hidden" in src and src["include_hidden"] is not None:
            if is_user:
                inc_hidden = bool(src["include_hidden"])
                hidden_forced = True
            elif not hidden_forced:
                inc_hidden = inc_hidden and bool(src["include_hidden"])
        if src.get("follow_symlinks") is not None:
            follow = bool(src["follow_symlinks"])
        if src.get("max_depth") is not None:
            max_depth = int(src["max_depth"])
        if src.get("max_bytes") is not None:
            max_bytes = int(src["max_bytes"])

    for n in names:
        _merge(PRESETS[n], is_user=False)
    _merge(kw, is_user=True)

    return Filters(
        exts=frozenset(exts) or None,
        exclude_exts=frozenset(xexts),
        dir_names=frozenset(dirnames) or None,
        exclude_dirs=frozenset(xdirs),
        include_globs=tuple(iglobs),
        exclude_globs=tuple(xglobs),
        max_bytes=max_bytes,
        include_hidden=inc_hidden,
        follow_symlinks=follow,
        max_depth=max_depth,
        presets=tuple(names),
    )


def effective_filters(filt: Filters) -> dict:
    return {
        "presets": list(filt.presets),
        "exts": sorted(filt.exts) if filt.exts else None,
        "exclude_exts": sorted(filt.exclude_exts),
        "dir_names": sorted(filt.dir_names) if filt.dir_names else None,
        "exclude_dirs": sorted(filt.exclude_dirs),
        "include_globs": list(filt.include_globs),
        "exclude_globs": list(filt.exclude_globs),
        "include_hidden": filt.include_hidden,
        "follow_symlinks": filt.follow_symlinks,
        "max_depth": filt.max_depth,
        "max_bytes": filt.max_bytes,
    }


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------
def iter_files(roots, filt: Optional[Filters] = None, *,
               on_dir: Optional[Callable[[str], None]] = None,
               stats: Optional[dict] = None, preset=None, **kw) -> Iterator[str]:
    """Yield absolute file paths under ``roots``. With no filters, yields
    EVERYTHING. ``stats`` (if given) counts ``symlink_loop``."""
    if filt is None:
        filt = make_filters(preset=preset, **kw)
    if isinstance(roots, (str, os.PathLike)):
        roots = [roots]

    seen_dirs: set = set()
    for root in roots:
        root = os.fspath(root)
        if not root:
            continue
        root = os.path.abspath(root)
        if os.path.isfile(root):
            name = os.path.basename(root)
            ext = os.path.splitext(name)[1].lower()
            if filt.file_ok(name, name, ext):
                yield root
            continue
        if not os.path.isdir(root):
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            if on_dir is not None:
                on_dir(d)
            try:
                it = os.scandir(d)
            except OSError:
                continue
            with it:
                subdirs = []
                for entry in it:
                    try:
                        is_dir = entry.is_dir(follow_symlinks=filt.follow_symlinks)
                    except OSError:
                        continue
                    rel = os.path.relpath(entry.path, root)
                    if is_dir:
                        if filt.max_depth is not None and depth + 1 > filt.max_depth:
                            continue
                        if not filt.dir_ok(entry.name, rel):
                            continue
                        if filt.follow_symlinks:
                            try:
                                st = entry.stat()
                                key = (st.st_dev, st.st_ino)
                            except OSError:
                                continue
                            if key in seen_dirs:
                                if stats is not None:
                                    stats["symlink_loop"] = stats.get("symlink_loop", 0) + 1
                                continue
                            seen_dirs.add(key)
                        subdirs.append((entry.path, depth + 1))
                    else:
                        name = entry.name
                        ext = os.path.splitext(name)[1].lower()
                        if filt.file_ok(name, rel, ext):
                            yield entry.path
                stack.extend(reversed(subdirs))


def collect(roots, filt: Optional[Filters] = None, *, include_files: bool = True,
            on_dir: Optional[Callable[[str], None]] = None, preset=None,
            report: bool = False, **kw):
    """Return ``(dirs, files)`` under ``roots``; records :func:`last_report`.
    ``report=True`` -> ``((dirs, files), report)``."""
    if filt is None:
        filt = make_filters(preset=preset, **kw)
    skipped = _blank_skipped()
    files = list(iter_files(roots, filt, on_dir=on_dir, stats=skipped))
    dirs = sorted({os.path.dirname(f) for f in files})
    result = (dirs, ([] if not include_files else files))
    rep = _set_report({"op": "collect", "effective_filters": effective_filters(filt),
                       "scanned": len(files), "skipped": skipped})
    return (result, rep) if report else result


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def _read_reasoned(path: str, max_bytes: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    try:
        if max_bytes is not None and os.path.getsize(path) > max_bytes:
            return None, "too_large"
        with open(path, "rb") as f:
            head = f.read(_BINARY_SNIFF_BYTES)
            if b"\x00" in head:
                return None, "binary"
            rest = f.read()
    except OSError:
        return None, "unreadable"
    return (head + rest).decode("utf-8", "replace"), None


def read_text(path: str, max_bytes: Optional[int] = None) -> Optional[str]:
    """Decode a text file; ``None`` for binary/oversize/unreadable."""
    return _read_reasoned(path, max_bytes)[0]


def read_any_file(full_path: str, max_bytes: Optional[int] = None) -> str:
    """Text of a file (utf-8/replace). Missing -> FileNotFoundError; binary/
    oversize -> ValueError (callers catch OSError/ValueError/UnicodeError)."""
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"not a valid path: {full_path!r}")
    text, reason = _read_reasoned(full_path, max_bytes)
    if text is None:
        raise ValueError(f"not readable as text ({reason}): {full_path!r}")
    return text


# --------------------------------------------------------------------------
# Content search
# --------------------------------------------------------------------------
def _rg_prefilter(term, roots, case_sensitive):
    exe = shutil.which("rg")
    if not exe or not term or "\n" in term:
        return None
    cmd = [exe, "--files-with-matches", "--fixed-strings", "--null",
           "-a", "-uuu", "--no-messages"]
    if not case_sensitive:
        cmd.append("--ignore-case")
    cmd.append("--")
    cmd.append(term)
    cmd.extend(roots)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode not in (0, 1):
        return None
    return {os.path.abspath(p.decode("utf-8", "replace"))
            for p in proc.stdout.split(b"\x00") if p}


def stringInContent(content, strings, total_strings=False, normalize=False,
                    regex=False, **_kw) -> bool:
    if not content:
        return False
    if isinstance(strings, str):
        strings = [strings]
    strings = [s for s in strings if s]
    if not strings:
        return False
    c = str(content)
    if regex:
        flags = re.IGNORECASE if normalize else 0
        found = [s for s in strings if re.search(s, c, flags)]
    elif normalize:
        cl = c.lower()
        found = [s for s in strings if s.lower() in cl]
    else:
        found = [s for s in strings if s in c]
    if not found:
        return False
    return len(found) == len(strings) if total_strings else True


def find_lines(content, strings, total_strings=False, normalize=True,
               any_per_line=True, regex=False, max_lines=0, **_kw) -> list:
    if isinstance(strings, str):
        strings = [strings]
    strings = [s for s in strings if s]
    total = total_strings and not any_per_line
    out = []
    for i, line in enumerate(str(content).split("\n"), 1):
        if stringInContent(line, strings, total_strings=total, normalize=normalize,
                           regex=regex):
            out.append({"line": i, "content": line})
            if max_lines and len(out) >= max_lines:
                break
    return out


def getPaths(files, strings) -> tuple:
    if isinstance(strings, str):
        strings = [strings]
    strings = [s for s in strings if s]
    nu_files, found_paths = [], []
    for fp in files:
        text = read_text(fp)
        if text is None:
            continue
        low = text.lower()
        if not all(s.lower() in low for s in strings):
            continue
        nu_files.append(fp)
        found_paths.append({"file_path": fp,
                            "lines": find_lines(text, strings, normalize=True)})
    return nu_files, found_paths


def _match_lines_literal(text, terms, case_sensitive, max_lines) -> list:
    if case_sensitive:
        needles = list(terms)
        def hit(line):
            return any(t in line for t in needles)
    else:
        needles = [t.lower() for t in terms]
        def hit(line):
            low = line.lower()
            return any(t in low for t in needles)
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        if hit(line):
            out.append({"line": i, "content": line})
            if max_lines and len(out) >= max_lines:
                break
    return out


def search_content(roots, strings=(), *, filt: Optional[Filters] = None,
                   preset=None, reader=None, get_lines=True, total_strings=True,
                   case_sensitive=False, regex=False, limit=None, max_lines=50,
                   use_rg=False, threads=None, on_dir=None, report=False, **kw):
    """Native search entry point (see :func:`find_content`). Records
    :func:`last_report`; ``report=True`` -> ``(result, report)``."""
    if filt is None:
        filt = make_filters(preset=preset, **kw)
    read = reader or (lambda p: _read_reasoned(p, filt.max_bytes))
    roots = _as_roots(roots) or [os.getcwd()]
    terms = [str(s) for s in strings if str(s) != ""]

    skipped = _blank_skipped()
    files = list(iter_files(roots, filt, on_dir=on_dir, stats=skipped))
    if not terms:
        rep = _set_report({"op": "search", "effective_filters": effective_filters(filt),
                           "scanned": len(files), "matched": 0, "skipped": skipped})
        return ([], rep) if report else []

    if (not regex) and use_rg and len(terms) == 1:
        rg_hits = _rg_prefilter(terms[0], roots, case_sensitive)
        if rg_hits is not None:
            files = [f for f in files if f in rg_hits]

    lock = threading.Lock()

    def bump(reason):
        if reason:
            with lock:
                skipped[reason] = skipped.get(reason, 0) + 1

    if regex:
        compiled = [re.compile(t, 0 if case_sensitive else re.IGNORECASE) for t in terms]
        check = _regex_checker(read, compiled, get_lines, total_strings, max_lines, bump)
    else:
        check = _literal_checker(read, terms, get_lines, total_strings,
                                 case_sensitive, max_lines, bump)
    out = _run(files, check, limit, threads)
    rep = _set_report({"op": "search", "effective_filters": effective_filters(filt),
                       "scanned": len(files), "matched": len(out), "skipped": skipped})
    result = out if get_lines else [h["file_path"] for h in out]
    return (result, rep) if report else result


def _literal_checker(read, terms, get_lines, total_strings, case_sensitive, max_lines, bump):
    needles = terms if case_sensitive else [t.lower() for t in terms]

    def check(path):
        if should_stop():
            return None
        text, reason = read(path)
        if text is None:
            bump(reason)
            return None
        hay = text if case_sensitive else text.lower()
        ok = all(n in hay for n in needles) if total_strings else any(n in hay for n in needles)
        if not ok:
            return None
        if not get_lines:
            return {"file_path": path, "lines": []}
        return {"file_path": path,
                "lines": _match_lines_literal(text, terms, case_sensitive, max_lines)}
    return check


def _regex_checker(read, compiled, get_lines, total_strings, max_lines, bump):
    def check(path):
        if should_stop():
            return None
        text, reason = read(path)
        if text is None:
            bump(reason)
            return None
        present = [rx for rx in compiled if rx.search(text)]
        ok = (len(present) == len(compiled)) if total_strings else bool(present)
        if not ok:
            return None
        if not get_lines:
            return {"file_path": path, "lines": []}
        lines = []
        for i, line in enumerate(text.split("\n"), 1):
            if any(rx.search(line) for rx in compiled):
                lines.append({"line": i, "content": line})
                if len(lines) >= max_lines:
                    break
        return {"file_path": path, "lines": lines}
    return check


def _run(files, check, limit, threads) -> list:
    out = []
    if threads and threads > 1 and len(files) > 4:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for res in ex.map(check, files):
                if res is not None:
                    out.append(res)
                    if limit is not None and len(out) >= limit:
                        break
    else:
        for f in files:
            res = check(f)
            if res is not None:
                out.append(res)
                if limit is not None and len(out) >= limit:
                    break
    out.sort(key=lambda d: d["file_path"])
    return out


def _as_roots(val) -> list:
    if not val:
        return []
    if isinstance(val, (str, os.PathLike)):
        return [os.path.abspath(os.fspath(val))]
    if isinstance(val, (list, tuple, set)):
        out = []
        for v in val:
            out.extend(_as_roots(v))
        seen, uniq = set(), []
        for r in out:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return uniq
    return [os.path.abspath(str(val))]


def _collect_roots(directory, args) -> list:
    roots = []
    for a in args:
        roots.extend(_as_roots(a))
    roots.extend(_as_roots(directory))
    seen, uniq = set(), []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq or [os.getcwd()]


# --------------------------------------------------------------------------
# abstract_search-compatible drop-in API
# --------------------------------------------------------------------------
def find_content(*args, directory=None, strings: Sequence[str] = (),
                 parse_lines: bool = False, get_lines: bool = True,
                 total_strings: bool = True, spec_line=False, structured: bool = False,
                 diffs: bool = False, case_sensitive: bool = False, regex: bool = False,
                 limit: Optional[int] = None, max_lines: int = 50, use_rg: bool = False,
                 threads: Optional[int] = None, cfg: Optional[Filters] = None,
                 filt: Optional[Filters] = None, preset=None, report: bool = False,
                 on_dir=None, **kw) -> list:
    """Drop-in for ``abstract_search.findContent`` (NO implicit filtering; opt in
    with ``preset=``). Returns ``[{"file_path","lines":[...]}, ...]`` (get_lines)
    or a list of paths. ``report=True`` -> ``(result, report)``; else see
    :func:`last_report`."""
    filt = filt or (cfg if isinstance(cfg, Filters) else None) or make_filters(preset=preset, **kw)
    roots = _collect_roots(directory, args)
    return search_content(roots, strings, filt=filt, get_lines=get_lines,
                          total_strings=total_strings, case_sensitive=case_sensitive,
                          regex=regex, limit=limit, max_lines=max_lines, use_rg=use_rg,
                          threads=threads, on_dir=on_dir, report=report)


findContent = find_content


def get_file_filters(*args, preset=None, **kw):
    """Return ``(directories, cfg, allowed, include_files, recursive)``. ``cfg``
    is a :class:`Filters` reflecting ONLY user filters + any ``preset`` (no
    silent defaults); inspect via :func:`effective_filters`."""
    filt = make_filters(preset=preset, **kw)
    directories = _collect_roots(kw.get("directory"), args)
    recursive = bool(kw.get("recursive", True))
    include_files = bool(kw.get("include_files", True))

    def allowed(path: str) -> bool:
        for d in directories:
            try:
                rel = os.path.relpath(path, d)
            except ValueError:
                continue
            if rel.startswith(".."):
                continue
            name = os.path.basename(path)
            if os.path.isdir(path):
                return filt.dir_ok(name, rel)
            ext = os.path.splitext(name)[1].lower()
            return filt.file_ok(name, rel, ext)
        return False

    return directories, filt, allowed, include_files, recursive


def get_files_and_dirs(*args, directory=None, cfg: Optional[Filters] = None,
                       recursive: bool = True, include_files: bool = True,
                       preset=None, report: bool = False, **kw):
    """Return ``(dirs, files)`` under the roots. NO implicit filtering; opt in
    with ``preset=``. ``report=True`` -> ``((dirs, files), report)``; else see
    :func:`last_report`. ``recursive=False`` -> depth 0."""
    filt = cfg if isinstance(cfg, Filters) else make_filters(preset=preset, **kw)
    if not recursive:
        filt = Filters(**{**filt.__dict__, "max_depth": 0})
    roots = _collect_roots(directory, args)
    return collect(roots, filt, include_files=include_files, report=report)


# --------------------------------------------------------------------------
# Provider seam
# --------------------------------------------------------------------------
class LocalSearchProvider:
    """Duck-typed ``local_search`` adapter backed by this module. Stateless."""
    find_content = staticmethod(find_content)
    get_file_filters = staticmethod(get_file_filters)
    get_files_and_dirs = staticmethod(get_files_and_dirs)
    read_any_file = staticmethod(read_any_file)
    last_report = staticmethod(last_report)


_PROVIDER = LocalSearchProvider()


def provider() -> LocalSearchProvider:
    """Zero-argument factory returning the built-in provider."""
    return _PROVIDER

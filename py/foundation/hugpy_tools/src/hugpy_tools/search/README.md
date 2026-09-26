# `hugpy_tools.search` — fast, self-contained local content search

Stdlib-only. Imports nothing from the rest of hugpy. Give it roots + terms; it
returns line-level hits. It replaces the need for `abstract_search` /
`abstract_utilities` in the agent's filesystem search path: same behaviour and
API, correct include/exclude, materially faster.

## Why

`abstract_search` shells out to Unix `find` (a subprocess per query, no
directory pruning) and its include/exclude filters are wrong (see
"abstract_search bugs" at the bottom). This module walks with `os.scandir`,
prunes excluded directories at walk time (they are never entered), filters by
extension before any `stat`/`open`, and matches literals on bytes — decoding
only the lines it needs.

## NO implicit filtering

A search engine must not allow/exclude anything silently. **With no user
filters, you get EVERYTHING under the roots** — dotfiles, `node_modules`,
`.venv`, binaries. There is no default extension whitelist, no default excluded
dirs, no default hidden-file skipping.

Opt into the old conveniences EXPLICITLY with named **presets**:

| preset | applies |
|---|---|
| `"code"` | `CODE_TEXT_EXTS` allow-list + `EXCLUDE_NOISE_DIRS` + skip hidden |
| `"noise"` | `EXCLUDE_NOISE_DIRS` + `BINARY_EXTS` excludes + skip hidden |
| `"text"` | `BINARY_EXTS` + `EXCLUDE_NOISE_DIRS` excludes + skip hidden |

`CODE_TEXT_EXTS` / `EXCLUDE_NOISE_DIRS` / `BINARY_EXTS` are exported; pass them
as filter kwargs if you want.

**Preset merge = UNION (never be surprised).** Presets and your kwargs are
unioned field-by-field. Good for excludes; but the allow-whitelist is unioned
too, so a whitelist preset **broadens**: `preset="code"` + `allowed_exts=[".rs"]`
matches code/text **and** `.rs`, not just `.rs`. To **narrow** by extension use
`preset="noise"` (excludes only) + your own `allowed_exts`, or no preset.
Everything applied shows in the report's `effective_filters`.

Nothing is dropped silently. Every `find_content` / `get_files_and_dirs` /
`search_content` / `collect` records **`last_report()`**: `effective_filters`
(exactly what applied, incl. presets), `scanned`, `matched` (search) and
`skipped` counts by reason (`binary` / `unreadable` / `too_large` /
`symlink_loop`). Content search still skips binary files (a NUL-byte match is
meaningless) — but reports them. Pass `report=True` for `(result, report)`
inline.

## Include / exclude semantics (when you DO filter)

- **Exclude always wins.** Anything matched by an exclude is gone, even if an
  include also matches it.
- **Include narrows.** No include ⇒ everything not excluded is in scope.
- **Directory excludes** match an exact **path segment** and are **pruned**
  during the walk: `build` drops `.../build/...` but never `.../rebuild/...`,
  and the excluded subtree is never descended.
- **Globs** (`include_globs` / `exclude_globs`, aliases `allowed_patterns` /
  `exclude_patterns`) match with `fnmatch` against **both** the root-relative
  path **and** the basename, so `**/mct/*.py` and `*session*` both work (`*`
  crosses `/`; a leading `**/` is normalised to `*`).
- **Extensions** (`exts` / `exclude_exts`) compare lower-cased, with a leading
  dot.
- **Case:** name/glob/extension matching is case-insensitive. Literal content
  matching is case-insensitive unless `case_sensitive=True`.
- **Symlinks** are not followed by default (least surprising; loops avoided).
  `max_depth` / `max_bytes` are explicit params (defaults: unbounded, no cap).

`__init__.py` is never treated specially (the old `abstract_search` dropped
every `__init__.py` by default — one of the bugs this replaced).

```python
from hugpy_tools.search import get_files_and_dirs, find_content, last_report

files = get_files_and_dirs(directory="/srv/app")[1]          # EVERYTHING
files = get_files_and_dirs(directory="/srv/app", preset="code")[1]   # code/text, no noise
hits, rep = find_content(directory="/srv/app", strings=["x"], preset="noise", report=True)
rep["effective_filters"]; rep["skipped"]                     # or last_report()
```

## Examples

```python
from hugpy_tools.search import find_content, iter_files, make_filters, read_any_file

# 1. Enumerate .py files (node_modules etc. only pruned if you ask — preset/exclude).
files = list(iter_files("/srv/app", make_filters(allowed_exts=[".py"], preset="noise")))

# 2. Literal search (case-insensitive). Returns
#    [{"file_path": str, "lines": [{"line": int, "content": str}, ...]}, ...]
hits = find_content(directory="/srv/app", strings=["assure_model_key"])

# 3. Intersection: files containing BOTH terms (total_strings defaults True).
hits = find_content(directory="/srv/app", strings=["nginx", "ssl"])

# 4. Path-scoped: only .py under any mct/ directory.
hits = find_content(directory="/srv/app", strings=["submit_pull"],
                    allowed_patterns=["**/mct/*.py"])

# 5. Regex.
hits = find_content(directory="/srv/app", strings=[r"def\s+\w+\("], regex=True)

# 6. Read one file as text (binary/oversize -> ValueError; missing -> FileNotFoundError).
text = read_any_file("/srv/app/README.md")
```

### Directive helpers used by the steward

`get_file_filters(root, **kw) -> (dirs, cfg, allowed, include_files, recursive)`
and `get_files_and_dirs(*, directory, cfg, recursive) -> (dirs, files)` mirror
`abstract_search`'s shapes so callers unpack them unchanged. `all` / `any` /
`none` directive logic lives in the caller (e.g. hugpy-agent's
`mct/session.py`): `all` = intersection, `any` = union, `none` = a file-level
veto (a file with an excluded term is dropped, not merely demoted).

## Speed

`use_rg` (default **False**) enables an optional ripgrep prefilter. It is a
*pure* prefilter — ripgrep only narrows which files the Python line-matcher
opens, so results are byte-for-byte identical to the pure-Python path (tested).
It is off by default because the pure-Python pruned walk benchmarked **faster**
on real trees (ripgrep run with `-uuu` rescans ignored/venv subtrees the walk
prunes, and its subprocess cost rarely pays off); turn it on only for very large
trees scanned with a highly selective term.

## Benchmark (this machine, 2026-09-24)

Enumerate `*.py` (excl. node_modules/build), literal `import`, regex
`def\s+\w+\(`. Wall time, best-of-2.

| tree | op | hugpy_tools | abstract_search | speedup |
|---|---|---|---|---|
| `/srv/hugpy/src/hugpy` | enumerate | 10.6 ms (n=1133) | 210 ms (n=863) | ~20x |
| | literal | 86 ms (n=1106) | 422 ms (n=860) | ~5x |
| | regex | 43 ms (n=1011) | 266 ms (n=820) | ~6x |
| `/srv/pyit/dev` | enumerate | 35 ms (n=3347) | 455 ms (n=4566) | ~13x |
| | literal | 114 ms (n=3205) | 1000 ms (n=4329) | ~9x |
| | regex | 108 ms (n=1914) | 853 ms (n=3377) | ~8x |

`abstract_utilities` delegates its search to `abstract_search`, so its numbers
track `abstract_search`'s (1010 ms literal on `/srv/pyit/dev`).

**Count differences are `abstract_search` bugs, not misses:**
- `hugpy_tools` finds the `__init__.py` files `abstract_search` drops (its
  default `__init__*` exclude-pattern + `__init__` exclude-dir): 270 real source
  files on the first tree, ~1010 on the second.
- `abstract_search` returns thousands of `.venv` / `site-packages` /
  hidden-backup files (2229 on `/srv/pyit/dev`) that `hugpy_tools` prunes by
  default — and pays for descending them.

## abstract_search bugs (for reference; not fixed here)

Paths under `/srv/pyit/dev/abstract_search/src/abstract_search/`:
- `find_collect.py:89-91` — excluded dirs become `find ... ! -path '*d*'`: a
  **substring** match on the full path (so `build` also drops `rebuild`,
  `prebuilder`) that is a result **filter, not a `-prune`**, so `find` still
  descends the excluded tree (slow).
- `find_collect.py:93-99` & `filters.py:296-301` — include/exclude *patterns*
  match the **basename only** (`-name` / `fnmatch(name, …)`), so path globs like
  `**/mct/session.py` match nothing.
- `constants.py:98,102` — defaults exclude the `__init__` dir **and** the
  `__init__*` pattern, silently dropping every `__init__.py`.
- `filters.py:154-169` (`ensure_patterns`) — a pattern with no `*`/`?` is
  rewritten to a prefix/suffix glob, so an intended exact match (`session.py`)
  becomes `session.py*`.
- `find_content.py:35-44,186` (`_normalize`) — final line matching strips `//`
  comments and everything after them, so a term inside a comment or after
  `http://` never matches, even though the file passed the prefilter.
- `find_content.py:103,173` — each matched file is read from disk **twice**
  (once in `getPaths`, once in the match loop).
- `find_content.py:108` (`getPaths`) — literal prefilter uses
  `tot_strings not in og_content`: **case-sensitive**, which is why callers had
  to retry every term in both cases.

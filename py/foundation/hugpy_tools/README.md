# hugpy-tools

`hugpy_tools` — a slim, **stdlib-only** capability suite for autonomous agents,
extracted from `abstract_utilities` / `abstract_webtools` and rebuilt with no
third-party runtime dependency. Foundation layer: it imports no other `hugpy_*`
package, so it can be lifted out as its own distribution unchanged.

Ownership and allowed dependencies are declared in `py/partition.toml`; see
`PARTITION.md` at the workspace root.

Allowed Python dependencies inside the ecosystem: **none** (stdlib only).

## Built-in tools

Every `fs` function takes an **already-confined** absolute path — call
`paths.confine(root, path)` first to enforce a workspace jail (rejects `..`,
absolute-outside, and symlink escapes in one check).

| Module | Function | What it does |
| --- | --- | --- |
| `paths` | `confine(root, path, for_write=False)` | Resolve a path under `root`; raise `PathEscape` on any escape. |
| `paths` | `file_parts(path)` | Decompose into dir/base/name/ext + two enclosing dir levels. |
| `paths` | `sanitize_filename(name)` | Reduce to one safe path component, length-bounded. |
| `fs` | `read_text(path, max_bytes=None)` | Read with encoding detection (BOM → utf-8 → cp1252). |
| `fs` | `read_lines(path, start, end, number=False)` | 1-based inclusive line-range read. |
| `fs` | `atomic_write(path, content, append=False)` | Durable write (temp file + `os.replace`); append supported. |
| `fs` | `edit_replace(path, old, new, count=None)` | Exact string replace with an occurrence-count contract; atomic. |
| `fs` | `file_info(path, hash_files=False)` | Size, mtime, type, symlink flag, optional sha256. |
| `fs` | `list_dir(path, glob=None, ...)` | Immediate entries (dirs first), optional fnmatch. |
| `fs` | `tree(path, max_depth=3, max_entries=500)` | Bounded recursive tree text (never follows dir symlinks). |
| `hashkit` | `sha256_text` / `sha256_file` / `quick_hash` | Content hashing (files streamed in 1 MiB windows). |
| `text` | `count_tokens(text)` | Dependency-free approximate BPE token count. |
| `text` | `chunk_by_lines` / `chunk_by_tokens` | Chunk on line or token budgets (paragraph-aware). |
| `text` | `unified_diff(before, after)` | Unified diff between two texts. |
| `data` | `read_data(path, fmt=None)` | Parse JSON always, TOML via stdlib `tomllib`, YAML if PyYAML present. |
| `data` | `write_json(path, obj)` / `safe_json_dumps(obj)` | Atomic JSON write / non-exploding dumps. |
| `timekit` | `now_iso` / `now_epoch` / `epoch_to_iso` / `iso_to_epoch` | UTC time conversions. |
| `web` | `assess_webpage(url, ...)` | assessManager-parity structured page assessment over urllib + `html.parser`. |
| `web` | `prescreen_webpage(url, ...)` | Cheap title + description + lede relevance pre-screen. |

### Web assessment

`assess_webpage` returns the same dict shape as
`abstract_webtools.assessManager.assess_webpage`
(`{url, title, description, text, metadata, jsonld, links, truncated, render}`)
plus additive keys `final_url`, `status`, `canonical`, `lang`, `headings`, and
`error`. The fetch is http(s)-only, refuses redirects to other schemes,
disables `file://` / `ftp://`, caps the body, times out, handles gzip/deflate,
and detects charset (Content-Type → BOM → `<meta charset>` → utf-8 → cp1252).

JS rendering is opt-in and lazy: `force_render=True` (or an automatic fall-back
when the cheap fetch yields almost no text) drives a headless browser **only if
Playwright or Selenium is installed**, and raises a clear "install X" error
otherwise. The default path never needs a browser.

## Layer

Foundation. `hugpy_tools` imports no other `hugpy_*` package. The `search`
subpackage is owned by a separate work-stream.

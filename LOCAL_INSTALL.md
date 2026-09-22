# Local install

How to get a working, importable Hugpy from this checkout without publishing
anything. Everything is installed **editable** (`pip install -e`) so edits in
`py/*/hugpy_*` are live in the interpreter.

```bash
./local_install.sh                          # ./.venv, all 13 hugpy-* + compat shell
./local_install.sh --venv /tmp/hugpy-local-venv --check
./local_install.sh --extras server          # hugpy[server]: the central/coordinator box
./local_install.sh --extras worker          # a GPU-less worker box
./local_install.sh --check --no-install     # verify an existing venv only
```

`local_install.sh` picks the newest `python3.x` on PATH (override with
`PYTHON=/path/to/python`) and runs `py/local_install.py`, which is stdlib-only
and works on Python 3.10+.

## What it does

1. Creates the virtualenv if it does not exist (`python -m venv --upgrade-deps`).
   A non-venv interpreter is refused: the dev box's miniconda still carries the
   retired monolith `abstract_hugpy_dev` 0.1.266, which silently satisfies old
   imports and hides partition mistakes. `--allow-system` overrides.
2. Reads `py/partition.toml` and runs **one** `pip install -e` for all thirteen
   distributions in cut order, plus `py/compat/abstract_hugpy_dev` unless
   `--no-compat`. One invocation is what makes `hugpy-engine`'s dependency on
   `hugpy-storage` resolve from the checkout instead of PyPI. Third-party
   dependencies (Flask, abstract-*, huggingface_hub, ...) still come from PyPI.
3. With `--check`, in the venv interpreter: imports every `hugpy_*` package and
   confirms it resolves to this checkout; confirms `abstract_hugpy_dev` is the
   compat shell (0.1.267.dev0 under `py/compat`) and not a stale monolith;
   imports `hugpy_server.wsgi_app` and `hugpy_server.wiring`; then runs
   `abstract-hugpy-dev-check`, which imports every aliased old module path and
   fails on any that does not resolve to the same object as its new home
   (modules needing an uninstalled extra such as `discord.py` are reported as
   SKIP, not failures).

## Extras

`--extras` applies to the `hugpy` meta package; see
`py/meta/hugpy/pyproject.toml` for the full list. The box profiles are
`worker`, `cpu-worker`, `gpu-worker`, `server` and `all`; capability extras
(`gguf`, `transformers`, `media`, `video`, `bot`, ...) can be combined:
`--extras server,gpu`.

## Subsets

`--only platform,control,storage` installs just those ids (ids are the
`[[package]]` ids in `py/partition.toml`, plus `compat`). The check then covers
only that subset.

## Offline / air-gapped

`--pip-arg=--no-index --pip-arg=--find-links=/path/to/wheels` passes through to
pip. Editable builds need `setuptools>=77` available to the build isolation
step, so include it in the wheelhouse or add `--pip-arg=--no-build-isolation`.

## Test venv conventions

The workspace's own `.venv` is exactly this install (no torch). Package test
suites run strict, with the monolith blocked by each package's `conftest.py`:

```bash
for d in py/*/hugpy_* py/meta/hugpy py/compat/abstract_hugpy_dev; do
  (cd "$d" && ../../../.venv/bin/python -m pytest -q --timeout=120)
done
python py/validate_partition.py --edges     # must report 0 forbidden edges
```

## Live host switch (later, see HANDOFF.md section 5)

On the host, from the workspace checkout: `./local_install.sh --venv
/srv/hugpy/venv --extras server --check`, then move the systemd units that now
live in the package `deploy/` directories and restart. Nothing here publishes
to PyPI, npm, GitHub or Hugging Face.

# Contributing

hugpy is source-available. By opening a pull request you license your
contribution under the repository `LICENSE`; no separate agreement is needed.

## Ground rules

1. One package, one owner. Find the owning distribution in `py/partition.toml`
   before adding code; cross-package needs go through the seams listed in
   `py/WIRING.md`, never through new imports.
2. `python py/validate_partition.py --edges` must stay green. It fails on any
   import the manifest forbids, on wildcard imports, and on dependency cycles.
3. Every package must import with the monolith and its optional dependencies
   blocked (`tests/test_import_policy.py` in each package checks this).
4. Tests live with the code they exercise and run in strict mode:

   ```bash
   cd py/<route>/<package>
   python -m build --wheel
   pytest -q --timeout=120
   ```

5. Heavy dependencies (torch, diffusers, llama.cpp, whisper, discord) are
   extras and are imported lazily inside the function that uses them.

## Releases

Follow the checklist in `ECOSYSTEM.md` section C.5. Tags are
`<distribution>/v<version>`, for example `hugpy-engine/v0.2.0`; the meta
distribution `hugpy` is released last.

## Reporting problems

Bugs and feature requests: https://github.com/hugpy/hugpy/issues.
Security issues: see `SECURITY.md`.

# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package uses
[Semantic Versioning](https://semver.org/). Dates are the npm publish dates
(`npm view <name> time`); entries before the monorepo carry no per-release
detail beyond "published".

## [Unreleased]

### Changed
- Moved into the `hugpy/hugpy` monorepo as an npm workspace package
  (`react/agents_ui`); `repository.directory` points there. The source
  imports the shared Help widget and navbar links from the sibling
  `react/ui_shared` package by relative path.
- Publishing is tag-triggered only (`npm-agents-ui-v<version>` via
  `.github/workflows/npm-publish.yml`), never `npm publish` by hand.
- `files` now includes `dist` (built with `base: "/fleet/"`), so the
  tarball ships the built bundle; earlier published tarballs did not.

## [0.3.0] - 2026-08-27
- Published.

## [0.1.0] - 2026-08-13
- Published.

[Unreleased]: https://github.com/hugpy/hugpy/tree/main/react/agents_ui
[0.3.0]: https://www.npmjs.com/package/@hugpy/agents-ui/v/0.3.0
[0.1.0]: https://www.npmjs.com/package/@hugpy/agents-ui/v/0.1.0

# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package uses
[Semantic Versioning](https://semver.org/). Dates are the npm publish dates
(`npm view <name> time`); entries before the monorepo carry no per-release
detail beyond "published".

## [Unreleased]

### Added
- `npm test` runs the existing vitest suite (`src/**/*.test.ts`: transport
  client and telemetry, schemas, upload normalisation and helpers) in a Node
  environment; `vitest` is a devDependency and CI runs the suite after the
  build.

### Changed
- Moved into the `hugpy/hugpy` monorepo as an npm workspace package
  (`react/media_intelligence_ui`); `repository.directory` points there. The
  source imports the shared Help widget and navbar links from the sibling
  `react/ui_shared` package by relative path.
- Publishing is tag-triggered only (`npm-media-intelligence-ui-v<version>`
  via `.github/workflows/npm-publish.yml`), never `npm publish` by hand.
- `files` now includes `dist` (built with `base: "/media/"`), so the
  tarball ships the built bundle; earlier published tarballs did not.

## [0.3.0] - 2026-08-27
- Published.

## [0.1.0] - 2026-08-13
- Published. The console wiring it packages is described in
  `docs/CHANGES-media-intelligence-2026-06-25.md`.

[Unreleased]: https://github.com/hugpy/hugpy/tree/main/react/media_intelligence_ui
[0.3.0]: https://www.npmjs.com/package/@hugpy/media-intelligence-ui/v/0.3.0
[0.1.0]: https://www.npmjs.com/package/@hugpy/media-intelligence-ui/v/0.1.0

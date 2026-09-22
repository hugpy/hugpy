# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package uses
[Semantic Versioning](https://semver.org/). Dates are the npm publish dates
(`npm view <name> time`); entries before the monorepo carry no per-release
detail beyond "published".

## [Unreleased]

### Added
- `package-lock.json`, so the package installs reproducibly with
  `npm ci --workspaces=false` like the other arms.

### Changed
- Moved into the `hugpy/hugpy` monorepo as an npm workspace package
  (`react/video_intelligence_ui`); `repository.directory` points there.
- Publishing is tag-triggered only (`npm-video-intelligence-ui-v<version>`
  via `.github/workflows/npm-publish.yml`), never `npm publish` by hand.
- `files` now includes `dist` (built with `base: "/video/"`), so the
  tarball ships the built bundle; earlier published tarballs did not.

## [0.3.2] - 2026-08-27
- Published. Demo media (`?demo=1`) is hosted, not bundled: resolved from
  `?mediaBase=`, `window.__HUGPY_MEDIA_BASE__`, `VITE_HUGPY_DEMO_MEDIA_BASE`
  or the default `https://hugpy.ai/demo-media` (see README).

## [0.3.1] - 2026-08-27
- Published.

## [0.3.0] - 2026-08-27
- Published.

## [0.1.1] - 2026-08-13
- Published.

## [0.1.0] - 2026-08-13
- Published. The four stations (image crop, audio crop, frames/models,
  generate) and their build order are recorded in `docs/ROADMAP.md`.

[Unreleased]: https://github.com/hugpy/hugpy/tree/main/react/video_intelligence_ui
[0.3.2]: https://www.npmjs.com/package/@hugpy/video-intelligence-ui/v/0.3.2
[0.3.1]: https://www.npmjs.com/package/@hugpy/video-intelligence-ui/v/0.3.1
[0.3.0]: https://www.npmjs.com/package/@hugpy/video-intelligence-ui/v/0.3.0
[0.1.1]: https://www.npmjs.com/package/@hugpy/video-intelligence-ui/v/0.1.1
[0.1.0]: https://www.npmjs.com/package/@hugpy/video-intelligence-ui/v/0.1.0

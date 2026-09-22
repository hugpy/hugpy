# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package uses
[Semantic Versioning](https://semver.org/). Dates are the npm publish dates
(`npm view <name> time`); entries before the monorepo carry no per-release
detail beyond "published".

## [Unreleased]

### Changed
- Moved into the `hugpy/hugpy` monorepo as an npm workspace package
  (`react/ui`); `repository.directory` points there.
- Publishing is tag-triggered only (`npm-ui-v<version>` via
  `.github/workflows/npm-publish.yml`), never `npm publish` by hand.
- `files` now includes `dist` (the built console, with the arms bundled by
  `postbuild`) alongside `dist-lib`, so the tarball ships the built bundle;
  earlier published tarballs did not include it.

## [0.3.0] - 2026-08-27
- Published.

## [0.2.2] - 2026-08-13
- Published.

## [0.2.1] - 2026-08-13
- Published.

## Earlier
- `0.2.0` (2026-07-21) and `0.1.0` (2026-06-15) appear in the registry's
  publish log but are no longer listed as installable versions (unpublished).

[Unreleased]: https://github.com/hugpy/hugpy/tree/main/react/ui
[0.3.0]: https://www.npmjs.com/package/@hugpy/ui/v/0.3.0
[0.2.2]: https://www.npmjs.com/package/@hugpy/ui/v/0.2.2
[0.2.1]: https://www.npmjs.com/package/@hugpy/ui/v/0.2.1

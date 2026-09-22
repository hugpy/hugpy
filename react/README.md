# @hugpy

Front-ends and desktop tooling for hugpy, the self-hosted LLM console.

| Package | Mount | Purpose |
|---|---|---|
| @hugpy/ui | / | Embeddable console: chat, models, workers, API keys, Discord |
| @hugpy/agents-ui | /fleet | Agent fleet console for hugpy-agent nodes |
| @hugpy/media-intelligence-ui | /media | Transcription, summaries, embeddings, vision, documents |
| @hugpy/video-intelligence-ui | /video | Video stations over the hugpy-video job bus |
| @hugpy/ui-shared | – | Help widget and navbar links, zero dependencies |
| @hugpy/station | – | Desktop cockpit (Electron); installers at https://hugpy.ai/station |

All packages talk to a same-origin hugpy-server (`/api`). Build each package
with `npm run build`; `hugpy-server` mounts the built bundles as package data
(see `py/services/hugpy_server/BUILD_CONSOLE.md`). License: see LICENSE in
each package.

`@hugpy/console` and `@hugpy/vm-mgr` are published on npm but have no source in
this repository; locate their source before any further release (ECOSYSTEM.md B.2, D.5).

## Workspace

`react/package.json` is a private npm workspace root listing the five packages
(`ui_shared` first). `npm run build` here builds every package in order
(`ui_shared` has no build step; the three arms; then `ui`, whose `postbuild`
also rebuilds the arms into `ui/dist/<arm>/`). `npm run build:ui` and friends
build one package. There is no root `test` script: the media arm has
`*.test.ts` files written for vitest, but vitest is not a dependency yet.

There is no root lockfile. Each package keeps its own `package-lock.json`, so
install inside a package with `npm ci --workspaces=false` (a plain `npm ci`
there looks for `react/package-lock.json` and fails; a plain `npm install`
writes a root lockfile and hoists into `react/node_modules`).

The apps import `@hugpy/ui-shared` by relative path (`../ui_shared/...`), not
as a workspace dependency. Switching would need a root `npm install` to create
the `react/node_modules/@hugpy/ui-shared` link. That replaces the per-package
lockfiles and `node_modules` with a hoisted tree, and the extensionless imports
would have to become `@hugpy/ui-shared/help/helpWidget.js` to match its `exports`.
`react/agents_ui/ui_shared/` is an empty leftover directory; nothing imports it.

## Publishing

Every package has its own semver. Publishing happens only through
`.github/workflows/npm-publish.yml`, never by running `npm publish` by hand.
Push a tag named `npm-<pkg>-v<version>`, where `<pkg>` is the npm name without
the scope and `<version>` equals `version` in that package's `package.json`:

| Tag | Package |
|---|---|
| `npm-ui-v0.3.1` | `@hugpy/ui` (`react/ui`) |
| `npm-ui-shared-v0.3.1` | `@hugpy/ui-shared` (`react/ui_shared`) |
| `npm-agents-ui-v0.3.1` | `@hugpy/agents-ui` (`react/agents_ui`) |
| `npm-media-intelligence-ui-v0.3.1` | `@hugpy/media-intelligence-ui` (`react/media_intelligence_ui`) |
| `npm-video-intelligence-ui-v0.3.3` | `@hugpy/video-intelligence-ui` (`react/video_intelligence_ui`) |

The workflow checks the tag against `package.json`, runs `npm ci` and
`npm run build`, then runs `npm publish --access public --provenance` (OIDC
`id-token: write`, `NPM_TOKEN` secret, `npm-publish` environment). Publish
`ui-shared` first. A tag is pushed only after an explicit go for that release
(ECOSYSTEM.md D.5); bump the version (and add a `CHANGELOG.md` entry per
ECOSYSTEM.md C.1; the React packages have none yet) in a reviewed commit
before tagging. After a release that ships `dist/`, update the pinned
version in `py/services/hugpy_server/console_manifest.json`.

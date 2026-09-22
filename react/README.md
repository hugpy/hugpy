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

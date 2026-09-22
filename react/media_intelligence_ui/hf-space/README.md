---
title: Media Intelligence
emoji: 💬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
license: other
license_name: hugpy-source-available
license_link: LICENSE
short_description: Your Models, Your Data
tags:
  - media-intelligence
  - chat
  - multimodal
  - whisper
  - transcription
  - summarization
  - embeddings
  - vision
  - document-analysis
  - llm
  - self-hosted
---

# Media Intelligence

A live mirror of hugpy's **media-intelligence** arm: chat that summarizes,
extracts keywords, analyzes documents/URLs, and more — driven by hugpy's
self-hosted, GPU-pooling inference (not Hugging Face compute).

This Space is a **thin nginx reverse-proxy** to the live arm — no app bundle is
built here, so it always reflects the current build with nothing to re-ship. The
browser stays on this origin; nginx proxies the UI and API server-side.

- Built on [`abstract-hugpy`](https://pypi.org/project/abstract-hugpy/)
- Full project: [hugpy.ai](https://hugpy.ai) · by [John R. Putkey](https://jrputkey.com)

> Interim showcase wiring — a better medium/fallback is planned.

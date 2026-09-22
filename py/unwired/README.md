# Unwired code parked during the partition

Copies of code that nothing reachable calls (or that cannot run without live
services), moved here instead of being shimmed alive. One line per entry.

- `video/_selftest_scene.py` — headless generate_scene self-test needing
  locally loadable sd-turbo weights and a real DEFAULT_ROOT; not convertible
  to a fake-free unit test (video agent, 2026-09-22).
- `curation/review_pipeline_download.py` — the review pipeline's former
  in-process download driver (own job row + `downloader.engine.run_download_job`);
  replaced by `hugpy_curation.review.download` over storage's public queue /
  `download_one` (curation agent, 2026-09-22).

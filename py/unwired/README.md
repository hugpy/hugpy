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
- `fleet/get_size.py` — scratch script (pasted console table, sums the sizes,
  `input()` at import time); nothing imports it and importing it blocks on
  stdin. Retired from `hugpy_fleet.worker` (compat pass, 2026-09-22).
- `monolith/` — the retired aggregator layer, dead code and the monolith's own
  docs left after the partition; see `monolith/README.md` (2026-09-22).
- `archive/abstract_hugpy_dev-pre-partition.bundle` — `git bundle --all` of the
  monolith repo (checkpoint `7c19ce7`, 20 MB, gitignored; keep a copy on the
  host). `git clone archive/abstract_hugpy_dev-pre-partition.bundle` restores it.

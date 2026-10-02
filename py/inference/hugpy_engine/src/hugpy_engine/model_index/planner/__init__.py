"""hugpy planner: computed-once weights facts, worker budget earmarks and
per-(model, worker, quant) verdicts in the DB. Moved from react/testshell
2026-10-02; the testshell imports it."""
from . import compute  # noqa: F401
from .store import (DSN, discover_all, discover_model, ensure_all, ensure_model, load_models, refresh_worker_budgets)  # noqa: F401

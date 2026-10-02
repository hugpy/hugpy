"""SHIM (2026-10-02): the planner's compute moved into the hugpy engine
package — hugpy_engine.model_index.planner.compute. This module IS that module
(sys.modules alias), so `import compute` in api.py and tests keeps working."""
from __future__ import annotations

import sys

_PY = "/srv/hugpy/src/hugpy/py"
# The editable checkouts must win over any copy installed in the venv: these
# are the trees central itself runs from (PYTHONPATH in 7002_hugpy_api.service),
# and the marker/stamp code here must be the same code that writes hugpy.json.
for _p in reversed((f"{_PY}/inference/hugpy_engine/src", f"{_PY}/storage/hugpy_storage/src",
                    f"{_PY}/fleet/hugpy_fleet/src", f"{_PY}/services/hugpy_server/src",
                    f"{_PY}/platform/hugpy_platform/src")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


from hugpy_engine.model_index.planner import compute as _planner_compute  # noqa: E402

sys.modules[__name__] = _planner_compute

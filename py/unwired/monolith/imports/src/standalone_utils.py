"""De-vendored core helpers.

The implementation that used to live here (a stdlib-only carve of abstract_utilities)
now ships as the published, zero-dependency ``abstract_essentials`` package. This
module re-exports it so the internal import path
``abstract_hugpy_dev.imports.src.standalone_utils`` stays stable for existing call
sites (init_imports `import *`, module_imports `lazy_import/nullProxy`,
constants/_platform/flask_app `get_env_value`).
"""
from abstract_essentials import *          # noqa: F401,F403  — the __all__ surface
from abstract_essentials import __all__    # noqa: F401
# Explicit re-exports for names imported directly by hugpy (in __all__ today, but
# pinned here so a future __all__ change can't silently break these import sites):

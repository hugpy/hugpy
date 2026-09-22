"""Make ``import pydantic`` work on platforms without ``pydantic_core``.

pydantic v2 is split into a Python frontend and a Rust-compiled core
(``pydantic_core``). PyPI ships ``pydantic_core`` only as *manylinux* wheels,
which do not install on Termux/Android (bionic libc) — and building it from
source there needs a full Rust toolchain. Without it, ``import pydantic`` raises
``ModuleNotFoundError: No module named 'pydantic_core'`` and, because the base
import chain imports pydantic eagerly, the *entire* package fails to import on a
phone.

``ensure_pydantic()`` fixes that: if the real pydantic imports, it does nothing
(every existing install is byte-for-byte unchanged). Only when pydantic is
missing/broken does it register a minimal pure-Python stand-in under
``sys.modules['pydantic']`` so the rest of the package — which uses pydantic for
model construction + attribute access, not heavy validation — imports and runs.

The shim is intentionally lenient: it constructs models, applies ``Field``
defaults/``default_factory``, supports ``model_dump``/``dict``/``model_dump_json``,
and turns validators into no-ops. It does NOT validate or coerce types. It exists
so a phone can act as a coordinator/worker, not to replace pydantic on a server.
"""
from __future__ import annotations

import sys
import types

_UNSET = object()


def _build_shim() -> types.ModuleType:
    mod = types.ModuleType("pydantic")
    mod.__doc__ = "Pure-Python pydantic shim (no validation) — see _compat_pydantic."
    mod.__version__ = "0-shim"
    mod.SHIM = True  # marker so callers can detect the fallback if they care

    class FieldInfo:
        __slots__ = ("default", "default_factory")

        def __init__(self, default=_UNSET, default_factory=None):
            self.default = default
            self.default_factory = default_factory

    def Field(default=_UNSET, *, default_factory=None, **_ignored):
        return FieldInfo(default, default_factory)

    def ConfigDict(**kwargs):
        return dict(kwargs)

    def _annotations_of(cls):
        anns: dict = {}
        for klass in reversed(cls.__mro__):
            anns.update(getattr(klass, "__annotations__", {}) or {})
        anns.pop("model_config", None)  # config, not a data field
        return anns

    class BaseModel:
        """Minimal stand-in: stores fields as attributes; no type validation."""

        model_config: dict = {}

        def __init__(self, **data):
            anns = _annotations_of(type(self))
            for name in anns:
                if name in data:
                    setattr(self, name, data.pop(name))
                    continue
                default = getattr(type(self), name, _UNSET)
                if isinstance(default, FieldInfo):
                    if default.default_factory is not None:
                        setattr(self, name, default.default_factory())
                    elif default.default is not _UNSET:
                        setattr(self, name, default.default)
                    # required-but-missing: left unset (shim is lenient)
                elif default is not _UNSET:
                    setattr(self, name, default)
            # keep any extra kwargs the caller passed
            for key, val in data.items():
                setattr(self, key, val)

        def model_dump(self, **_kwargs):
            out = {}
            for name in _annotations_of(type(self)):
                if hasattr(self, name):
                    val = getattr(self, name)
                    out[name] = val.model_dump() if isinstance(val, BaseModel) else val
            return out

        # pydantic v1 alias
        dict = model_dump

        def model_dump_json(self, **_kwargs):
            import json
            return json.dumps(self.model_dump(), default=str)

        @classmethod
        def model_validate(cls, obj):
            return cls(**dict(obj)) if not isinstance(obj, cls) else obj

        # pydantic v1 alias
        parse_obj = model_validate

        def __repr__(self):
            inner = ", ".join(f"{k}={v!r}" for k, v in self.model_dump().items())
            return f"{type(self).__name__}({inner})"

    def _passthrough_decorator(*d_args, **_d_kwargs):
        # Usable as @model_validator(mode="after") or bare @model_validator.
        if len(d_args) == 1 and callable(d_args[0]) and not _d_kwargs:
            return d_args[0]

        def deco(fn):
            return fn

        return deco

    class ValidationError(ValueError):
        pass

    class _Stub:
        """Lenient catch-all for any other pydantic name the code imports."""

        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]  # behave as a no-op decorator
            return self

    mod.BaseModel = BaseModel
    mod.Field = Field
    mod.FieldInfo = FieldInfo
    mod.ConfigDict = ConfigDict
    mod.model_validator = _passthrough_decorator
    mod.field_validator = _passthrough_decorator
    mod.validator = _passthrough_decorator        # v1 name
    mod.root_validator = _passthrough_decorator    # v1 name
    mod.ValidationError = ValidationError

    def _module_getattr(name):  # PEP 562: long-tail names → lenient stub
        if name.startswith("__") and name.endswith("__"):
            # Dunders must stay honest: inspect/importlib walk sys.modules and
            # call str methods on e.g. __file__ — a stub there crashes them.
            raise AttributeError(name)
        return _Stub

    mod.__getattr__ = _module_getattr
    return mod


def ensure_pydantic() -> bool:
    """Guarantee ``import pydantic`` works. Returns True if the shim was used.

    No-op (returns False) when the real pydantic is importable.
    """
    try:
        import pydantic  # noqa: F401
        return False
    except Exception:
        pass

    # Provide a stub pydantic_core too, so any direct `import pydantic_core`
    # (or a half-initialized real pydantic left in sys.modules) doesn't re-trip.
    if "pydantic_core" not in sys.modules or getattr(
        sys.modules.get("pydantic_core"), "__version__", None
    ) is None:
        core = types.ModuleType("pydantic_core")
        core.__version__ = "0-shim"

        class PydanticUndefined:  # referenced by some pydantic-adjacent code
            pass

        core.PydanticUndefined = PydanticUndefined
        sys.modules["pydantic_core"] = core

    sys.modules["pydantic"] = _build_shim()
    return True

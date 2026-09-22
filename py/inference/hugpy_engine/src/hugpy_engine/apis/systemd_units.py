"""Deprecated — systemd unit rendering now lives in the unified serve module.

The standalone systemd-unit generator here predated the pluggable serve-driver
registry and hardcoded Linux-only paths (``/srv/abstractendeavors/...``,
``/etc/systemd/system``). Unit rendering is now the ``SystemdDriver`` inside
``hugpy.managers.serve.serve`` (one of several OS-aware drivers — see also
``SupervisedDriver`` for Windows/macOS). This shim re-exports that module so the
old import path resolves to the live implementation.
"""

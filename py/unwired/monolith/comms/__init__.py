"""comms — the shared F1/F5 substrate every transport imports.

Two primitives, one home:

    jobs.py  (F5)  One Job schema + JobStore for *every* unit of work — chat
                   requests from any transport (web SSE, /v1, Discord, CLI,
                   worker relay) and model downloads. Replaces the flask-only
                   download JobStore and the ephemeral dispatch.activity dict.
    bus.py   (F1)  One typed, frozen, addressed message envelope + in-process
                   pub/sub. Control messages (cancel/stop/restart) travel the
                   SAME bus as lifecycle events — no side channels.

This package is deliberately stdlib-only and imports nothing from the rest of
abstract_hugpy_dev, so flask_app, managers.dispatch, bot, worker_agent and the
keeper adapters can all import it without cycle risk.
"""
from hugpy_control.jobs import (
    CANONICAL_STATUSES,
    TERMINAL_STATUSES,
    Job,
    JobError,
    JobStore,
    job_store,
    normalize_status,
)
from hugpy_control.bus import (
    TOPIC_CONTROL_CANCEL,
    TOPIC_JOB_CREATED,
    TOPIC_JOB_DONE,
    TOPIC_JOB_STATUS,
    Bus,
    BusMessage,
    Subscription,
    bus,
    wire_cancel,
    wire_job_events,
)
from hugpy_control.principals import Principal, PrincipalStore, allowed, principal_store
from hugpy_control.settings import SettingsStore, settings_store, wire_settings_events
from hugpy_fleet.central.blocklist import (
    BLOCKED_MARKER,
    NS as BLOCKED_NS,
    block as block_model,
    block_info,
    block_reason,
    blocked_keys,
    is_blocked,
    unblock as unblock_model,
)
from hugpy_fleet.central.calibration import (
    CalibrationStore,
    calibration_store,
    calibration_table,
    clamp_correction,
    corrections_for,
    record_samples,
)
from hugpy_fleet.central.model_metrics import ModelMetricsStore, derive_variant, model_metrics_store

__all__ = [
    "CANONICAL_STATUSES",
    "TERMINAL_STATUSES",
    "Job",
    "JobError",
    "JobStore",
    "job_store",
    "normalize_status",
    "TOPIC_CONTROL_CANCEL",
    "TOPIC_JOB_CREATED",
    "TOPIC_JOB_DONE",
    "TOPIC_JOB_STATUS",
    "Bus",
    "BusMessage",
    "Subscription",
    "bus",
    "wire_cancel",
    "wire_job_events",
    "Principal",
    "PrincipalStore",
    "allowed",
    "principal_store",
    "SettingsStore",
    "settings_store",
    "wire_settings_events",
    "BLOCKED_MARKER",
    "BLOCKED_NS",
    "block_model",
    "block_info",
    "block_reason",
    "blocked_keys",
    "is_blocked",
    "unblock_model",
    "CalibrationStore",
    "calibration_store",
    "calibration_table",
    "clamp_correction",
    "corrections_for",
    "record_samples",
    "ModelMetricsStore",
    "derive_variant",
    "model_metrics_store",
]

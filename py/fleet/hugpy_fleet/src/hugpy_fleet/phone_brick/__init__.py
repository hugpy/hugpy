"""Phone-brick: distributed PPE detection across a chain of phone workers.

A small mechanic for running ONNX YOLO PPE detection on a fleet of cheap Android
phones (Termux) and collating their verdicts by plurality consensus. Two halves:

* the **worker** (:mod:`.worker`) — runs on each phone, serving a ``yolo <url>``
  HTTP verb backed by :class:`.detector.OnnxYoloDetector`;
* the **orchestrator** (:mod:`.orchestrator`) — runs on a control box, fanning
  one image across the chain and writing an annotated, consensus-labelled result.

The two share a single wire :mod:`.protocol`, so the worker's output format and
the orchestrator's parser can never drift apart.

    python -m abstract_hugpy_dev.phone_brick worker
    python -m abstract_hugpy_dev.phone_brick orchestrate --image img.jpg \\
        --file-server http://192.168.0.26:8088/chain-outputs/ \\
        --phones red:192.168.0.32:5002:#f85149,blue:192.168.0.70:5003:#58a6ff
"""
from hugpy_fleet.phone_brick.schemas import (
    DEFAULT_MODEL_PATH,
    ChainConfig,
    ChainResult,
    Detection,
    DetectorConfig,
    PhaseResult,
    PhoneSpec,
    RunCancelled,
    WorkerConfig,
)
from hugpy_fleet.phone_brick.detector import OnnxYoloDetector
from hugpy_fleet.phone_brick.protocol import format_detections, parse_detections
from hugpy_fleet.phone_brick.consensus import plurality_consensus
from hugpy_fleet.phone_brick.client import WorkerClient
from hugpy_fleet.phone_brick.rendering import draw_detections
from hugpy_fleet.phone_brick.orchestrator import ChainOrchestrator
from hugpy_fleet.phone_brick.worker import Worker, build_app, config_from_env
from hugpy_fleet.phone_brick.rpc_backend import (
    RpcBackendAgent,
    RpcBackendConfig,
    maybe_start_rpc_backend,
)
from hugpy_fleet.phone_brick.analyze import (
    analyze_chain_result,
    analyze_detections,
    analyze_scene_text,
    chain_result_to_text,
    detections_to_text,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "ChainConfig", "ChainResult", "Detection", "DetectorConfig",
    "PhaseResult", "PhoneSpec", "RunCancelled", "WorkerConfig",
    "OnnxYoloDetector",
    "format_detections", "parse_detections",
    "plurality_consensus",
    "WorkerClient",
    "draw_detections",
    "ChainOrchestrator",
    "Worker", "build_app", "config_from_env",
    "RpcBackendAgent", "RpcBackendConfig", "maybe_start_rpc_backend",
    "analyze_chain_result", "analyze_detections", "analyze_scene_text",
    "chain_result_to_text", "detections_to_text",
]

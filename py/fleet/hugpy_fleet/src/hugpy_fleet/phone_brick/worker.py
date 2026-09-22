"""Phone-brick worker — the canonical, consolidated PPE-detection service.

This replaces the family of near-identical field scripts (``worker.py``,
``worker-yolo-6class-v0.0.1.py``, ``worker-multiclass-v0.0.2.py``, the 2-class
helmet variant, and the verbs-only ``worker-free.py``) with one configurable
service. Behaviour is driven by :class:`WorkerConfig`; every knob also has an
environment fallback so it drops onto a Termux phone with no code edits.

Wire protocol (unchanged, so existing orchestrators keep working)::

    POST /queue     {"task": "yolo <image_url>"}  -> {"status": "queued", "id"}
    GET  /results                                  -> [ {id, status, result}, … ]
    GET  /status                                   -> health + model/class info

Supported task verbs:

    yolo <url>      run detection on an image URL (the core verb)
    fetch <url>     GET a URL, return the first 50 KB
    count <text>    chars / words / lines
    upper|lower|reverse <text>   small text helpers
    sh <command>    run a shell command — DISABLED unless enable_shell is set
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import urllib.request
from datetime import datetime

from hugpy_fleet.phone_brick.detector import OnnxYoloDetector
from hugpy_fleet.phone_brick.protocol import (
    VERB_COUNT,
    VERB_FETCH,
    VERB_LOWER,
    VERB_REVERSE,
    VERB_SHELL,
    VERB_UPPER,
    VERB_YOLO,
    format_detections,
    format_no_detections,
)
from hugpy_fleet.phone_brick.schemas import DEFAULT_MODEL_PATH, DetectorConfig, WorkerConfig


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def config_from_env() -> WorkerConfig:
    """Build a :class:`WorkerConfig` from environment variables.

    Honoured vars: ``MODEL_PATH``, ``YOLO_IMGSZ``, ``YOLO_CONF``, ``PORT``,
    ``HOST``, ``PHONE_BRICK_ENABLE_SHELL``.
    """
    detector = DetectorConfig(
        model_path=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        imgsz=int(os.environ.get("YOLO_IMGSZ", "640")),
        conf_threshold=float(os.environ.get("YOLO_CONF", "0.25")),
    )
    return WorkerConfig(
        detector=detector,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5002")),
        enable_shell=_env_flag("PHONE_BRICK_ENABLE_SHELL", False),
    )


class Worker:
    """A single-threaded job queue wrapping an :class:`OnnxYoloDetector`.

    Jobs are processed one at a time by a background thread (ONNX CPU inference
    is the bottleneck, so serialising keeps memory predictable on a phone).
    Completed jobs accumulate until drained via ``drain_completed``.
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.detector = OnnxYoloDetector(config.detector)
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._completed: list[dict] = []
        self._current: dict | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    # -- queue api ---------------------------------------------------------
    def enqueue(self, task: str, job_id: str | None = None) -> str:
        job = {
            "id": job_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "task": task,
            "status": "queued",
            "created": datetime.now().isoformat(),
            "result": None,
        }
        self._queue.put(job)
        return job["id"]

    def drain_completed(self) -> list[dict]:
        with self._lock:
            done = self._completed.copy()
            self._completed.clear()
        return done

    def status(self) -> dict:
        cfg = self.config
        return {
            "status": "running",
            "queue_size": self._queue.qsize(),
            "processing": self._current["id"] if self._current else None,
            "completed_count": len(self._completed),
            "model_loaded": self.detector.loaded,
            "model_path": cfg.detector.model_path,
            "model_exists": os.path.exists(cfg.detector.model_path),
            "classes": self.detector.classes,
            "classes_source": self.detector.classes_source,
            "shell_enabled": cfg.enable_shell,
            "port": cfg.port,
        }

    # -- processing --------------------------------------------------------
    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            with self._lock:
                self._current = job
                job["status"] = "processing"
            job["result"] = self.process(job["task"])
            with self._lock:
                job["status"] = "complete"
                self._completed.append(job)
                self._current = None
            self._queue.task_done()

    def process(self, task: str) -> str:
        t = task.strip()
        if t.startswith(VERB_YOLO + " "):
            return self._run_yolo(t[len(VERB_YOLO) + 1:].strip())
        if t.startswith(VERB_FETCH + " "):
            return self._run_fetch(t[len(VERB_FETCH) + 1:].strip())
        if t.startswith(VERB_SHELL + " "):
            return self._run_shell(t[len(VERB_SHELL) + 1:].strip())
        if t.startswith(VERB_COUNT + " "):
            text = t[len(VERB_COUNT) + 1:]
            return (f"chars={len(text)} words={len(text.split())} "
                    f"lines={len(text.splitlines())}")
        if t.startswith(VERB_UPPER + " "):
            return t[len(VERB_UPPER) + 1:].upper()
        if t.startswith(VERB_LOWER + " "):
            return t[len(VERB_LOWER) + 1:].lower()
        if t.startswith(VERB_REVERSE + " "):
            return t[len(VERB_REVERSE) + 1:][::-1]
        return ("Echo: " + task + "\n\nVerbs: yolo <url>, fetch <url>, "
                "sh <cmd>, count, upper, lower, reverse")

    def _run_yolo(self, url: str) -> str:
        try:
            detections, width, height = self.detector.infer_url(url)
        except Exception as exc:  # noqa: BLE001 — report cleanly, don't crash the loop
            return f"yolo error: {type(exc).__name__}: {exc}"
        if not detections:
            return format_no_detections(url, self.config.detector.conf_threshold)
        return format_detections(url, width, height, detections)

    @staticmethod
    def _run_fetch(url: str) -> str:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return resp.read(50000).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return f"fetch error: {exc}"

    def _run_shell(self, command: str) -> str:
        if not self.config.enable_shell:
            return ("shell error: the 'sh' verb is disabled; set "
                    "PHONE_BRICK_ENABLE_SHELL=1 to enable arbitrary shell "
                    "execution on a trusted network")
        try:
            out = subprocess.run(command, shell=True, capture_output=True,
                                 text=True, timeout=30)
            return (out.stdout or "") + (
                ("\n[stderr]\n" + out.stderr) if out.stderr else "")
        except Exception as exc:  # noqa: BLE001
            return f"shell error: {exc}"


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
def build_app(config: WorkerConfig | None = None, worker: Worker | None = None):
    """Construct the worker's Flask app. Starts the background job thread."""
    from flask import Flask, jsonify, request

    if worker is None:
        worker = Worker(config or config_from_env())
    worker.start()

    # Announce this phone to the console pool when PHONE_BRICK_CENTRAL is set
    # (no-op otherwise, so standalone/CLI use is unchanged).
    from hugpy_fleet.phone_brick.registration import maybe_start
    maybe_start(worker, worker.config.port)

    # Optionally also lend this phone's compute to the LLM shard pool as an RPC
    # backend (no-op unless PHONE_BRICK_RPC is set). Orthogonal to YOLO serving.
    from hugpy_fleet.phone_brick.rpc_backend import maybe_start_rpc_backend
    maybe_start_rpc_backend()

    app = Flask("abstract_hugpy_dev_phone_brick_worker")

    @app.route("/status")
    def status():
        return jsonify(worker.status())

    @app.route("/queue", methods=["POST"])
    def add_job():
        data = request.get_json(silent=True)
        if not data or "task" not in data:
            return jsonify({"error": "missing task"}), 400
        job_id = worker.enqueue(data["task"], data.get("id"))
        return jsonify({"status": "queued", "id": job_id})

    @app.route("/results")
    def results():
        return jsonify(worker.drain_completed())

    return app


def main(argv: list[str] | None = None) -> int:
    config = config_from_env()
    print(f"PHONE BRICK WORKER on {config.host}:{config.port}")
    print(f"Model: {config.detector.model_path}  "
          f"exists={os.path.exists(config.detector.model_path)}")
    print(f"Shell verb enabled: {config.enable_shell}")
    app = build_app(config)
    app.run(host=config.host, port=config.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

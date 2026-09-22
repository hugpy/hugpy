"""GPU worker agent — see ``agent.py`` for the runnable entry point.

    python -m abstract_hugpy.worker_agent --central https://abstractgpt.ai
"""
from hugpy_fleet.worker.agent import main, build_app, detect_gpus, CentralClient

__all__ = ["main", "build_app", "detect_gpus", "CentralClient"]

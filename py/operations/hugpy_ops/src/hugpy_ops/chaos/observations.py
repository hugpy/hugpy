"""Observation-file writer shared by chaos runs and sweeps."""

import json

from hugpy_ops.chaos.schema import validate_observation


def append_observation(out_dir, obs_path, obs: dict) -> None:
    problems = validate_observation(obs)
    if problems:
        obs.setdefault("_schema_problems", problems)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(obs_path, "a") as fh:
        fh.write(json.dumps(obs) + "\n")

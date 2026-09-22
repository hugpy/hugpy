"""Dependency-free public contracts for model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ModelDescriptor:
    """Transport-neutral description of a model in the live catalog."""

    key: str
    name: str
    hub_id: Optional[str]
    framework: Optional[str]
    tasks: tuple[str, ...]
    primary_task: Optional[str]
    context_length: Optional[int]
    status: Optional[str]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class QueryResult:
    """Completed text and metadata produced by one model query."""

    text: str
    request_id: str
    finish_reason: str = "stop"
    usage: Optional[Mapping[str, Any]] = None
    timings: Optional[Mapping[str, Any]] = None


class QueryError(RuntimeError):
    """A model query failed before it produced a successful result."""

    def __init__(self, message: str, *, request_id: str, partial_text: str = ""):
        super().__init__(message)
        self.request_id = request_id
        self.partial_text = partial_text


__all__ = ["ModelDescriptor", "QueryError", "QueryResult"]

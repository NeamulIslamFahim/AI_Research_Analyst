"""Reusable retry-based workflow base class."""

from __future__ import annotations

import abc
from typing import Any, Dict, Generic, Optional, TypeVar

from langgraph.graph import END, StateGraph

from backend.services.validation import strict_validate

TState = TypeVar("TState")


class RetryWorkflow(abc.ABC, Generic[TState]):
    """Base class for run/check/retry graph workflows."""

    state_schema: Any = None
    result_schema: Any = None
    max_retries: int = 2

    def __init__(self) -> None:
        self.graph = self._build_graph()

    @abc.abstractmethod
    def _run_step(self, state: TState) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def _check_step(self, state: TState) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_graph(self):
        graph = StateGraph(self.state_schema)
        graph.add_node("run", self._run_step)
        graph.add_node("check", self._check_step)
        graph.set_entry_point("run")
        graph.add_edge("run", "check")
        graph.add_conditional_edges(
            "check",
            lambda state: "run" if state.get("validation_error") and state.get("retries", 0) < self.max_retries else END,
        )
        return graph.compile()

    def _strict_validate_result(self, cleaned: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        return strict_validate(self.result_schema, cleaned)

    def run(self, initial_state: TState) -> Dict[str, Any]:
        out = self.graph.invoke(initial_state)
        return out.get("result") or {}

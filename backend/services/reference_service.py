"""Reference generation service wrapper."""

from __future__ import annotations

from typing import Any, Dict


class ReferenceService:
    def run(self, topic: str) -> Dict[str, Any]:
        from backend.main import _run_reference_generator_impl

        return _run_reference_generator_impl(topic)


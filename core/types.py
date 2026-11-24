from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunConfig:
    # retrieval / selection
    k: int = 7
    min_score: float = 0.0

    # behavior
    generate_rule: bool = False
    only_answer: bool = False

    # diversify
    diversify_method: bool = False
    max_per_method: int = 999

    diversify_tag: bool = False
    max_per_tag: int = 999

    # extra (추후 확장용)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    query: str
    selected: List[Dict[str, Any]]
    guardrail: Optional[Dict[str, Any]] = None
    debug: Dict[str, Any] = field(default_factory=dict)

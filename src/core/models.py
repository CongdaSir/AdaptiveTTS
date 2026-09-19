from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional


class Action(str, Enum):
    ANSWER = "ANSWER"
    SAMPLE = "SAMPLE"
    REFINE = "REFINE"
    VERIFY = "VERIFY"


@dataclass(frozen=True)
class TrajectoryStep:
    action: str
    observation: str = ""
    tool: Optional[str] = None
    tokens: int = 0


@dataclass(frozen=True)
class ResourceUsage:
    tokens: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    cost: float = 1.0


@dataclass(frozen=True)
class RolloutResult:
    answer: Any
    trajectory: tuple[TrajectoryStep, ...] = ()
    method: str = "unknown"
    summary: str = ""
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    failed: bool = False
    error: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class VerificationResult:
    passed: Optional[bool]
    score: float = 0.0
    feedback: str = ""
    counterexample: Optional[str] = None
    usage: ResourceUsage = field(default_factory=ResourceUsage)


@dataclass(frozen=True)
class RolloutNode:
    """Immutable record of one complete executor rollout."""

    node_id: str
    task_id: str
    answer: Any
    trajectory: tuple[TrajectoryStep, ...]
    method: str
    summary: str
    usage: ResourceUsage
    parent_id: Optional[str] = None
    created_by: Action = Action.SAMPLE
    failed: bool = False
    error: Optional[str] = None
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class NodeView:
    node: RolloutNode
    verifications: tuple[VerificationResult, ...] = ()

    @property
    def best_score(self) -> float:
        return max((v.score for v in self.verifications), default=0.0)

    @property
    def passed(self) -> Optional[bool]:
        values = [v.passed for v in self.verifications if v.passed is not None]
        return True if True in values else (False if values and all(v is False for v in values) else None)


@dataclass(frozen=True)
class ActionRequest:
    action: Action
    target_node_id: Optional[str] = None


@dataclass(frozen=True)
class SearchState:
    task_id: str
    prompt: str
    nodes: tuple[NodeView, ...]
    remaining_budget: float
    spent_budget: float
    step: int
    max_nodes: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def best_node(self) -> Optional[NodeView]:
        valid = (n for n in self.nodes if not n.node.failed and n.node.answer is not None)
        return max(valid, key=lambda n: (n.best_score, n.passed is True, -n.node.usage.cost), default=None)

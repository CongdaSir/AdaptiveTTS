from __future__ import annotations

from typing import Callable, Protocol, Sequence
from .models import (Action, ActionRequest, ResourceUsage, RolloutNode, RolloutResult,
                     SearchState, VerificationResult)


class Executor(Protocol):
    def sample(self, prompt: str, metadata: dict) -> RolloutResult: ...
    def refine(self, prompt: str, parent: RolloutNode, feedback: Sequence[VerificationResult], metadata: dict) -> RolloutResult: ...


class Verifier(Protocol):
    def verify(self, prompt: str, node: RolloutNode, metadata: dict) -> VerificationResult: ...


class ValueModel(Protocol):
    def __call__(self, state: SearchState, request: ActionRequest) -> float: ...


class Controller(Protocol):
    def decide(self, state: SearchState, legal_actions: Sequence[ActionRequest]) -> ActionRequest: ...


class FunctionVerifier:
    """Adapter for a simple ``fn(answer, prompt)`` verifier."""
    def __init__(self, fn: Callable[[object, str], bool], cost: float = 1.0):
        self.fn, self.cost = fn, cost

    def verify(self, prompt: str, node: RolloutNode, metadata: dict) -> VerificationResult:
        passed = bool(self.fn(node.answer, prompt))
        return VerificationResult(passed=passed, score=1.0 if passed else 0.0,
                                  feedback="passed" if passed else "verification failed",
                                  usage=ResourceUsage(cost=self.cost))

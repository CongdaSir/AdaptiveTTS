from __future__ import annotations

from typing import Sequence
from .interfaces import ValueModel
from .models import Action, ActionRequest, SearchState


class LearnedController:
    """Cost-sensitive controller backed by any learned value model.

    The model returns estimated final reward gain over ANSWER. Cost is subtracted
    here, so the model can be trained independently of a particular lambda.
    """
    def __init__(self, value_model: ValueModel, cost_weight: float = 1.0):
        self.value_model, self.cost_weight = value_model, cost_weight

    def decide(self, state: SearchState, legal_actions: Sequence[ActionRequest]) -> ActionRequest:
        if not legal_actions:
            raise ValueError("controller received no legal actions")
        answer = next((x for x in legal_actions if x.action is Action.ANSWER), None)
        scored = [(self.value_model(state, x) - self.cost_weight * self._cost(x, state), x) for x in legal_actions]
        best_score, best = max(scored, key=lambda pair: pair[0])
        return answer if answer is not None and best_score <= 0 else best

    @staticmethod
    def _cost(request: ActionRequest, state: SearchState) -> float:
        if request.action is Action.SAMPLE: return 1.0
        if request.action is Action.REFINE: return 1.0
        if request.action is Action.VERIFY: return 0.25
        return 0.0


class HeuristicController(LearnedController):
    """Deterministic baseline useful before a learned model is trained."""
    def __init__(self, cost_weight: float = 0.2):
        def value(state: SearchState, req: ActionRequest) -> float:
            if req.action is Action.ANSWER: return 0.0
            if req.action is Action.VERIFY: return 0.55 if req.target_node_id else 0.0
            if req.action is Action.REFINE:
                target = next((n for n in state.nodes if n.node.node_id == req.target_node_id), None)
                return 0.9 if target and target.passed is False else 0.25
            valid_nodes = sum(not node.node.failed and node.node.answer is not None
                              for node in state.nodes)
            return 0.35 if valid_nodes < 2 else 0.05
        super().__init__(value, cost_weight)

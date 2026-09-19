from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from .models import Action, ActionRequest, NodeView, SearchState


@dataclass(frozen=True)
class PruningConfig:
    max_depth: int = 4
    max_refinements_per_node: int = 2
    max_nodes: int = 16
    deduplicate_answers: bool = True


class RulePruner:
    """Hard constraints only; quality remains the controller's responsibility."""
    def __init__(self, config: PruningConfig = PruningConfig()): self.config = config

    def filter(self, state: SearchState, actions: Sequence[ActionRequest]) -> list[ActionRequest]:
        if len(state.nodes) >= self.config.max_nodes:
            actions = [a for a in actions if a.action in (Action.ANSWER, Action.VERIFY)]
        result = []
        for a in actions:
            target = next((n for n in state.nodes if n.node.node_id == a.target_node_id), None)
            if a.action is Action.REFINE and (target is None or self._depth(target, state) >= self.config.max_depth): continue
            if a.action is Action.REFINE and sum(n.node.parent_id == a.target_node_id for n in state.nodes) >= self.config.max_refinements_per_node: continue
            if a.action is Action.VERIFY and target and target.passed is not None: continue
            result.append(a)
        return result or [ActionRequest(Action.ANSWER, state.best_node.node.node_id if state.best_node else None)]

    def _depth(self, node: NodeView, state: SearchState) -> int:
        depth, parent = 0, node.node.parent_id
        while parent:
            depth += 1
            p = next((n for n in state.nodes if n.node.node_id == parent), None)
            parent = p.node.parent_id if p else None
        return depth

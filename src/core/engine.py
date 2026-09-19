from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4
from .interfaces import Controller, Executor, Verifier
from .models import Action, ActionRequest, NodeView, RolloutNode, SearchState, VerificationResult
from .rules import PruningConfig, RulePruner


@dataclass(frozen=True)
class SearchConfig:
    budget: float = 10.0
    max_steps: int = 32
    max_nodes: int = 16
    pruning: PruningConfig = PruningConfig()


@dataclass(frozen=True)
class SearchResult:
    answer: Any
    node: Optional[RolloutNode]
    nodes: tuple[NodeView, ...]
    spent_budget: float
    actions: tuple[ActionRequest, ...]
    stopped_reason: str


class AdaptiveSearch:
    def __init__(self, executor: Executor, verifier: Verifier, controller: Controller, config: SearchConfig = SearchConfig()):
        self.executor, self.verifier, self.controller, self.config = executor, verifier, controller, config
        self.pruner = RulePruner(config.pruning)

    @property
    def rollout_cost(self) -> float:
        """Conservative cost reserved before starting SAMPLE or REFINE."""
        return max(0.0, float(getattr(self.executor, "estimated_cost", 1.0)))

    def run(self, task_id: str, prompt: str, metadata: Optional[dict] = None) -> SearchResult:
        metadata = dict(metadata or {})
        nodes: dict[str, RolloutNode] = {}
        evidence: dict[str, list[VerificationResult]] = {}
        spent, actions_taken = 0.0, []
        for step in range(self.config.max_steps):
            views = tuple(NodeView(n, tuple(evidence.get(nid, ()))) for nid, n in nodes.items())
            state = SearchState(task_id, prompt, views, self.config.budget - spent, spent, step, self.config.max_nodes, metadata)
            legal = self._legal(state)
            legal = self.pruner.filter(state, legal)
            request = self.controller.decide(state, legal)
            actions_taken.append(request)
            if request.action is Action.ANSWER:
                chosen = self._choose(state, request.target_node_id)
                if chosen:
                    reason = "answer"
                elif nodes and all(node.failed for node in nodes.values()):
                    reason = "all_rollouts_failed"
                else:
                    reason = "no_valid_answer"
                return SearchResult(chosen.node.answer if chosen else None, chosen.node if chosen else None, views, spent, tuple(actions_taken), reason)
            if request.action is Action.VERIFY:
                target = self._target(state, request.target_node_id)
                if target is None: continue
                result = self.verifier.verify(prompt, target.node, metadata)
                spent += result.usage.cost
                evidence.setdefault(target.node.node_id, []).append(result)
            else:
                if request.action is Action.SAMPLE:
                    result = self.executor.sample(prompt, metadata)
                else:
                    target = self._target(state, request.target_node_id)
                    if target is None: continue
                    result = self.executor.refine(prompt, target.node, target.verifications, metadata)
                spent += result.usage.cost
                node = RolloutNode(str(uuid4()), task_id, result.answer, tuple(result.trajectory), result.method, result.summary, result.usage,
                                    request.target_node_id, request.action, result.failed, result.error,
                                    result.finish_reason)
                nodes[node.node_id] = node
            if spent >= self.config.budget:
                stop_reason = "budget_exhausted"
                break
        else:
            stop_reason = "step_limit"
        views = tuple(NodeView(n, tuple(evidence.get(nid, ()))) for nid, n in nodes.items())
        chosen = self._choose(SearchState(task_id, prompt, views, 0, spent, self.config.max_steps, self.config.max_nodes, metadata), None)
        if not chosen and nodes and all(node.failed for node in nodes.values()):
            stop_reason = "all_rollouts_failed"
        return SearchResult(chosen.node.answer if chosen else None, chosen.node if chosen else None, views, spent, tuple(actions_taken), stop_reason)

    def _legal(self, state: SearchState) -> list[ActionRequest]:
        out = [ActionRequest(Action.ANSWER, state.best_node.node.node_id if state.best_node else None)]
        if state.remaining_budget >= self.rollout_cost:
            out.append(ActionRequest(Action.SAMPLE))
        for view in state.nodes:
            if (state.remaining_budget >= self.rollout_cost and not view.node.failed
                    and view.node.answer is not None):
                out.append(ActionRequest(Action.REFINE, view.node.node_id))
            if (view.passed is None and not view.node.failed and view.node.answer is not None
                    and state.remaining_budget >= 0.25):
                out.append(ActionRequest(Action.VERIFY, view.node.node_id))
        return out

    @staticmethod
    def _target(state: SearchState, node_id: Optional[str]) -> Optional[NodeView]:
        return next((x for x in state.nodes if x.node.node_id == node_id), None)

    def _choose(self, state: SearchState, node_id: Optional[str]) -> Optional[NodeView]:
        return self._target(state, node_id) or state.best_node

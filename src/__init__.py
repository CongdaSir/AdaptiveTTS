"""Adaptive test-time scaling package."""

from .core import (Action, ActionRequest, AdaptiveSearch, HeuristicController, LearnedController,
                   RolloutNode, RolloutResult, SearchConfig, SearchResult, SearchState,
                   VerificationResult)

_EVALUATION_EXPORTS = {
    "VLLMReasoningExecutor", "CallableReasoningExecutor", "EvaluationReport",
    "ReasoningTask", "ReasoningVerifier", "answers_equivalent",
    "evaluate_reasoning", "extract_answer", "search_result_record",
}

__all__ = [
    "Action", "ActionRequest", "AdaptiveSearch", "HeuristicController", "LearnedController",
    "RolloutNode", "RolloutResult", "SearchConfig", "SearchResult", "SearchState",
    "VerificationResult", *_EVALUATION_EXPORTS,
]

def __getattr__(name: str):
    if name in _EVALUATION_EXPORTS:
        from .evaluation.evaluate import (
            CallableReasoningExecutor, EvaluationReport, ReasoningTask, ReasoningVerifier,
            answers_equivalent, evaluate_reasoning, extract_answer, search_result_record,
        )
        from .executors.vllm_executor import VLLMReasoningExecutor
        values = locals()
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

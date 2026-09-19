"""Reasoning-task evaluation and command-line vLLM evaluation entry point."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..core.controller import HeuristicController
from ..core.engine import AdaptiveSearch, SearchConfig
from ..core.models import ResourceUsage, RolloutNode, RolloutResult, TrajectoryStep, VerificationResult


_BOXED = re.compile(r"\\boxed\s*\{(.+)\}\s*$", re.IGNORECASE)
_FINAL = re.compile(r"(?:final answer|answer)\s*(?::|is)\s*(.+)", re.IGNORECASE)

def _extract_boxed(value: str) -> str | None:
    marker = r"\boxed"
    positions = []
    offset = 0
    while True:
        pos = value.find(marker, offset)
        if pos < 0:
            break
        positions.append(pos)
        offset = pos + len(marker)
    for pos in reversed(positions):
        opening = value.find("{", pos + len(marker))
        if opening < 0:
            continue
        depth = 0
        for index in range(opening, len(value)):
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    return value[opening + 1:index].strip()
    return None

def extract_answer(text: Any) -> str:
    value = "" if text is None else str(text).strip()
    boxed = _extract_boxed(value)
    if boxed is not None:
        return boxed.rstrip(".。$")
    matches = _FINAL.findall(value)
    if matches:
        return matches[-1].strip().splitlines()[0].rstrip(".。$")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return (lines[-1] if lines else value).rstrip(".。$")

def _normalise(value: Any) -> str:
    text = extract_answer(value).lower().replace("\\,", "").replace(",", "")
    text = text.replace("$", "").replace(" ", "")
    return text[:-1] + "/100" if text.endswith("%") else text

def answers_equivalent(predicted: Any, expected: Any) -> bool:
    left, right = _normalise(predicted), _normalise(expected)
    if left == right:
        return True
    try:
        return Fraction(left) == Fraction(right)
    except (ValueError, ZeroDivisionError):
        return False

class ReasoningVerifier:
    """Verifier for datasets whose gold answer is in metadata['answer']."""
    def verify(self, prompt: str, node: RolloutNode, metadata: dict) -> VerificationResult:
        predicted, expected = extract_answer(node.answer), metadata.get("answer")
        passed = answers_equivalent(predicted, expected)
        feedback = "correct" if passed else f"expected {expected!r}, got {predicted!r}"
        return VerificationResult(passed, 1.0 if passed else 0.0, feedback, usage=ResourceUsage(cost=0.25))

class CallableReasoningExecutor:
    """Adapter for any text generator function."""
    def __init__(self, generate: Callable[[str], str], cost: float = 1.0):
        self.generate, self.cost = generate, cost
    def sample(self, prompt: str, metadata: dict) -> RolloutResult:
        return self._result(self.generate(prompt), "sample")
    def refine(self, prompt: str, parent: RolloutNode, feedback: Sequence[VerificationResult], metadata: dict) -> RolloutResult:
        notes = "\n".join(v.feedback for v in feedback if v.feedback)
        text = self.generate(f"{prompt}\n\nPrevious reasoning:\n{parent.answer}\n\nVerifier feedback:\n{notes}\n\nSolve again and give a final answer.")
        return self._result(text, "refine")
    def _result(self, text: str, method: str) -> RolloutResult:
        return RolloutResult(text, (TrajectoryStep("generate", text),), method, extract_answer(text), ResourceUsage(cost=self.cost))

@dataclass(frozen=True)
class ReasoningTask:
    task_id: str
    prompt: str
    answer: Any
    @classmethod
    def from_dict(cls, item: dict, index: int = 0) -> "ReasoningTask":
        prompt = item.get("prompt", item.get("question"))
        if prompt is None or "answer" not in item:
            raise ValueError("each task needs prompt/question and answer fields")
        return cls(str(item.get("id", index)), str(prompt), item["answer"])

@dataclass(frozen=True)
class EvaluationReport:
    total: int
    correct: int
    accuracy: float
    average_cost: float
    average_nodes: float
    action_counts: dict[str, int]
    results: tuple[Any, ...]

def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)

def search_result_record(task: ReasoningTask, result: Any) -> dict[str, Any]:
    """Convert one completed search, including its immutable tree, to JSON data."""
    nodes = []
    for view in result.nodes:
        node = view.node
        nodes.append({
            "node_id": node.node_id,
            "task_id": node.task_id,
            "parent_id": node.parent_id,
            "created_by": node.created_by.value,
            "answer": _jsonable(node.answer),
            "method": node.method,
            "summary": node.summary,
            "failed": node.failed,
            "error": node.error,
            "finish_reason": node.finish_reason,
            "usage": _jsonable(node.usage.__dict__),
            "trajectory": [_jsonable(step.__dict__) for step in node.trajectory],
            "verifications": [_jsonable(item.__dict__) for item in view.verifications],
        })
    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "gold_answer": _jsonable(task.answer),
        "prediction": _jsonable(result.answer),
        "correct": answers_equivalent(result.answer, task.answer),
        "spent_budget": result.spent_budget,
        "stopped_reason": result.stopped_reason,
        "actions": [{"action": item.action.value, "target_node_id": item.target_node_id}
                    for item in result.actions],
        "nodes": nodes,
    }

def evaluate_reasoning(search_factory: Callable[[ReasoningTask], Any], tasks: Iterable[ReasoningTask], on_result: Callable[[ReasoningTask, Any], None] | None = None) -> EvaluationReport:
    from collections import Counter
    task_list, results, actions = list(tasks), [], Counter()
    for task in task_list:
        result = search_factory(task)
        if on_result is not None:
            on_result(task, result)
        results.append(result)
        actions.update(a.action.value for a in result.actions)
    correct = sum(answers_equivalent(r.answer, t.answer) for r, t in zip(results, task_list))
    total = len(task_list)
    return EvaluationReport(total, correct, correct / total if total else 0.0,
                            sum(r.spent_budget for r in results) / total if total else 0.0,
                            sum(len(r.nodes) for r in results) / total if total else 0.0,
                            dict(actions), tuple(results))


def load_tasks(path: str) -> list[ReasoningTask]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in file_path.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(file_path.read_text())
        items = payload if isinstance(payload, list) else payload.get("data", payload.get("examples", []))
    return [ReasoningTask.from_dict(item, index) for index, item in enumerate(items)]



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adaptive test-time scaling on reasoning tasks.")
    parser.add_argument("--data", required=True, help="Local .json or .jsonl file with id, prompt/question, answer.")
    parser.add_argument("--model", required=True, help="Served model name exposed by vLLM, e.g. Qwen3-4B.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1",
                        help="vLLM OpenAI-compatible server base URL.")
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    tasks = load_tasks(args.data)
    from ..executors.vllm_executor import VLLMReasoningExecutor

    executor = VLLMReasoningExecutor(
        args.server_url, args.model, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    config = SearchConfig(budget=args.budget)
    def run(task):
        return AdaptiveSearch(executor, ReasoningVerifier(), HeuristicController(), config).run(
            task.task_id, task.prompt, {"answer": task.answer}
        )
    dataset_label = Path(args.data).stem
    model_label = args.model
    safe_dataset = "".join(char if char.isalnum() or char in "._-" else "_" for char in dataset_label)
    safe_model = "".join(char if char.isalnum() or char in "._-" else "_" for char in model_label)
    tree_path = Path("results") / f"{safe_dataset}-{safe_model}-tree.jsonl"
    tree_path.parent.mkdir(parents=True, exist_ok=True)

    # Persistence is intentionally always enabled: every completed task is flushed
    # immediately, so an interrupted run still leaves a usable prefix of JSONL.
    with tree_path.open("w", encoding="utf-8") as tree_file:
        def save_tree(task, result):
            tree_file.write(json.dumps(search_result_record(task, result), ensure_ascii=False) + "\n")
            tree_file.flush()

        report = evaluate_reasoning(run, tasks, on_result=save_tree)
    print(json.dumps({
        "model": args.model, "dataset": args.data,
        "total": report.total, "correct": report.correct,
        "accuracy": report.accuracy, "average_cost": report.average_cost,
        "average_nodes": report.average_nodes, "action_counts": report.action_counts,
        "tree_file": str(tree_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

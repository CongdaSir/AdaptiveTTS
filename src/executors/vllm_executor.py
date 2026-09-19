"""Rollout executor that calls an already-running vLLM OpenAI API."""
from __future__ import annotations
import json
from urllib.request import Request, urlopen
from ..core.models import ResourceUsage, RolloutNode, RolloutResult, TrajectoryStep, VerificationResult
from ..evaluation.evaluate import extract_answer

class VLLMReasoningExecutor:
    def __init__(self, base_url: str, model: str, max_new_tokens: int=256,
                 temperature: float=0.0, top_p: float=0.95, timeout: int=1800,
                 cost_per_1k_tokens: float=1.0):
        self.base_url=base_url.rstrip("/")
        self.model=model
        self.max_new_tokens=max_new_tokens
        self.temperature=temperature
        self.top_p=top_p
        self.timeout=timeout
        self.cost_per_1k_tokens=cost_per_1k_tokens

    @property
    def estimated_cost(self) -> float:
        return max(0.01, self.max_new_tokens / 1000 * self.cost_per_1k_tokens)

    def _complete(self, prompt: str) -> tuple[str, str, str | None, int]:
        payload={"model":self.model, "messages":[{"role":"user","content":prompt}],
                 "max_tokens":self.max_new_tokens, "temperature":self.temperature,
                 "top_p":self.top_p, "top_k":20}
        request=Request(self.base_url+"/chat/completions",
                        data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            body=json.loads(response.read())
        choice=body["choices"][0]
        message=choice["message"]
        content=message.get("content") or ""
        reasoning=message.get("reasoning_content") or message.get("reasoning") or ""
        finish_reason=choice.get("finish_reason")
        usage=body.get("usage", {})
        tokens=int(usage.get("completion_tokens", 0))
        return content, reasoning, finish_reason, tokens

    def _result(self, content: str, reasoning: str, finish_reason: str | None,
                method: str, tokens: int) -> RolloutResult:
        cost=max(0.01, tokens/1000*self.cost_per_1k_tokens) if tokens else 1.0
        trajectory=[]
        if reasoning:
            trajectory.append(TrajectoryStep("reasoning", reasoning))
        if content:
            trajectory.append(TrajectoryStep("answer", content))
        failed=not content
        error=None
        if failed:
            error="max_tokens_exceeded" if finish_reason == "length" else "missing_final_answer"
        return RolloutResult(content or None, tuple(trajectory), method,
                             extract_answer(content) if content else "",
                             ResourceUsage(tokens=tokens, cost=cost), failed, error,
                             finish_reason)

    def sample(self, prompt: str, metadata: dict) -> RolloutResult:
        content,reasoning,finish_reason,tokens=self._complete(prompt)
        return self._result(content, reasoning, finish_reason, "sample", tokens)

    def refine(self, prompt: str, parent: RolloutNode,
               feedback: list[VerificationResult], metadata: dict) -> RolloutResult:
        notes="\n".join(v.feedback for v in feedback if v.feedback) or "Re-check the solution carefully."
        content,reasoning,finish_reason,tokens=self._complete(
            f"{prompt}\n\nPrevious reasoning:\n{parent.answer}"
            f"\n\nVerifier feedback:\n{notes}"
            "\n\nSolve the problem again. Return only the final integer answer."
        )
        return self._result(content, reasoning, finish_reason, "refine", tokens)

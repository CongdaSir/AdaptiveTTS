import unittest

from src import AdaptiveSearch, HeuristicController, SearchConfig
from src.core.models import ResourceUsage, RolloutResult, TrajectoryStep, VerificationResult


class ToyExecutor:
    def sample(self, prompt, metadata):
        return RolloutResult("42", (TrajectoryStep("generate", "42"),), "sample", "42", ResourceUsage(cost=1.0))

    def refine(self, prompt, parent, feedback, metadata):
        return RolloutResult("42", (TrajectoryStep("refine", "42"),), "refine", "42", ResourceUsage(cost=1.0))


class ToyVerifier:
    def verify(self, prompt, node, metadata):
        passed = node.answer == metadata["answer"]
        return VerificationResult(passed, float(passed), "correct" if passed else "wrong", usage=ResourceUsage(cost=0.25))


class AdaptiveSearchTest(unittest.TestCase):
    def test_end_to_end_and_budget(self):
        result = AdaptiveSearch(ToyExecutor(), ToyVerifier(), HeuristicController(), SearchConfig(budget=3)).run("t", "q", {"answer": "42"})
        self.assertEqual(result.answer, "42")
        self.assertLessEqual(result.spent_budget, 3.0)
        self.assertTrue(result.nodes)

    def test_answer_is_legal_without_nodes(self):
        result = AdaptiveSearch(ToyExecutor(), ToyVerifier(), HeuristicController(), SearchConfig(budget=0, max_steps=2)).run("t", "q", {"answer": "42"})
        self.assertIsNone(result.answer)


if __name__ == "__main__":
    unittest.main()

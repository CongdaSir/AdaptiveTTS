import unittest

from src.executors.vllm_executor import VLLMReasoningExecutor


class VLLMExecutorTest(unittest.TestCase):
    def setUp(self):
        self.executor = VLLMReasoningExecutor(
            "http://127.0.0.1:8000/v1", "Qwen3-4B", max_new_tokens=16384
        )

    def test_preserves_reasoning_and_marks_length_truncation(self):
        result = self.executor._result("", "unfinished reasoning", "length", "sample", 16384)
        self.assertTrue(result.failed)
        self.assertEqual(result.error, "max_tokens_exceeded")
        self.assertEqual(result.finish_reason, "length")
        self.assertIsNone(result.answer)
        self.assertEqual(result.trajectory[0].action, "reasoning")
        self.assertEqual(result.trajectory[0].observation, "unfinished reasoning")

    def test_preserves_final_answer(self):
        result = self.executor._result("Final Answer: 42", "calculation", "stop", "sample", 100)
        self.assertFalse(result.failed)
        self.assertEqual(result.summary, "42")
        self.assertEqual([step.action for step in result.trajectory], ["reasoning", "answer"])


if __name__ == "__main__":
    unittest.main()

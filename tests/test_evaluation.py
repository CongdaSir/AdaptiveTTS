import unittest

from src.evaluation.evaluate import answers_equivalent, extract_answer


class ReasoningHelpersTest(unittest.TestCase):
    def test_extract_and_compare(self):
        self.assertEqual(extract_answer(r"\boxed{42}"), "42")
        self.assertTrue(answers_equivalent("50%", "1/2"))
        self.assertTrue(answers_equivalent("\​frac{1}{2}", "1/2") is False)


if __name__ == "__main__":
    unittest.main()

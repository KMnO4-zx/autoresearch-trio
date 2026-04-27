import unittest

from train import TinyTokenizer, build_sft_arrays, extract_sql


class TokenMaskTest(unittest.TestCase):
    def test_prompt_and_completion_weights_shift(self):
        tokenizer = TinyTokenizer()
        example = {
            "question": "How many users?",
            "evidence": "",
            "schema": "Table users: id INTEGER",
            "sql": "SELECT COUNT(*) FROM users;",
        }
        arrays = build_sft_arrays(example, tokenizer, max_seq_len=4096)
        self.assertIsNotNone(arrays)
        assert arrays is not None
        self.assertEqual(len(arrays.input_tokens), len(arrays.target_tokens))
        self.assertEqual(len(arrays.input_tokens), len(arrays.weights))
        self.assertTrue(any(weight == 0.0 for weight in arrays.weights))
        self.assertTrue(any(weight == 1.0 for weight in arrays.weights))
        # After shifting, the last prompt target may have weight 0, but the SQL
        # completion and EOS positions must be supervised.
        self.assertEqual(arrays.weights[-1], 1.0)

    def test_too_long_example_is_skipped(self):
        tokenizer = TinyTokenizer()
        example = {
            "question": "x" * 1000,
            "evidence": "",
            "schema": "Table t: x TEXT",
            "sql": "SELECT x FROM t;",
        }
        self.assertIsNone(build_sft_arrays(example, tokenizer, max_seq_len=32))

    def test_max_seq_len_applies_to_model_input_length(self):
        tokenizer = TinyTokenizer()
        example = {
            "question": "How many users?",
            "evidence": "",
            "schema": "Table users: id INTEGER",
            "sql": "SELECT COUNT(*) FROM users;",
        }
        arrays = build_sft_arrays(example, tokenizer, max_seq_len=4096)
        self.assertIsNotNone(arrays)
        assert arrays is not None
        self.assertIsNotNone(
            build_sft_arrays(example, tokenizer, max_seq_len=len(arrays.input_tokens))
        )
        self.assertIsNone(
            build_sft_arrays(example, tokenizer, max_seq_len=len(arrays.input_tokens) - 1)
        )

    def test_extract_sql(self):
        self.assertEqual(extract_sql("```sql\nSELECT 1;\n```"), "SELECT 1;")
        self.assertEqual(
            extract_sql("Here is it:\nWITH x AS (SELECT 1) SELECT * FROM x; done"),
            "WITH x AS (SELECT 1) SELECT * FROM x;",
        )
        self.assertEqual(extract_sql("I cannot answer"), "")


if __name__ == "__main__":
    unittest.main()

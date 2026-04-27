import csv
import tempfile
import unittest
from pathlib import Path

from prepare import append_results_csv


class ResultsCsvTest(unittest.TestCase):
    def test_append_creates_header_once_and_preserves_commas(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "results.csv"
            append_results_csv(
                path,
                {
                    "commit": "abc1234",
                    "run_tag": "apr27-sql",
                    "ex": "0.100000",
                    "status": "baseline",
                    "description": "baseline, with comma",
                },
            )
            append_results_csv(
                path,
                {
                    "commit": "def5678",
                    "run_tag": "apr27-sql",
                    "ex": "0.200000",
                    "status": "keep",
                    "description": "better prompt",
                },
            )

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(sum(1 for line in lines if line.startswith("commit,")), 1)
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["description"], "baseline, with comma")
            self.assertEqual(rows[1]["status"], "keep")


if __name__ == "__main__":
    unittest.main()

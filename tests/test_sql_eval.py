import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from prepare import DataError, classify_sql_safety, evaluate_sql_predictions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.executemany(
            "INSERT INTO users(name, age) VALUES (?, ?)",
            [("alice", 30), ("bob", 20), ("carol", 20)],
        )
        conn.commit()


class SqlEvalTest(unittest.TestCase):
    def test_execution_metrics_and_buckets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = root / "toy.sqlite"
            make_db(db_path)
            eval_path = root / "eval.jsonl"
            pred_path = root / "pred.jsonl"
            write_jsonl(
                eval_path,
                [
                    {
                        "question_id": "q1",
                        "db_id": "toy",
                        "sql": "SELECT COUNT(*) FROM users;",
                        "difficulty": "simple",
                        "db_path": str(db_path),
                    },
                    {
                        "question_id": "q2",
                        "db_id": "toy",
                        "sql": "SELECT name FROM users WHERE age = 20 ORDER BY name;",
                        "difficulty": "moderate",
                        "db_path": str(db_path),
                    },
                    {
                        "question_id": "q3",
                        "db_id": "toy",
                        "sql": "SELECT name FROM users;",
                        "difficulty": "challenging",
                        "db_path": str(db_path),
                    },
                ],
            )
            write_jsonl(
                pred_path,
                [
                    {"question_id": "q1", "db_id": "toy", "sql": "SELECT COUNT(*) FROM users;"},
                    {"question_id": "q2", "db_id": "toy", "sql": "SELECT name FROM users WHERE age = 20;"},
                    {"question_id": "q3", "db_id": "toy", "sql": "SELECT name FROM users WHERE age = 20;"},
                ],
            )
            metrics = evaluate_sql_predictions(pred_path, eval_path)
            self.assertAlmostEqual(metrics["ex"], 2 / 3)
            self.assertEqual(metrics["simple_ex"], 1.0)
            self.assertEqual(metrics["moderate_ex"], 1.0)
            self.assertEqual(metrics["challenging_ex"], 0.0)
            self.assertGreater(metrics["soft_f1"], metrics["ex"])
            self.assertEqual(metrics["valid_sql_rate"], 1.0)
            self.assertEqual(metrics["unsafe_sql_rate"], 0.0)

    def test_invalid_and_unsafe_sql(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = root / "toy.sqlite"
            make_db(db_path)
            eval_path = root / "eval.jsonl"
            pred_path = root / "pred.jsonl"
            write_jsonl(
                eval_path,
                [
                    {
                        "question_id": "q1",
                        "db_id": "toy",
                        "sql": "SELECT COUNT(*) FROM users;",
                        "difficulty": "simple",
                        "db_path": str(db_path),
                    },
                    {
                        "question_id": "q2",
                        "db_id": "toy",
                        "sql": "SELECT name FROM users;",
                        "difficulty": "simple",
                        "db_path": str(db_path),
                    },
                ],
            )
            write_jsonl(
                pred_path,
                [
                    {"question_id": "q1", "db_id": "toy", "sql": "nonsense"},
                    {"question_id": "q2", "db_id": "toy", "sql": "DROP TABLE users;"},
                ],
            )
            metrics = evaluate_sql_predictions(pred_path, eval_path)
            self.assertEqual(metrics["valid_sql_rate"], 0.0)
            self.assertEqual(metrics["unsafe_sql_rate"], 0.5)
            self.assertEqual(metrics["ex"], 0.0)

    def test_db_root_uses_nested_sqlite_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_root = root / "dbs"
            db_path = db_root / "toy" / "sqlite" / "toy.sqlite"
            make_db(db_path)
            eval_path = root / "eval.jsonl"
            pred_path = root / "pred.jsonl"
            write_jsonl(
                eval_path,
                [
                    {
                        "question_id": "q1",
                        "db_id": "toy",
                        "sql": "SELECT COUNT(*) FROM users;",
                        "difficulty": "simple",
                    }
                ],
            )
            write_jsonl(
                pred_path,
                [{"question_id": "q1", "db_id": "toy", "sql": "SELECT COUNT(*) FROM users;"}],
            )
            metrics = evaluate_sql_predictions(pred_path, eval_path, db_root=db_root)
            self.assertEqual(metrics["ex"], 1.0)

    def test_eval_requires_db_path_or_db_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            eval_path = root / "eval.jsonl"
            pred_path = root / "pred.jsonl"
            write_jsonl(
                eval_path,
                [
                    {
                        "question_id": "q1",
                        "db_id": "toy",
                        "sql": "SELECT 1;",
                        "difficulty": "simple",
                    }
                ],
            )
            write_jsonl(pred_path, [{"question_id": "q1", "db_id": "toy", "sql": "SELECT 1;"}])
            with self.assertRaisesRegex(DataError, "provide db_path"):
                evaluate_sql_predictions(pred_path, eval_path)

    def test_safety_classifier(self):
        self.assertEqual(classify_sql_safety("SELECT 1;")[:2], (True, False))
        self.assertEqual(classify_sql_safety("WITH x AS (SELECT 1) SELECT * FROM x")[:2], (True, False))
        self.assertEqual(classify_sql_safety("DROP TABLE x")[:2], (False, True))
        self.assertEqual(classify_sql_safety("SELECT 1; SELECT 2")[:2], (False, True))
        self.assertEqual(classify_sql_safety("explain query plan SELECT 1")[:2], (False, False))


if __name__ == "__main__":
    unittest.main()

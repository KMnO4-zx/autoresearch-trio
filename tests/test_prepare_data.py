import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path

from prepare import DataError, install_train_databases, prepare_data, read_jsonl


def make_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO users(name) VALUES ('alice')")
        conn.commit()


class PrepareDataTest(unittest.TestCase):
    def test_prepare_jsonl_to_processed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            out = root / "processed"
            make_db(raw / "bird23_train_filtered" / "train_databases" / "toy" / "toy.sqlite")
            make_db(raw / "bird_mini_dev" / "dev_databases" / "toy" / "toy.sqlite")

            train_dir = raw / "bird23_train_filtered"
            train_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "train.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": 1,
                        "db_id": "toy",
                        "question": "How many users?",
                        "SQL": "SELECT COUNT(*) FROM users;",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            eval_dir = raw / "bird_mini_dev"
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "mini_dev_sqlite.json").write_text(
                json.dumps(
                    [
                        {
                            "question_id": 2,
                            "db_id": "toy",
                            "question": "List users",
                            "evidence": "Return names.",
                            "SQL": "SELECT name FROM users;",
                            "difficulty": "simple",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            manifest = prepare_data(raw_dir=raw, out_dir=out, eval_source="mini-dev")
            self.assertEqual(manifest["train_records"], 1)
            self.assertEqual(manifest["eval_records"], 1)
            eval_rows = read_jsonl(out / "eval.jsonl")
            self.assertIn("Table users:", eval_rows[0]["schema"])
            self.assertEqual(eval_rows[0]["sql"], "SELECT name FROM users;")

    def test_missing_required_field_errors(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            out = root / "processed"
            make_db(raw / "bird23_train_filtered" / "train_databases" / "toy" / "toy.sqlite")
            make_db(raw / "bird_mini_dev" / "dev_databases" / "toy" / "toy.sqlite")
            train_dir = raw / "bird23_train_filtered"
            eval_dir = raw / "bird_mini_dev"
            train_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "train.json").write_text(
                json.dumps([{"db_id": "toy", "question": "missing sql"}]),
                encoding="utf-8",
            )
            (eval_dir / "mini_dev_sqlite.json").write_text(
                json.dumps(
                    [{"db_id": "toy", "question": "ok", "SQL": "SELECT 1;", "question_id": "e"}]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataError, "missing required"):
                prepare_data(raw_dir=raw, out_dir=out, eval_source="mini-dev")

    def test_train_holdout_uses_disjoint_database_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw"
            out = root / "processed"
            train_dir = raw / "bird23_train_filtered"
            train_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            for db_id in ("alpha", "beta", "gamma"):
                make_db(raw / "bird23_train_filtered" / "train_databases" / db_id / f"{db_id}.sqlite")
                for index in range(3):
                    rows.append(
                        {
                            "question_id": f"{db_id}-{index}",
                            "db_id": db_id,
                            "question": "How many users?",
                            "SQL": "SELECT COUNT(*) FROM users;",
                        }
                    )
            (train_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            manifest = prepare_data(
                raw_dir=raw,
                out_dir=out,
                eval_source="train-holdout",
                holdout_eval_size=2,
            )
            train_rows = read_jsonl(out / "train.jsonl")
            eval_rows = read_jsonl(out / "eval.jsonl")
            train_db_ids = {row["db_id"] for row in train_rows}
            eval_db_ids = {row["db_id"] for row in eval_rows}

            self.assertEqual(manifest["eval_source"], "train-holdout")
            self.assertEqual(manifest["eval_records"], 2)
            self.assertTrue(eval_db_ids)
            self.assertTrue(train_db_ids.isdisjoint(eval_db_ids))

    def test_install_train_databases_from_nested_zip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_db = root / "source" / "train_databases" / "toy" / "toy.sqlite"
            make_db(source_db)
            package_root = root / "package"
            nested_zip = package_root / "train" / "train_databases.zip"
            nested_zip.parent.mkdir(parents=True)
            with zipfile.ZipFile(nested_zip, "w") as zf:
                zf.write(source_db, "train_databases/toy/toy.sqlite")
                zf.writestr("__MACOSX/._toy.sqlite", "ignored")

            raw = root / "raw"
            install_train_databases(package_root, raw)

            installed = raw / "bird23_train_filtered" / "train_databases" / "toy" / "toy.sqlite"
            self.assertTrue(installed.exists())
            self.assertFalse((raw / "bird23_train_filtered" / "__MACOSX").exists())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Fixed data preparation and SQLite evaluation for autoresearch-trio.

This file plays the same role as prepare.py in karpathy/autoresearch: it is
the stable benchmark harness. Agents should normally experiment in train.py,
not by changing the evaluation code here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import zipfile
import time
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
TIME_BUDGET = 300
MAX_SEQ_LEN = 4096
SQL_TIMEOUT_SECONDS = 30.0
PROJECT_NAME = "autoresearch-trio"
RESULTS_CSV = "results.csv"

DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_DATASET_NAME = "birdsql/bird23-train-filtered"
EVAL_DATASET_NAME = "birdsql/bird_mini_dev"
TRAIN_DB_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
MINI_DEV_GOOGLE_DRIVE_ID = "13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG"
DOWNLOAD_DIR = ROOT / "data" / "downloads"

RESULTS_FIELDS = [
    "commit",
    "run_tag",
    "swanlab_run_id",
    "ex",
    "soft_f1",
    "r_ves",
    "valid_sql_rate",
    "unsafe_sql_rate",
    "latency_ms",
    "cloud_path",
    "sft_loss",
    "eval_limit",
    "eval_total",
    "sampled_eval",
    "steps",
    "examples_seen",
    "skipped_long",
    "skipped_eval_long",
    "status",
    "description",
]

DESTRUCTIVE_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "REPLACE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
}


class DataError(RuntimeError):
    """Raised when required dataset files or fields are missing."""


@dataclass
class SqlRunResult:
    ok: bool
    rows: list[tuple[Any, ...]]
    elapsed_ms: float
    error: str = ""
    unsafe: bool = False


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records from a JSON or JSONL file.

    JSON files may be a list, a single dict, or a dict containing a common
    split key such as "train", "data", or "validation".
    """

    if not path.exists():
        raise DataError(f"Missing data file: {path}")

    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(item, dict):
                    raise DataError(f"JSONL row must be an object at {path}:{line_no}")
                records.append(item)
        return records

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        if not all(isinstance(row, dict) for row in obj):
            raise DataError(f"JSON list must contain objects: {path}")
        return obj

    if isinstance(obj, dict):
        for key in ("train", "validation", "dev", "test", "data", "rows"):
            rows = obj.get(key)
            if isinstance(rows, list):
                if not all(isinstance(row, dict) for row in rows):
                    raise DataError(f"JSON key {key!r} must contain objects: {path}")
                return rows
        if obj and all(isinstance(value, list) for value in obj.values()):
            lengths = {len(value) for value in obj.values()}
            if len(lengths) == 1:
                keys = list(obj.keys())
                return [
                    {key: obj[key][index] for key in keys}
                    for index in range(next(iter(lengths)))
                ]
        return [obj]

    raise DataError(f"Unsupported JSON shape in {path}")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_json_or_jsonl(path)


def download_hf_records(raw_dir: Path, force: bool = False, include_eval: bool = True) -> None:
    """Download train/eval JSON records from Hugging Face datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataError("Missing dependency `datasets`; run `uv sync` first.") from exc

    train_path = raw_dir / "bird23_train_filtered" / "train.jsonl"
    eval_path = raw_dir / "bird_mini_dev" / "mini_dev_sqlite.jsonl"

    if force or not train_path.exists():
        print(f"Downloading {TRAIN_DATASET_NAME} -> {train_path}")
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_ds = load_dataset(TRAIN_DATASET_NAME)["train"]
        write_jsonl(train_path, [dict(row) for row in train_ds])
    else:
        print(f"Found train records: {train_path}")

    if not include_eval:
        return

    if force or not eval_path.exists():
        print(f"Downloading {EVAL_DATASET_NAME}/mini_dev_sqlite -> {eval_path}")
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_ds = load_dataset(EVAL_DATASET_NAME)["mini_dev_sqlite"]
        write_jsonl(eval_path, [dict(row) for row in eval_ds])
    else:
        print(f"Found eval records: {eval_path}")


def download_http_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download a URL to dest with a small progress indicator."""

    if dest.exists() and not force:
        print(f"Found download: {dest}")
        return dest

    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / total * 100:5.1f}%", end="", flush=True)
    if total:
        print()
    tmp.replace(dest)
    return dest


def download_google_drive_file(file_id: str, dest: Path, force: bool = False) -> Path:
    """Download a public Google Drive file without adding a gdown dependency."""

    if dest.exists() and not force:
        print(f"Found download: {dest}")
        return dest

    try:
        import requests
    except ImportError as exc:
        raise DataError("Missing dependency `requests`; run `uv sync` first.") from exc

    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)

    def confirm_token(response) -> str | None:
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        match = None
        try:
            import re

            match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
        except Exception:
            return None
        return match.group(1) if match else None

    print(f"Downloading Google Drive file {file_id} -> {dest}")
    response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
    token = confirm_token(response)
    if token:
        response = session.get(
            url,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=60,
        )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    if "text/html" in content_type and tmp.stat().st_size < 5 * 1024 * 1024:
        tmp.unlink(missing_ok=True)
        raise DataError(
            "Google Drive did not return the Mini-Dev zip directly. "
            "Please download it manually from "
            "https://drive.google.com/file/d/"
            f"{MINI_DEV_GOOGLE_DRIVE_ID}/view?usp=sharing"
        )
    tmp.replace(dest)
    return dest


def extract_zip(zip_path: Path, extract_dir: Path, force: bool = False) -> Path:
    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        print(f"Found extracted package: {extract_dir}")
        return extract_dir
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def ignored_zip_member(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return any(part == "__MACOSX" or part.startswith("._") for part in parts)


def extract_zip_filtered(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = [info for info in zf.infolist() if not ignored_zip_member(info.filename)]
        for index, info in enumerate(members, start=1):
            target = (root / info.filename).resolve()
            if target != root and root not in target.parents:
                raise DataError(f"Unsafe zip member path: {info.filename}")
            zf.extract(info, root)
            if index % 100 == 0 or index == len(members):
                print(f"\r  extracted {index}/{len(members)}", end="", flush=True)
    if members:
        print()


def copytree_merge(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copytree_merge(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def find_named_dir(root: Path, name: str) -> Path | None:
    for path in root.rglob(name):
        if path.is_dir():
            return path
    return None


def find_named_file(root: Path, names: set[str]) -> Path | None:
    for path in root.rglob("*"):
        if path.is_file() and path.name in names:
            return path
    return None


def has_sqlite_files(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.sqlite"))


def install_train_databases(package_root: Path, raw_dir: Path, force: bool = False) -> None:
    base = raw_dir / "bird23_train_filtered"
    dst = base / "train_databases"
    if force and dst.exists():
        shutil.rmtree(dst)
    if has_sqlite_files(dst):
        print(f"Found installed train databases: {dst}")
        return

    src = find_named_dir(package_root, "train_databases")
    if src is not None and has_sqlite_files(src):
        print(f"Installing train databases: {src} -> {dst}")
        copytree_merge(src, dst)
        return

    nested_zip = find_named_file(package_root, {"train_databases.zip"})
    if nested_zip is not None:
        print(f"Extracting nested train databases: {nested_zip} -> {base}")
        extract_zip_filtered(nested_zip, base)
        if has_sqlite_files(dst):
            return
        raise DataError(f"Nested train_databases.zip did not create usable databases under {dst}")

    if src is None:
        # Some mirrors unpack directly into a directory whose children are db ids.
        sqlite_files = [
            path
            for path in package_root.rglob("*.sqlite")
            if "__MACOSX" not in path.parts and not path.name.startswith("._")
        ]
        if sqlite_files:
            src = package_root
    if src is None:
        raise DataError(f"Could not find train_databases in {package_root}")
    print(f"Installing train databases: {src} -> {dst}")
    copytree_merge(src, dst)


def install_mini_dev_package(package_root: Path, raw_dir: Path) -> None:
    eval_base = raw_dir / "bird_mini_dev"
    data_file = find_named_file(
        package_root,
        {"mini_dev_sqlite.jsonl", "mini_dev_sqlite.json", "dev.jsonl", "dev.json"},
    )
    if data_file is not None:
        target_name = "mini_dev_sqlite.jsonl" if data_file.suffix == ".jsonl" else "mini_dev_sqlite.json"
        eval_base.mkdir(parents=True, exist_ok=True)
        print(f"Installing Mini-Dev data: {data_file} -> {eval_base / target_name}")
        shutil.copy2(data_file, eval_base / target_name)

    src = find_named_dir(package_root, "dev_databases")
    if src is None:
        src = find_named_dir(package_root, "databases")
    if src is None:
        raise DataError(f"Could not find dev_databases in {package_root}")
    dst = eval_base / "dev_databases"
    print(f"Installing Mini-Dev databases: {src} -> {dst}")
    copytree_merge(src, dst)


def download_database_packages(
    raw_dir: Path,
    force: bool = False,
    include_mini_dev: bool = True,
) -> None:
    train_zip = download_http_file(
        TRAIN_DB_ZIP_URL,
        DOWNLOAD_DIR / "bird_train.zip",
        force=force,
    )
    train_extract = extract_zip(train_zip, DOWNLOAD_DIR / "bird_train", force=force)
    install_train_databases(train_extract, raw_dir, force=force)

    if include_mini_dev:
        mini_zip = download_google_drive_file(
            MINI_DEV_GOOGLE_DRIVE_ID,
            DOWNLOAD_DIR / "bird_mini_dev.zip",
            force=force,
        )
        mini_extract = extract_zip(mini_zip, DOWNLOAD_DIR / "bird_mini_dev", force=force)
        install_mini_dev_package(mini_extract, raw_dir)


def download_data_assets(
    raw_dir: Path,
    force: bool = False,
    include_records: bool = True,
    include_databases: bool = True,
    include_eval: bool = True,
) -> None:
    if include_records:
        download_hf_records(raw_dir, force=force, include_eval=include_eval)
    if include_databases:
        download_database_packages(raw_dir, force=force, include_mini_dev=include_eval)


def find_first_existing(base: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            return path
    return None


def find_train_file(raw_dir: Path) -> Path | None:
    base = raw_dir / "bird23_train_filtered"
    return find_first_existing(
        base,
        [
            "train.jsonl",
            "train.json",
            "bird23_train_filtered.jsonl",
            "bird23_train_filtered.json",
            "data.jsonl",
            "data.json",
        ],
    )


def find_eval_file(raw_dir: Path) -> Path | None:
    base = raw_dir / "bird_mini_dev"
    return find_first_existing(
        base,
        [
            "mini_dev_sqlite.jsonl",
            "mini_dev_sqlite.json",
            "dev.jsonl",
            "dev.json",
            "data.jsonl",
            "data.json",
        ],
    )


def candidate_train_db_roots(raw_dir: Path) -> list[Path]:
    base = raw_dir / "bird23_train_filtered"
    return [
        base / "train_databases",
        base / "databases",
        base / "dev_databases",
        base / "database",
    ]


def eval_db_root(raw_dir: Path) -> Path:
    return raw_dir / "bird_mini_dev" / "dev_databases"


def resolve_db_path(db_id: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidates = [
            root / db_id / f"{db_id}.sqlite",
            root / db_id / "sqlite" / f"{db_id}.sqlite",
            root / f"{db_id}.sqlite",
        ]
        for path in candidates:
            if path.exists():
                return path
    return None


def normalize_example(raw: dict[str, Any], index: int) -> dict[str, str]:
    db_id = raw.get("db_id")
    question = raw.get("question")
    sql = raw.get("SQL", raw.get("sql", raw.get("query")))
    if not db_id or not question or not sql:
        missing = [
            name
            for name, value in [("db_id", db_id), ("question", question), ("sql", sql)]
            if not value
        ]
        raise DataError(f"Example {index} missing required field(s): {', '.join(missing)}")

    difficulty = str(raw.get("difficulty") or "unknown").lower()
    if difficulty not in {"simple", "moderate", "challenging"}:
        difficulty = "unknown"

    question_id = raw.get("question_id", raw.get("id"))
    if question_id is None:
        question_id = f"{db_id}:{index}"

    return {
        "question_id": str(question_id),
        "db_id": str(db_id),
        "question": str(question),
        "evidence": str(raw.get("evidence") or ""),
        "sql": str(sql),
        "difficulty": difficulty,
    }


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def open_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    db_path = db_path.resolve()
    uri = "file:" + quote(str(db_path), safe="/:") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def introspect_schema(db_path: Path) -> str:
    """Return a compact textual schema for prompt construction."""

    if not db_path.exists():
        raise DataError(f"Missing database: {db_path}")

    lines: list[str] = []
    with closing(open_readonly_sqlite(db_path)) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (table_name,) in tables:
            cols = conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
            col_parts = []
            for _, col_name, col_type, notnull, default, pk in cols:
                part = f"{col_name} {col_type or 'TEXT'}"
                if pk:
                    part += " PRIMARY KEY"
                if notnull:
                    part += " NOT NULL"
                if default is not None:
                    part += f" DEFAULT {default}"
                col_parts.append(part)
            lines.append(f"Table {table_name}: " + ", ".join(col_parts))

            fks = conn.execute(
                f"PRAGMA foreign_key_list({quote_ident(table_name)})"
            ).fetchall()
            for fk in fks:
                # id, seq, table, from, to, on_update, on_delete, match
                lines.append(f"  FK {table_name}.{fk[3]} -> {fk[2]}.{fk[4]}")

    return "\n".join(lines)


def build_processed_records(
    raw_records: list[dict[str, Any]],
    db_roots: list[Path],
    limit: int | None = None,
    require_db: bool = True,
    split_name: str = "data",
) -> tuple[list[dict[str, Any]], int]:
    processed: list[dict[str, Any]] = []
    skipped_missing_db = 0
    records = raw_records[:limit] if limit is not None else raw_records

    for index, raw in enumerate(records):
        example = normalize_example(raw, index)
        db_path = resolve_db_path(example["db_id"], db_roots)
        if db_path is None:
            skipped_missing_db += 1
            if require_db:
                expected = db_roots[0] / example["db_id"] / f"{example['db_id']}.sqlite"
                raise DataError(f"Missing database: {expected}")
            continue
        example["db_path"] = str(db_path)
        example["schema"] = introspect_schema(db_path)
        processed.append(example)

    if not processed:
        raise DataError(
            f"No usable {split_name} records after database checks "
            f"(skipped_missing_db={skipped_missing_db})."
        )

    return processed, skipped_missing_db


def stable_holdout_key(db_id: str) -> str:
    return hashlib.sha1(f"autoresearch-trio-holdout-v1:{db_id}".encode()).hexdigest()


def split_train_holdout(
    raw_records: list[dict[str, Any]],
    db_roots: list[Path],
    eval_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int]:
    if eval_size <= 0:
        raise DataError("holdout eval size must be positive")

    usable_by_db: dict[str, list[dict[str, Any]]] = {}
    skipped_missing_db = 0
    for index, raw in enumerate(raw_records):
        example = normalize_example(raw, index)
        if resolve_db_path(example["db_id"], db_roots) is None:
            skipped_missing_db += 1
            continue
        usable_by_db.setdefault(example["db_id"], []).append(raw)

    total_usable = sum(len(rows) for rows in usable_by_db.values())
    if total_usable <= eval_size:
        raise DataError(
            f"Need more than {eval_size} usable train records for train-holdout split; "
            f"found {total_usable}."
        )

    holdout_db_ids: list[str] = []
    eval_raw: list[dict[str, Any]] = []
    for db_id in sorted(usable_by_db, key=stable_holdout_key):
        holdout_db_ids.append(db_id)
        need = eval_size - len(eval_raw)
        eval_raw.extend(usable_by_db[db_id][:need])
        if len(eval_raw) >= eval_size:
            break

    holdout_set = set(holdout_db_ids)
    train_raw = [
        raw
        for index, raw in enumerate(raw_records)
        if normalize_example(raw, index)["db_id"] not in holdout_set
    ]
    if not train_raw:
        raise DataError("train-holdout split left no train records")

    return train_raw, eval_raw, holdout_db_ids, skipped_missing_db


def prepare_data(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_PROCESSED_DIR,
    limit_train: int | None = None,
    limit_eval: int | None = None,
    check_only: bool = False,
    eval_source: str = "train-holdout",
    holdout_eval_size: int = 50,
) -> dict[str, Any]:
    train_file = find_train_file(raw_dir)

    if train_file is None:
        raise DataError(
            "Missing train data file. Expected one of: "
            f"{raw_dir / 'bird23_train_filtered' / 'train.jsonl'} or train.json"
        )

    train_raw = read_json_or_jsonl(train_file)
    train_roots = candidate_train_db_roots(raw_dir)
    eval_file: Path | None = None
    eval_skipped = 0
    holdout_db_ids: list[str] = []

    if eval_source == "train-holdout":
        target_eval_size = limit_eval if limit_eval is not None else holdout_eval_size
        train_raw, eval_raw, holdout_db_ids, train_holdout_skipped = split_train_holdout(
            train_raw,
            train_roots,
            target_eval_size,
        )
        eval_roots = train_roots
        eval_dataset = f"{TRAIN_DATASET_NAME}:train-holdout"
    elif eval_source == "mini-dev":
        eval_file = find_eval_file(raw_dir)
        if eval_file is None:
            raise DataError(
                "Missing eval data file. Expected one of: "
                f"{raw_dir / 'bird_mini_dev' / 'mini_dev_sqlite.jsonl'} or mini_dev_sqlite.json"
            )
        eval_raw = read_json_or_jsonl(eval_file)
        eval_roots = [eval_db_root(raw_dir)]
        eval_dataset = EVAL_DATASET_NAME
        train_holdout_skipped = 0
    else:
        raise DataError(f"Unsupported eval source: {eval_source}")

    train_processed, train_skipped = build_processed_records(
        train_raw,
        train_roots,
        limit=limit_train,
        require_db=False,
        split_name="train",
    )
    eval_processed, eval_skipped = build_processed_records(
        eval_raw,
        eval_roots,
        limit=limit_eval if eval_source == "mini-dev" else None,
        require_db=True,
        split_name="eval",
    )

    manifest = {
        "project": PROJECT_NAME,
        "train_dataset": TRAIN_DATASET_NAME,
        "eval_dataset": eval_dataset,
        "eval_source": eval_source,
        "train_file": str(train_file),
        "eval_file": str(eval_file or train_file),
        "train_records": len(train_processed),
        "eval_records": len(eval_processed),
        "train_skipped_missing_db": train_skipped,
        "eval_skipped_missing_db": eval_skipped,
        "train_holdout_skipped_missing_db": train_holdout_skipped,
        "holdout_db_ids": holdout_db_ids,
        "time_budget": TIME_BUDGET,
        "max_seq_len": MAX_SEQ_LEN,
    }

    if not check_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "train.jsonl", train_processed)
        write_jsonl(out_dir / "eval.jsonl", eval_processed)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return manifest


def strip_sql_comments(sql: str) -> str:
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def has_multiple_statements(sql: str) -> bool:
    stripped = sql.strip()
    if not stripped:
        return False
    if stripped.endswith(";"):
        stripped = stripped[:-1]
    return ";" in stripped


def classify_sql_safety(sql: str) -> tuple[bool, bool, str]:
    """Return (safe, unsafe, reason)."""

    sql = strip_sql_comments(sql)
    if not sql:
        return False, False, "empty_sql"
    upper = sql.upper().lstrip()
    first = upper.split(None, 1)[0] if upper.split(None, 1) else ""
    if first in DESTRUCTIVE_KEYWORDS:
        return False, True, f"destructive_keyword:{first}"
    if has_multiple_statements(sql):
        return False, True, "multiple_statements"
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, False, "not_select_or_with"
    return True, False, ""


def execute_sql(
    db_path: Path,
    sql: str,
    timeout_seconds: float = SQL_TIMEOUT_SECONDS,
    enforce_safety: bool = True,
) -> SqlRunResult:
    if enforce_safety:
        safe, unsafe, reason = classify_sql_safety(sql)
        if not safe:
            return SqlRunResult(ok=False, rows=[], elapsed_ms=0.0, error=reason, unsafe=unsafe)

    deadline = time.perf_counter() + timeout_seconds
    start = time.perf_counter()
    try:
        with closing(open_readonly_sqlite(db_path)) as conn:
            def progress_handler() -> int:
                return 1 if time.perf_counter() > deadline else 0

            conn.set_progress_handler(progress_handler, 1000)
            rows = conn.execute(sql).fetchall()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return SqlRunResult(ok=True, rows=rows, elapsed_ms=elapsed_ms)
    except sqlite3.Error as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SqlRunResult(ok=False, rows=[], elapsed_ms=elapsed_ms, error=str(exc))


def normalize_rows_for_set(rows: list[tuple[Any, ...]]) -> set[tuple[str, ...]]:
    return {tuple(repr(cell) for cell in row) for row in rows}


def execution_match(pred_rows: list[tuple[Any, ...]], gold_rows: list[tuple[Any, ...]]) -> bool:
    return normalize_rows_for_set(pred_rows) == normalize_rows_for_set(gold_rows)


def soft_f1(pred_rows: list[tuple[Any, ...]], gold_rows: list[tuple[Any, ...]]) -> float:
    pred_cells = Counter(repr(cell) for row in pred_rows for cell in row)
    gold_cells = Counter(repr(cell) for row in gold_rows for cell in row)
    if not pred_cells and not gold_cells:
        return 1.0
    if not pred_cells or not gold_cells:
        return 0.0
    overlap = sum((pred_cells & gold_cells).values())
    precision = overlap / sum(pred_cells.values())
    recall = overlap / sum(gold_cells.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ves_score(gold_ms: float, pred_ms: float) -> float:
    if pred_ms <= 0:
        ratio = 2.0
    else:
        ratio = gold_ms / pred_ms
    if ratio >= 2:
        reward = 1.25
    elif ratio >= 1:
        reward = 1.0
    elif ratio >= 0.5:
        reward = 0.75
    elif ratio >= 0.25:
        reward = 0.5
    else:
        reward = 0.25
    return math.sqrt(reward) * 100.0


def load_predictions(predictions_path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for row in read_json_or_jsonl(predictions_path):
        question_id = row.get("question_id", row.get("id"))
        if question_id is None:
            continue
        predictions[str(question_id)] = str(row.get("sql", row.get("SQL", "")) or "")
    return predictions


def evaluate_sql_predictions(
    predictions_path: Path | str,
    eval_path: Path | str = DEFAULT_PROCESSED_DIR / "eval.jsonl",
    db_root: Path | str | None = None,
    output_path: Path | str | None = None,
    timeout_seconds: float = SQL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    predictions_path = Path(predictions_path)
    eval_path = Path(eval_path)
    predictions = load_predictions(predictions_path)
    eval_records = read_jsonl(eval_path)

    bucket_totals = {"simple": 0, "moderate": 0, "challenging": 0}
    bucket_correct = {"simple": 0, "moderate": 0, "challenging": 0}
    details = []
    ex_scores: list[float] = []
    f1_scores: list[float] = []
    ves_scores: list[float] = []
    valid_count = 0
    unsafe_count = 0
    pred_latency: list[float] = []

    for record in eval_records:
        qid = str(record["question_id"])
        pred_sql = predictions.get(qid, "")
        gold_sql = str(record["sql"])
        raw_db_path = record.get("db_path")
        db_path = Path(str(raw_db_path)) if raw_db_path else None
        if db_root is not None:
            resolved = resolve_db_path(str(record["db_id"]), [Path(db_root)])
            if resolved is not None:
                db_path = resolved
            else:
                db_path = Path(db_root) / record["db_id"] / f"{record['db_id']}.sqlite"
        if db_path is None:
            raise DataError(
                f"Missing database path for question_id={qid}; "
                "provide db_path in eval records or pass db_root."
            )
        if not db_path.exists():
            raise DataError(f"Missing database: {db_path}")

        pred = execute_sql(db_path, pred_sql, timeout_seconds=timeout_seconds, enforce_safety=True)
        gold = execute_sql(db_path, gold_sql, timeout_seconds=timeout_seconds, enforce_safety=False)

        if pred.unsafe:
            unsafe_count += 1
        if pred.ok:
            valid_count += 1
            pred_latency.append(pred.elapsed_ms)

        is_correct = bool(pred.ok and gold.ok and execution_match(pred.rows, gold.rows))
        ex_value = 1.0 if is_correct else 0.0
        f1_value = soft_f1(pred.rows, gold.rows) if pred.ok and gold.ok else 0.0
        ves_value = ves_score(gold.elapsed_ms, pred.elapsed_ms) if is_correct else 0.0

        ex_scores.append(ex_value)
        f1_scores.append(f1_value)
        ves_scores.append(ves_value)

        difficulty = record.get("difficulty", "unknown")
        if difficulty in bucket_totals:
            bucket_totals[difficulty] += 1
            bucket_correct[difficulty] += int(is_correct)

        details.append(
            {
                "question_id": qid,
                "db_id": record["db_id"],
                "difficulty": difficulty,
                "ex": ex_value,
                "soft_f1": f1_value,
                "r_ves": ves_value,
                "pred_ok": pred.ok,
                "pred_error": pred.error,
                "gold_ok": gold.ok,
                "gold_error": gold.error,
                "pred_latency_ms": pred.elapsed_ms,
            }
        )

    total = len(eval_records)
    if total == 0:
        raise DataError(f"No eval records in {eval_path}")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics: dict[str, Any] = {
        "ex": mean(ex_scores),
        "soft_f1": mean(f1_scores),
        "r_ves": mean(ves_scores),
        "valid_sql_rate": valid_count / total,
        "unsafe_sql_rate": unsafe_count / total,
        "latency_ms": mean(pred_latency),
        "total": total,
        "details": details,
    }
    for difficulty in ("simple", "moderate", "challenging"):
        denom = bucket_totals[difficulty]
        metrics[f"{difficulty}_ex"] = bucket_correct[difficulty] / denom if denom else 0.0

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return metrics


def append_results_csv(path: Path | str, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    clean_row = {field: row.get(field, "") for field in RESULTS_FIELDS}
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(clean_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BIRD data for autoresearch-trio")
    parser.add_argument("--check", action="store_true", help="validate inputs without writing processed files")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download HF JSON records and official database packages before preparing",
    )
    parser.add_argument(
        "--download-json-only",
        action="store_true",
        help="download only Hugging Face JSON/JSONL records",
    )
    parser.add_argument(
        "--download-databases",
        action="store_true",
        help="download only official SQLite database packages",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download requested assets and exit without running preparation",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download and re-extract assets even if local files exist",
    )
    parser.add_argument("--limit-train", type=int, default=None, help="limit processed train examples")
    parser.add_argument("--limit-eval", type=int, default=None, help="limit processed eval examples")
    parser.add_argument(
        "--eval-source",
        choices=["train-holdout", "mini-dev"],
        default="train-holdout",
        help="use held-out train databases or official Mini-Dev as eval",
    )
    parser.add_argument(
        "--holdout-eval-size",
        type=int,
        default=50,
        help="number of eval examples when --eval-source train-holdout",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.download or args.download_json_only or args.download_databases:
            download_data_assets(
                raw_dir=args.raw_dir,
                force=args.force_download,
                include_records=args.download or args.download_json_only,
                include_databases=args.download or args.download_databases,
                include_eval=args.eval_source == "mini-dev",
            )
            if args.download_only:
                print("Download complete.")
                return 0

        manifest = prepare_data(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            limit_train=args.limit_train,
            limit_eval=args.limit_eval,
            check_only=args.check,
            eval_source=args.eval_source,
            holdout_eval_size=args.holdout_eval_size,
        )
    except DataError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.check:
        print("Check passed.")
    else:
        print(f"Prepared data under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

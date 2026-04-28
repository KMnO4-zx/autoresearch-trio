#!/usr/bin/env python3
"""TRIO text-to-SQL training script.

This is intentionally the main experiment surface, mirroring
karpathy/autoresearch: agents should primarily edit this file when trying new
training ideas.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from prepare import (
    DEFAULT_PROCESSED_DIR,
    PROJECT_NAME,
    RESULTS_CSV,
    TIME_BUDGET,
    append_results_csv,
    evaluate_sql_predictions,
    read_jsonl,
)

# ---------------------------------------------------------------------------
# Experiment knobs. Agents are expected to edit these directly.
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_RANK = 64
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
MAX_STEPS = 100
EVAL_LIMIT = 120
SAMPLE_MAX_TOKENS = 512
SAMPLE_TEMPERATURE = 0.0
SHUFFLE_TRAINING_RECORDS = True
TRAIN_SHUFFLE_SEED = 42
TRAIN_MLP = True
TRAIN_ATTN = True
TRAIN_UNEMBED = False
WEIGHT_DECAY = 0.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-12
SWANLAB_ACTIVE = False


@dataclass
class SFTArrays:
    input_tokens: list[int]
    target_tokens: list[int]
    weights: list[float]
    prompt_tokens: int
    completion_tokens: int


class TinyTokenizer:
    """Small deterministic tokenizer used only for dry-run smoke tests."""

    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        prefix = [1] if add_special_tokens else []
        return prefix + [2 + (ord(ch) % 200) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(chr((int(i) - 2) % 200) for i in ids if int(i) >= 2)


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def default_run_tag() -> str:
    return time.strftime("%b%d-sql").lower()


def build_prompt(example: dict[str, Any], include_gold: bool = False) -> str:
    prompt = f"""You are a text-to-SQL model. Generate one SQLite query.

Database schema:
{example.get("schema", "")}

Question:
{example.get("question", "")}

Evidence:
{example.get("evidence", "")}

Rules:
- Return only SQL.
- Use SQLite syntax.
- Use table and column names exactly as they appear in the schema.
- Use Evidence for column meanings, aliases, units, and literal values when relevant.
- Do not explain.
- Do not use destructive statements.

SQL:
"""
    if include_gold:
        return prompt + str(example.get("sql", ""))
    return prompt


def tokenizer_encode(tokenizer: Any, text: str, add_special_tokens: bool) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        ids = tokenizer.encode(text)
    return list(ids)


def eos_token_id(tokenizer: Any) -> int | None:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is not None:
        return int(value)
    try:
        return int(tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0])
    except Exception:
        return None


def patch_pytrio_tokenizer() -> None:
    """Use transformers tokenizer loading when pytrio's modelscope loader is unavailable."""

    def transformers_get_tokenizer(base_model: str):
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(
                base_model,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception:
            print(f"Tokenizer not found locally, downloading with transformers: {base_model}")
            return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    try:
        import pytrio._utils as pytrio_utils
        import pytrio.lib.training_client as training_client_module

        pytrio_utils.get_tokenizer = transformers_get_tokenizer
        training_client_module.get_tokenizer = transformers_get_tokenizer
    except Exception as exc:
        print(f"WARNING: could not patch pytrio tokenizer loader: {exc}")


def build_sft_arrays(
    example: dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
) -> SFTArrays | None:
    prompt = build_prompt(example)
    completion = str(example["sql"]).strip()
    if not completion.endswith(";"):
        completion += ";"

    prompt_ids = tokenizer_encode(tokenizer, prompt, add_special_tokens=True)
    completion_ids = tokenizer_encode(tokenizer, completion, add_special_tokens=False)
    eos = eos_token_id(tokenizer)
    if eos is not None:
        completion_ids.append(eos)

    tokens = prompt_ids + completion_ids
    weights = [0.0] * len(prompt_ids) + [1.0] * len(completion_ids)
    if len(tokens) < 2 or len(tokens) - 1 > max_seq_len:
        return None

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    target_weights = weights[1:]
    if not (len(input_tokens) == len(target_tokens) == len(target_weights)):
        raise AssertionError("TRIO datum arrays must have matching lengths")

    return SFTArrays(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        weights=target_weights,
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(completion_ids),
    )


def extract_sql(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    match = re.search(r"\b(SELECT|WITH)\b", text, flags=re.IGNORECASE)
    if not match:
        return ""
    sql = text[match.start() :].strip()
    semicolon = sql.find(";")
    if semicolon >= 0:
        sql = sql[: semicolon + 1]
    return sql.strip()


def make_batches(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if SHUFFLE_TRAINING_RECORDS:
        records = list(records)
        random.Random(TRAIN_SHUFFLE_SEED).shuffle(records)
    index = 0
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(records[index % len(records)])
            index += 1
        yield batch


def build_eval_prompt_tokens(
    tokenizer: Any,
    example: dict[str, Any],
    max_seq_len: int,
) -> tuple[list[int] | None, int]:
    prompt_ids = tokenizer_encode(tokenizer, build_prompt(example), add_special_tokens=True)
    if len(prompt_ids) > max_seq_len:
        return None, len(prompt_ids)
    return prompt_ids, len(prompt_ids)


def metric_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    def num(name: str, default: float = 0.0) -> float:
        try:
            return float(row.get(name, default) or default)
        except (TypeError, ValueError):
            return default

    # Higher is better except unsafe and latency.
    return (
        num("ex"),
        num("soft_f1"),
        num("r_ves"),
        -num("unsafe_sql_rate"),
        -num("latency_ms", 1e12),
    )


def infer_status(results_path: Path, current_row: dict[str, Any], requested_status: str) -> str:
    if requested_status != "auto":
        return requested_status
    if not results_path.exists() or results_path.stat().st_size == 0:
        return "baseline"
    with results_path.open("r", encoding="utf-8", newline="") as f:
        previous = [
            row
            for row in csv.DictReader(f)
            if row.get("status") in {"baseline", "keep"} and row.get("ex") not in {None, ""}
        ]
    if not previous:
        return "baseline"
    best = max(metric_key(row) for row in previous)
    return "keep" if metric_key(current_row) > best else "discard"


def init_swanlab(args: argparse.Namespace, commit: str, run_id: str):
    global SWANLAB_ACTIVE
    SWANLAB_ACTIVE = False
    if args.swanlab_mode == "disabled":
        return None
    import swanlab

    run = swanlab.init(
        project=PROJECT_NAME,
        experiment_name=run_id,
        group=args.run_tag,
        tags=["trio", "bird", "text-to-sql", "autoresearch"],
        mode=args.swanlab_mode,
        logdir=args.swanlab_logdir,
        public=False,
        config={
            "base_model": args.base_model,
            "lora_rank": args.rank,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "time_budget": TIME_BUDGET,
            "dataset_train": "birdsql/bird23-train-filtered",
            "dataset_eval": "birdsql/bird_mini_dev",
            "git_commit": commit,
            "max_seq_len": args.max_seq_len,
            "eval_limit": args.eval_limit,
        },
    )
    SWANLAB_ACTIVE = True
    return run


def log_swanlab(data: dict[str, Any], step: int | None = None) -> None:
    if not SWANLAB_ACTIVE:
        return
    try:
        import swanlab

        swanlab.log(data, step=step)
    except Exception as exc:
        print(f"WARNING: swanlab.log failed: {exc}")


def finish_swanlab() -> None:
    if not SWANLAB_ACTIVE:
        return
    try:
        import swanlab

        swanlab.finish()
    except Exception:
        pass


def load_processed(path: Path, name: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name}: {path}. Run `uv run prepare.py` after placing BIRD data under data/raw/."
        )
    return read_jsonl(path)


def write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_dry_run(args: argparse.Namespace) -> None:
    tokenizer = TinyTokenizer()
    example = {
        "question_id": "dry-run-1",
        "db_id": "dry_run",
        "question": "How many users are there?",
        "evidence": "",
        "sql": "SELECT COUNT(*) FROM users;",
        "difficulty": "simple",
        "schema": "Table users: id INTEGER PRIMARY KEY, name TEXT",
    }
    arrays = build_sft_arrays(example, tokenizer, args.max_seq_len)
    if arrays is None:
        raise RuntimeError("Dry-run SFT example unexpectedly exceeded max_seq_len")
    print(
        "dry_run_datum: "
        f"input={len(arrays.input_tokens)} target={len(arrays.target_tokens)} "
        f"weights={len(arrays.weights)} prompt_tokens={arrays.prompt_tokens}"
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db_dir = root / "dry_run"
        db_dir.mkdir()
        db_path = db_dir / "dry_run.sqlite"
        import sqlite3

        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT)")
            conn.executemany("INSERT INTO users(name) VALUES (?)", [("a",), ("b",)])
            conn.commit()
        eval_path = root / "eval.jsonl"
        pred_path = root / "predictions.jsonl"
        metrics_path = root / "metrics.json"
        eval_record = dict(example)
        eval_record["db_path"] = str(db_path)
        write_predictions(eval_path, [eval_record])
        write_predictions(
            pred_path,
            [{"question_id": "dry-run-1", "db_id": "dry_run", "sql": "SELECT COUNT(*) FROM users;"}],
        )
        metrics = evaluate_sql_predictions(pred_path, eval_path, output_path=metrics_path)
        print(f"dry_run_eval_ex: {metrics['ex']:.6f}")
    print("Dry run passed.")


def train_and_eval(args: argparse.Namespace) -> dict[str, Any]:
    from pytrio import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient

    patch_pytrio_tokenizer()

    train_records = load_processed(args.train_path, "processed train data")
    eval_records = load_processed(args.eval_path, "processed eval data")[: args.eval_limit]
    if not train_records:
        raise RuntimeError(f"No training records in {args.train_path}")
    if not eval_records:
        raise RuntimeError(f"No eval records in {args.eval_path}")

    service = ServiceClient()
    train_client = service.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
        seed=args.seed,
        train_mlp=TRAIN_MLP,
        train_attn=TRAIN_ATTN,
        train_unembed=TRAIN_UNEMBED,
    )
    tokenizer = train_client.get_tokenizer()

    skipped_long = 0
    step = 0
    examples_seen = 0
    last_loss = 0.0
    t0 = time.perf_counter()
    train_start = time.perf_counter()
    deadline = train_start + TIME_BUDGET
    batches = make_batches(train_records, args.batch_size)
    consecutive_empty_batches = 0
    max_empty_batches = max(100, len(train_records) // max(args.batch_size, 1) + 1)

    while step < args.max_steps and time.perf_counter() < deadline:
        data: list[Any] = []
        for example in next(batches):
            arrays = build_sft_arrays(example, tokenizer, args.max_seq_len)
            if arrays is None:
                skipped_long += 1
                continue
            data.append(
                Datum(
                    model_input=ModelInput.from_ints(arrays.input_tokens),
                    loss_fn_inputs={
                        "target_tokens": arrays.target_tokens,
                        "weights": arrays.weights,
                    },
                )
            )
        if not data:
            consecutive_empty_batches += 1
            if consecutive_empty_batches >= max_empty_batches:
                raise RuntimeError(
                    "No usable training examples under max_seq_len="
                    f"{args.max_seq_len}; skipped_long={skipped_long}."
                )
            continue
        consecutive_empty_batches = 0

        fb = train_client.forward_backward(data, loss_fn="cross_entropy").result()
        metrics = getattr(fb, "metrics", {}) or {}
        if "loss" in metrics:
            last_loss = float(metrics["loss"])
        elif metrics:
            last_loss = float(next(iter(metrics.values())))

        train_client.optim_step(
            AdamParams(
                learning_rate=args.learning_rate,
                beta1=ADAM_BETA1,
                beta2=ADAM_BETA2,
                eps=ADAM_EPS,
                weight_decay=WEIGHT_DECAY,
            )
        ).result()

        step += 1
        examples_seen += len(data)
        elapsed = time.perf_counter() - train_start
        log_swanlab(
            {
                "train/loss": last_loss,
                "train/examples": examples_seen,
                "train/elapsed_seconds": elapsed,
                "train/skipped_long": skipped_long,
            },
            step=step,
        )
        print(f"step {step:05d} | loss {last_loss:.6f} | elapsed {elapsed:.1f}s")

    training_seconds = time.perf_counter() - train_start
    save_name = f"{args.run_id}-step{step}"
    save_response = train_client.save_weights_for_sampler(save_name).result()
    cloud_path = save_response.path
    sampler = train_client.create_sampling_client(model_path=cloud_path)

    output_dir = args.output_dir / args.run_id
    pred_path = output_dir / "predictions.jsonl"
    eval_subset_path = output_dir / "eval.jsonl"
    metrics_path = output_dir / "metrics.json"

    predictions = []
    skipped_eval_long = 0
    sampled_eval = 0
    sampling_params = SamplingParams(
        max_tokens=args.sample_max_tokens,
        seed=args.seed,
        temperature=args.sample_temperature,
        top_p=1.0,
    )
    for index, example in enumerate(eval_records, start=1):
        prompt_ids, prompt_len = build_eval_prompt_tokens(tokenizer, example, args.max_seq_len)
        if prompt_ids is None:
            skipped_eval_long += 1
            predictions.append(
                {
                    "question_id": example["question_id"],
                    "db_id": example["db_id"],
                    "sql": "",
                    "skipped_reason": "prompt_too_long",
                    "prompt_tokens": prompt_len,
                }
            )
            continue
        response = sampler.sample(
            ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=sampling_params,
        ).result()
        text = response.sequences[0].text if response.sequences else ""
        sampled_eval += 1
        predictions.append(
            {
                "question_id": example["question_id"],
                "db_id": example["db_id"],
                "sql": extract_sql(text),
            }
        )
        if index % 25 == 0:
            print(f"sampled {index}/{len(eval_records)} eval examples")

    write_predictions(eval_subset_path, eval_records)
    write_predictions(pred_path, predictions)
    eval_metrics = evaluate_sql_predictions(pred_path, eval_subset_path, output_path=metrics_path)
    total_seconds = time.perf_counter() - t0

    return {
        **eval_metrics,
        "cloud_path": cloud_path,
        "sft_loss": last_loss,
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "steps": step,
        "examples_seen": examples_seen,
        "skipped_long": skipped_long,
        "skipped_eval_long": skipped_eval_long,
        "sampled_eval": sampled_eval,
    }


def print_summary(metrics: dict[str, Any], swanlab_run_id: str) -> None:
    print("---")
    print(f"ex:               {metrics.get('ex', 0.0):.6f}")
    print(f"soft_f1:          {metrics.get('soft_f1', 0.0):.6f}")
    print(f"r_ves:            {metrics.get('r_ves', 0.0):.6f}")
    print(f"valid_sql_rate:   {metrics.get('valid_sql_rate', 0.0):.6f}")
    print(f"unsafe_sql_rate:  {metrics.get('unsafe_sql_rate', 0.0):.6f}")
    print(f"latency_ms:       {metrics.get('latency_ms', 0.0):.1f}")
    print(f"training_seconds: {metrics.get('training_seconds', 0.0):.1f}")
    print(f"total_seconds:    {metrics.get('total_seconds', 0.0):.1f}")
    print(f"cloud_path:        {metrics.get('cloud_path', '')}")
    print(f"swanlab_run_id:    {swanlab_run_id}")
    print(f"sft_loss:          {metrics.get('sft_loss', 0.0)}")
    print(f"eval_total:        {metrics.get('total', 0)}")
    print(f"sampled_eval:      {metrics.get('sampled_eval', 0)}")
    print(f"skipped_long:      {metrics.get('skipped_long', 0)}")
    print(f"skipped_eval_long: {metrics.get('skipped_eval_long', 0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TRIO autoresearch text-to-SQL experiment")
    parser.add_argument("--dry-run", action="store_true", help="validate local datum/eval pipeline only")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--eval-limit", type=int, default=EVAL_LIMIT)
    parser.add_argument("--sample-max-tokens", type=int, default=SAMPLE_MAX_TOKENS)
    parser.add_argument("--sample-temperature", type=float, default=SAMPLE_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default=default_run_tag())
    parser.add_argument("--status", default="auto", choices=["auto", "baseline", "keep", "discard", "crash"])
    parser.add_argument("--description", default="baseline")
    parser.add_argument(
        "--swanlab-mode",
        default="cloud",
        choices=["cloud", "local", "offline", "disabled"],
    )
    parser.add_argument("--swanlab-logdir", default="swanlog")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_PROCESSED_DIR / "train.jsonl")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_PROCESSED_DIR / "eval.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--results-csv", type=Path, default=Path(RESULTS_CSV))
    args = parser.parse_args()
    commit = git_commit()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.run_id = f"{args.run_tag}-{commit}-{timestamp}"
    return args


def main() -> int:
    args = parse_args()
    if args.dry_run:
        run_dry_run(args)
        return 0

    commit = git_commit()
    run = None
    swanlab_run_id = ""
    try:
        run = init_swanlab(args, commit, args.run_id)
        swanlab_run_id = getattr(run, "id", "") or ""
    except Exception as exc:
        append_results_csv(
            args.results_csv,
            {
                "commit": commit,
                "run_tag": args.run_tag,
                "swanlab_run_id": "",
                "status": "crash",
                "description": f"swanlab init failed: {exc}",
            },
        )
        print(f"ERROR: SwanLab init failed. Run `swanlab login` or use --swanlab-mode disabled. {exc}")
        return 1

    try:
        metrics = train_and_eval(args)
        eval_payload = {
            "eval/ex": metrics["ex"],
            "eval/soft_f1": metrics["soft_f1"],
            "eval/r_ves": metrics["r_ves"],
            "eval/valid_sql_rate": metrics["valid_sql_rate"],
            "eval/unsafe_sql_rate": metrics["unsafe_sql_rate"],
            "eval/latency_ms": metrics["latency_ms"],
            "eval/simple_ex": metrics["simple_ex"],
            "eval/moderate_ex": metrics["moderate_ex"],
            "eval/challenging_ex": metrics["challenging_ex"],
            "eval/eval_limit": args.eval_limit,
            "eval/total": metrics["total"],
            "eval/sampled_eval": metrics["sampled_eval"],
            "eval/skipped_eval_long": metrics["skipped_eval_long"],
            "train/skipped_long": metrics["skipped_long"],
        }
        log_swanlab(eval_payload)
        if metrics.get("cloud_path") and SWANLAB_ACTIVE:
            try:
                import swanlab

                log_swanlab({"checkpoint/cloud_path": swanlab.Text(str(metrics["cloud_path"]))})
            except Exception:
                pass

        row = {
            "commit": commit,
            "run_tag": args.run_tag,
            "swanlab_run_id": swanlab_run_id,
            "ex": f"{metrics['ex']:.6f}",
            "soft_f1": f"{metrics['soft_f1']:.6f}",
            "r_ves": f"{metrics['r_ves']:.6f}",
            "valid_sql_rate": f"{metrics['valid_sql_rate']:.6f}",
            "unsafe_sql_rate": f"{metrics['unsafe_sql_rate']:.6f}",
            "latency_ms": f"{metrics['latency_ms']:.1f}",
            "cloud_path": metrics.get("cloud_path", ""),
            "sft_loss": metrics.get("sft_loss", ""),
            "eval_limit": args.eval_limit,
            "eval_total": metrics.get("total", ""),
            "sampled_eval": metrics.get("sampled_eval", ""),
            "steps": metrics.get("steps", ""),
            "examples_seen": metrics.get("examples_seen", ""),
            "skipped_long": metrics.get("skipped_long", ""),
            "skipped_eval_long": metrics.get("skipped_eval_long", ""),
            "description": args.description,
        }
        row["status"] = infer_status(args.results_csv, row, args.status)
        append_results_csv(args.results_csv, row)
        print_summary(metrics, swanlab_run_id)
        return 0
    except Exception as exc:
        append_results_csv(
            args.results_csv,
            {
                "commit": commit,
                "run_tag": args.run_tag,
                "swanlab_run_id": swanlab_run_id,
                "status": "crash",
                "description": f"{args.description}: {exc}",
            },
        )
        print(f"ERROR: {exc}")
        return 1
    finally:
        if run is not None:
            finish_swanlab()


if __name__ == "__main__":
    raise SystemExit(main())

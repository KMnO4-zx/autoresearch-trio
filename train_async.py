#!/usr/bin/env python3
"""Async TRIO text-to-SQL training script.

This is an optional faster runner for the same fixed benchmark harness used by
train.py. It overlaps local batch preparation/logging with TRIO cloud execution
by submitting async forward/backward and optimizer tasks in a small pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any

from prepare import DEFAULT_PROCESSED_DIR, PROJECT_NAME, RESULTS_CSV, TIME_BUDGET
from prepare import append_results_csv, evaluate_sql_predictions, read_jsonl
from train import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    BASE_MODEL,
    BATCH_SIZE,
    EVAL_LIMIT,
    LORA_RANK,
    LEARNING_RATE,
    MAX_STEPS,
    SAMPLE_MAX_TOKENS,
    SAMPLE_TEMPERATURE,
    TRAIN_ATTN,
    TRAIN_MLP,
    TRAIN_UNEMBED,
    WEIGHT_DECAY,
    build_eval_prompt_tokens,
    build_prompt,
    build_sft_arrays,
    default_run_tag,
    extract_sql,
    git_commit,
    infer_status,
    init_swanlab,
    load_processed,
    log_swanlab,
    make_batches,
    patch_pytrio_tokenizer,
    print_summary,
    run_dry_run,
    tokenizer_encode,
    write_predictions,
)


ASYNC_TRAIN_PIPELINE_DEPTH = 4
ASYNC_EVAL_CONCURRENCY = 16


def extract_loss(fb_result: Any, data: list[Any]) -> float | None:
    metrics = getattr(fb_result, "metrics", {}) or {}
    if "loss" in metrics:
        return float(metrics["loss"])
    if metrics:
        return float(next(iter(metrics.values())))

    # Fallback for SDK versions that return token logprobs but no aggregate loss.
    try:
        weighted_logprob = 0.0
        weight_total = 0.0
        for datum, output in zip(data, fb_result.loss_fn_outputs, strict=True):
            logprobs = output.get("logprobs")
            logprob_values = logprobs.tolist()
            weights = datum.loss_fn_inputs["weights"].tolist()
            weighted_logprob += sum(float(lp) * float(w) for lp, w in zip(logprob_values, weights))
            weight_total += sum(float(w) for w in weights)
        if weight_total > 0:
            return -weighted_logprob / weight_total
    except Exception:
        return None
    return None


async def drain_train_task(
    task: asyncio.Task[dict[str, Any]],
    train_start: float,
) -> dict[str, Any]:
    result = await task
    elapsed = time.perf_counter() - train_start
    payload = {
        "train/examples": result["examples_seen"],
        "train/elapsed_seconds": elapsed,
        "train/skipped_long": result["skipped_long"],
        "train/async_inflight": result["inflight_at_submit"],
    }
    if result["loss"] is not None:
        payload["train/loss"] = result["loss"]
    log_swanlab(payload, step=result["step"])
    loss_text = "nan" if result["loss"] is None else f"{result['loss']:.6f}"
    print(f"step {result['step']:05d} | loss {loss_text} | elapsed {elapsed:.1f}s")
    return result


async def train_and_eval_async(args: argparse.Namespace) -> dict[str, Any]:
    import pytrio as trio

    patch_pytrio_tokenizer()

    train_records = load_processed(args.train_path, "processed train data")
    eval_records = load_processed(args.eval_path, "processed eval data")[: args.eval_limit]
    if not train_records:
        raise RuntimeError(f"No training records in {args.train_path}")
    if not eval_records:
        raise RuntimeError(f"No eval records in {args.eval_path}")

    service = trio.ServiceClient()
    train_client = await service.create_lora_training_client_async(
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
    pending: set[asyncio.Task[dict[str, Any]]] = set()

    async def submit_step(
        data: list[Any],
        submitted_step: int,
        submitted_examples_seen: int,
        submitted_skipped_long: int,
        inflight_at_submit: int,
    ) -> dict[str, Any]:
        fb_future = await train_client.forward_backward_async(data, loss_fn="cross_entropy")
        optim_future = await train_client.optim_step_async(
            trio.AdamParams(
                learning_rate=args.learning_rate,
                beta1=ADAM_BETA1,
                beta2=ADAM_BETA2,
                eps=ADAM_EPS,
                weight_decay=WEIGHT_DECAY,
            )
        )
        fb_result = await fb_future
        await optim_future
        return {
            "step": submitted_step,
            "examples_seen": submitted_examples_seen,
            "skipped_long": submitted_skipped_long,
            "loss": extract_loss(fb_result, data),
            "inflight_at_submit": inflight_at_submit,
        }

    while step < args.max_steps and time.perf_counter() < deadline:
        data: list[Any] = []
        for example in next(batches):
            arrays = build_sft_arrays(example, tokenizer, args.max_seq_len)
            if arrays is None:
                skipped_long += 1
                continue
            data.append(
                trio.Datum(
                    model_input=trio.ModelInput.from_ints(arrays.input_tokens),
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

        step += 1
        examples_seen += len(data)
        pending.add(
            asyncio.create_task(
                submit_step(
                    data,
                    step,
                    examples_seen,
                    skipped_long,
                    inflight_at_submit=len(pending) + 1,
                )
            )
        )

        if len(pending) >= args.train_pipeline_depth:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                result = await drain_train_task(task, train_start)
                if result["loss"] is not None:
                    last_loss = float(result["loss"])

    submitted_steps = step
    if pending:
        done, _ = await asyncio.wait(pending)
        for task in done:
            result = await drain_train_task(task, train_start)
            if result["loss"] is not None:
                last_loss = float(result["loss"])

    training_seconds = time.perf_counter() - train_start
    save_name = f"{args.run_id}-async-step{submitted_steps}"
    save_response_future = await train_client.save_weights_for_sampler_async(save_name)
    save_response = await save_response_future
    cloud_path = save_response.path
    sampler = await train_client.create_sampling_client_async(model_path=cloud_path)

    output_dir = args.output_dir / args.run_id
    pred_path = output_dir / "predictions.jsonl"
    eval_subset_path = output_dir / "eval.jsonl"
    metrics_path = output_dir / "metrics.json"

    sampling_params = trio.SamplingParams(
        max_tokens=args.sample_max_tokens,
        seed=args.seed,
        temperature=args.sample_temperature,
        top_p=1.0,
    )
    predictions: list[dict[str, Any]] = []
    skipped_eval_long = 0
    sampled_eval = 0

    async def sample_one(index: int, example: dict[str, Any]) -> dict[str, Any]:
        prompt_ids, prompt_len = build_eval_prompt_tokens(tokenizer, example, args.max_seq_len)
        if prompt_ids is None:
            return {
                "question_id": example["question_id"],
                "db_id": example["db_id"],
                "sql": "",
                "skipped_reason": "prompt_too_long",
                "prompt_tokens": prompt_len,
                "index": index,
            }
        future = await sampler.sample_async(
            trio.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=sampling_params,
        )
        response = await future
        text = response.sequences[0].text if response.sequences else ""
        return {
            "question_id": example["question_id"],
            "db_id": example["db_id"],
            "sql": extract_sql(text),
            "index": index,
        }

    for start in range(0, len(eval_records), args.eval_concurrency):
        chunk = eval_records[start : start + args.eval_concurrency]
        tasks = [
            sample_one(start + offset + 1, example)
            for offset, example in enumerate(chunk)
        ]
        for row in await asyncio.gather(*tasks):
            if row.get("skipped_reason") == "prompt_too_long":
                skipped_eval_long += 1
            else:
                sampled_eval += 1
            row.pop("index", None)
            predictions.append(row)
        print(f"sampled {min(start + len(chunk), len(eval_records))}/{len(eval_records)} eval examples")

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
        "steps": submitted_steps,
        "examples_seen": examples_seen,
        "skipped_long": skipped_long,
        "skipped_eval_long": skipped_eval_long,
        "sampled_eval": sampled_eval,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run async TRIO autoresearch text-to-SQL experiment")
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
    parser.add_argument("--description", default="async baseline")
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
    parser.add_argument("--train-pipeline-depth", type=int, default=ASYNC_TRAIN_PIPELINE_DEPTH)
    parser.add_argument("--eval-concurrency", type=int, default=ASYNC_EVAL_CONCURRENCY)
    args = parser.parse_args()
    commit = git_commit()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.run_id = f"{args.run_tag}-{commit}-{timestamp}-async"
    return args


async def async_main() -> int:
    args = parse_args()
    if args.dry_run:
        run_dry_run(args)
        return 0
    if args.train_pipeline_depth < 1:
        print("ERROR: --train-pipeline-depth must be >= 1")
        return 1
    if args.eval_concurrency < 1:
        print("ERROR: --eval-concurrency must be >= 1")
        return 1

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
                "description": f"async swanlab init failed: {exc}",
            },
        )
        print(f"ERROR: SwanLab init failed. Run `swanlab login` or use --swanlab-mode disabled. {exc}")
        return 1

    try:
        metrics = await train_and_eval_async(args)
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
            "train/async_pipeline_depth": args.train_pipeline_depth,
            "eval/async_concurrency": args.eval_concurrency,
        }
        log_swanlab(eval_payload)

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
            from train import finish_swanlab

            finish_swanlab()


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())

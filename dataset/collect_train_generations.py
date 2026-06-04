#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset/collect_train_generations.py

Purpose:
  For GPQA, MATH-500, and DROP-800 training sets, call a 6-model OpenRouter pool
  and collect repeated stochastic generations.

Input:
  data/gpqa/uncertainty_train_rewritten.jsonl
  data/math-500/uncertainty_train_rewritten.jsonl
  data/drop-800/uncertainty_train_rewritten.jsonl

Each input record should contain:
  - id
  - dataset
  - question
  - context
  - choices
  - answer / answer_index / answer_text
  - prompt_variants: list of rewritten questions

Output:
  data/<dataset>/train_generations.jsonl
  data/<dataset>/train_generation_errors.jsonl
  data/<dataset>/train_generation_summary.json

One output line = one model response for:
  dataset × query × prompt_variant × model × decoding_sample

Default configuration:
  - 6 models
  - 5 prompt variants per query
  - 5 stochastic decoding samples per prompt variant
  - only uncertainty_train_rewritten.jsonl is used

Resume:
  Rerun the same command. The script loads existing train_generations.jsonl,
  checks completed unique keys, and only continues missing generations.

Usage:
  export OPENROUTER_API_KEY=xxx

  python dataset/collect_train_generations.py \
    --data_dir data \
    --datasets gpqa math-500 drop-800 \
    --num_prompt_variants 5 \
    --num_decodes 5 \
    --max_workers 12

Notes:
  - Do NOT hard-code API keys in this file.
  - Increase --max_workers cautiously because OpenRouter/provider rate limits vary.
"""

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


DEFAULT_MODEL_POOL = [
    "google/gemma-3-12b-it",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3-32b",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-chat-v3.1",
]


# -----------------------------
# IO utilities
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc

    return records


def write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def append_jsonl(record: Dict[str, Any], path: Path, lock: threading.Lock) -> None:
    ensure_dir(path.parent)
    line = json.dumps(record, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def write_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def clean_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    text = str(x).strip()
    return text if text else None


def compact_text(text: Optional[str], max_chars: int) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]"


# -----------------------------
# Key / resume utilities
# -----------------------------


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def make_generation_key(
    dataset: str,
    query_id: str,
    model: str,
    prompt_variant_id: int,
    decode_id: int,
) -> str:
    raw = f"{dataset}||{query_id}||{model}||{prompt_variant_id}||{decode_id}"
    return stable_hash(raw, n=24)


def is_valid_generation_record(record: Dict[str, Any]) -> bool:
    if not record.get("generation_key"):
        return False
    if not record.get("model"):
        return False
    if record.get("prompt_variant_id") is None:
        return False
    if record.get("decode_id") is None:
        return False
    output_text = clean_text(record.get("output_text"))
    if not output_text:
        return False
    return True


def load_completed_generations(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Treat train_generations.jsonl as an append log.
    If duplicate generation_key exists, keep the latest valid record.
    """
    completed: Dict[str, Dict[str, Any]] = {}

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            if not is_valid_generation_record(record):
                continue

            key = str(record["generation_key"])
            completed[key] = record

    return completed


# -----------------------------
# Prompt construction
# -----------------------------


def extract_prompt_variants(
    record: Dict[str, Any],
    num_prompt_variants: int,
    include_original: bool,
) -> List[Dict[str, Any]]:
    """
    Returns prompt variants as:
      [{"prompt_variant_id": int, "question": str, "source": "rewrite/original"}]

    Default:
      only use rewritten variants, with variant ids from existing prompt_variants.

    If include_original=True:
      original question is inserted as prompt_variant_id=-1.
    """
    variants: List[Dict[str, Any]] = []

    if include_original:
        original_question = clean_text(record.get("question"))
        if original_question:
            variants.append(
                {
                    "prompt_variant_id": -1,
                    "question": original_question,
                    "source": "original",
                }
            )

    raw_variants = record.get("prompt_variants", [])
    if not isinstance(raw_variants, list):
        raw_variants = []

    for idx, item in enumerate(raw_variants):
        if isinstance(item, dict):
            question = clean_text(item.get("question"))
            variant_id = item.get("variant_id", idx)
        elif isinstance(item, str):
            question = clean_text(item)
            variant_id = idx
        else:
            continue

        if question is None:
            continue

        try:
            variant_id = int(variant_id)
        except Exception:
            variant_id = idx

        variants.append(
            {
                "prompt_variant_id": variant_id,
                "question": question,
                "source": "rewrite",
            }
        )

    # Deduplicate by question text while preserving order.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for variant in variants:
        key = variant["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)

    if include_original:
        # Original + num_prompt_variants rewrites.
        max_total = num_prompt_variants + 1
    else:
        max_total = num_prompt_variants

    return deduped[:max_total]


def format_choices(choices: Any) -> str:
    if choices is None:
        return ""

    if not isinstance(choices, list):
        return str(choices)

    lines = []
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def build_user_prompt(record: Dict[str, Any], variant_question: str) -> str:
    dataset = record.get("dataset")
    task_type = record.get("task_type")
    context = clean_text(record.get("context"))
    choices = record.get("choices")

    if dataset == "gpqa" or task_type == "multiple_choice":
        choice_block = format_choices(choices)
        return (
            "Answer the following multiple-choice question.\n\n"
            "Question:\n"
            f"{variant_question}\n\n"
            "Choices:\n"
            f"{choice_block}\n\n"
            "Return your final answer in this exact format at the end:\n"
            "Final Answer: <one option letter>\n"
        )

    if dataset == "math-500" or task_type == "free_form_math":
        return (
            "Solve the following mathematics problem.\n\n"
            "Problem:\n"
            f"{variant_question}\n\n"
            "After any reasoning, put only the final result on the last line in this exact format:\n"
            "Final Answer: <answer>\n"
            "Do not write any prose after the final answer line.\n"
        )

    if dataset == "drop-800" or task_type == "reading_comprehension":
        return (
            "Answer the question using only the passage.\n\n"
            "Passage:\n"
            f"{compact_text(context, max_chars=12000)}\n\n"
            "Question:\n"
            f"{variant_question}\n\n"
            "Return your final answer in this exact format at the end:\n"
            "Final Answer: <short answer>\n"
        )

    # Fallback.
    if context:
        return (
            "Answer the question using the provided context.\n\n"
            "Context:\n"
            f"{compact_text(context, max_chars=12000)}\n\n"
            "Question:\n"
            f"{variant_question}\n\n"
            "Return your final answer in this exact format at the end:\n"
            "Final Answer: <answer>\n"
        )

    return (
        "Answer the following question.\n\n"
        f"{variant_question}\n\n"
        "Return your final answer in this exact format at the end:\n"
        "Final Answer: <answer>\n"
    )


def build_messages(record: Dict[str, Any], variant_question: str) -> List[Dict[str, str]]:
    system = (
        "You are answering benchmark questions for an LLM routing experiment.\n"
        "Follow the user instructions exactly.\n"
        "You may reason internally, but the final response must contain a clearly marked "
        "`Final Answer:` line.\n"
        "Do not mention that this is a paraphrase or dataset example."
    )

    user = build_user_prompt(record, variant_question)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def dataset_max_tokens(record: Dict[str, Any], default_max_tokens: int) -> int:
    dataset = record.get("dataset")
    task_type = record.get("task_type")

    if dataset == "gpqa" or task_type == "multiple_choice":
        return min(default_max_tokens, 768)

    if dataset == "math-500" or task_type == "free_form_math":
        return min(default_max_tokens, 3072)

    if dataset == "drop-800" or task_type == "reading_comprehension":
        return min(default_max_tokens, 512)

    return default_max_tokens


# -----------------------------
# OpenRouter client
# -----------------------------


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    referer: Optional[str],
    title: Optional[str],
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    if extra_headers:
        headers.update(extra_headers)

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=body,
        timeout=timeout,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenRouter request failed with HTTP {response.status_code}: "
            f"{response.text[:1200]}"
        )

    return response.json()


def parse_response_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    # Some APIs may return structured content.
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()

    return str(content).strip()


def extract_usage(data: Dict[str, Any]) -> Dict[str, Any]:
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost": usage.get("cost"),
    }


def extract_finish_reason(data: Dict[str, Any]) -> Optional[str]:
    choices = data.get("choices", [])
    if not choices:
        return None
    return choices[0].get("finish_reason")


# -----------------------------
# Task creation and execution
# -----------------------------


def build_task_list(
    dataset_name: str,
    records: Sequence[Dict[str, Any]],
    models: Sequence[str],
    num_prompt_variants: int,
    num_decodes: int,
    include_original: bool,
    completed: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    tasks: List[Dict[str, Any]] = []

    stats = {
        "records": len(records),
        "expected_total": 0,
        "completed_existing": 0,
        "missing_to_run": 0,
        "records_with_too_few_variants": 0,
    }

    for record in records:
        query_id = str(record.get("id"))
        dataset = str(record.get("dataset") or dataset_name)

        variants = extract_prompt_variants(
            record=record,
            num_prompt_variants=num_prompt_variants,
            include_original=include_original,
        )

        required_variant_count = num_prompt_variants + (1 if include_original else 0)
        if len(variants) < required_variant_count:
            stats["records_with_too_few_variants"] += 1

        for variant in variants:
            variant_id = int(variant["prompt_variant_id"])
            variant_question = str(variant["question"])

            for model in models:
                for decode_id in range(num_decodes):
                    key = make_generation_key(
                        dataset=dataset,
                        query_id=query_id,
                        model=model,
                        prompt_variant_id=variant_id,
                        decode_id=decode_id,
                    )

                    stats["expected_total"] += 1

                    if key in completed:
                        stats["completed_existing"] += 1
                        continue

                    tasks.append(
                        {
                            "generation_key": key,
                            "dataset": dataset,
                            "query_id": query_id,
                            "model": model,
                            "prompt_variant_id": variant_id,
                            "prompt_variant_source": variant["source"],
                            "variant_question": variant_question,
                            "decode_id": decode_id,
                            "record": record,
                        }
                    )

    stats["missing_to_run"] = len(tasks)
    return tasks, stats


def generation_task(
    task: Dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> Dict[str, Any]:
    record = task["record"]
    model = task["model"]
    variant_question = task["variant_question"]

    messages = build_messages(record, variant_question=variant_question)
    max_tokens = dataset_max_tokens(record, default_max_tokens=args.max_tokens)

    last_error: Optional[Exception] = None

    for attempt in range(args.max_retries):
        try:
            # Slight temperature jitter across retries only, not across decode ids.
            temperature = args.temperature

            data = call_openrouter(
                messages=messages,
                model=model,
                api_key=api_key,
                temperature=temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
                timeout=args.timeout,
                referer=args.openrouter_referer,
                title=args.openrouter_title,
            )

            output_text = parse_response_text(data)
            if not output_text:
                raise ValueError("Empty output_text")

            usage = extract_usage(data)

            out = {
                "generation_key": task["generation_key"],
                "dataset": task["dataset"],
                "query_id": task["query_id"],
                "model": model,
                "prompt_variant_id": task["prompt_variant_id"],
                "prompt_variant_source": task["prompt_variant_source"],
                "decode_id": task["decode_id"],
                "input_question": variant_question,
                "original_question": record.get("question"),
                "context": record.get("context"),
                "choices": record.get("choices"),
                "answer": record.get("answer"),
                "answer_index": record.get("answer_index"),
                "answer_text": record.get("answer_text"),
                "category": record.get("category"),
                "task_type": record.get("task_type"),
                "score_type": record.get("score_type"),
                "messages": messages if args.save_messages else None,
                "output_text": output_text,
                "finish_reason": extract_finish_reason(data),
                "openrouter_response_id": data.get("id"),
                "usage": usage,
                "request_config": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": max_tokens,
                },
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            if not args.save_messages:
                out.pop("messages", None)

            return out

        except Exception as exc:
            last_error = exc
            sleep_s = args.retry_sleep * (2 ** attempt) + random.random()
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed generation after {args.max_retries} retries. "
        f"key={task['generation_key']} model={model} "
        f"query_id={task['query_id']} variant={task['prompt_variant_id']} "
        f"decode={task['decode_id']} last_error={last_error}"
    )


# -----------------------------
# Dataset processing
# -----------------------------


def compact_output_file(
    output_path: Path,
    completed: Dict[str, Dict[str, Any]],
    expected_keys_order: Sequence[str],
) -> None:
    """
    Rewrites output JSONL in deterministic expected-key order, dropping duplicates.
    """
    ordered_records: List[Dict[str, Any]] = []
    missing = 0

    for key in expected_keys_order:
        record = completed.get(key)
        if record is None:
            missing += 1
            continue
        ordered_records.append(record)

    write_jsonl(ordered_records, output_path)

    if missing:
        print(f"Compaction warning: {missing} expected keys still missing in {output_path}")


def process_dataset(dataset_name: str, args: argparse.Namespace, api_key: str) -> None:
    dataset_dir = Path(args.data_dir) / dataset_name
    input_path = dataset_dir / args.input_filename
    output_path = dataset_dir / args.output_filename
    error_path = dataset_dir / args.error_filename
    summary_path = dataset_dir / args.summary_filename

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    records = read_jsonl(input_path)

    if args.limit_records is not None:
        records = records[: args.limit_records]

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if error_path.exists():
            error_path.unlink()

    completed = load_completed_generations(output_path)

    tasks, stats = build_task_list(
        dataset_name=dataset_name,
        records=records,
        models=args.models,
        num_prompt_variants=args.num_prompt_variants,
        num_decodes=args.num_decodes,
        include_original=args.include_original,
        completed=completed,
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Records: {stats['records']}")
    print(f"Expected generations: {stats['expected_total']}")
    print(f"Completed existing: {stats['completed_existing']}")
    print(f"Missing to run: {stats['missing_to_run']}")
    print(f"Records with too few prompt variants: {stats['records_with_too_few_variants']}")
    print(f"Models: {len(args.models)}")
    print(f"Prompt variants per query: {args.num_prompt_variants}")
    print(f"Decodes per prompt variant: {args.num_decodes}")
    print(f"Max workers: {args.max_workers}")

    if not tasks:
        expected_keys_order = build_expected_key_order(
            dataset_name=dataset_name,
            records=records,
            models=args.models,
            num_prompt_variants=args.num_prompt_variants,
            num_decodes=args.num_decodes,
            include_original=args.include_original,
        )
        compact_output_file(output_path, completed, expected_keys_order)
        write_dataset_summary(
            summary_path=summary_path,
            dataset_name=dataset_name,
            records=records,
            args=args,
            completed=completed,
            expected_total=stats["expected_total"],
            failed_count=0,
        )
        return

    output_lock = threading.Lock()
    error_lock = threading.Lock()
    failed_count = 0

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(generation_task, task, args, api_key): task
                for task in tasks
            }

            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc=f"Generating {dataset_name}",
            ):
                task = future_to_task[future]

                try:
                    record = future.result()
                    append_jsonl(record, output_path, output_lock)
                    completed[record["generation_key"]] = record

                    if args.sleep > 0:
                        time.sleep(args.sleep)

                except Exception as exc:
                    failed_count += 1
                    error_record = {
                        "generation_key": task["generation_key"],
                        "dataset": task["dataset"],
                        "query_id": task["query_id"],
                        "model": task["model"],
                        "prompt_variant_id": task["prompt_variant_id"],
                        "decode_id": task["decode_id"],
                        "error": str(exc),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    append_jsonl(error_record, error_path, error_lock)

                    if args.fail_fast:
                        raise

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed outputs have already been appended. "
            "Rerun the same command to resume missing generations."
        )
        raise

    # Reload output file to be robust against any process-local inconsistency.
    completed = load_completed_generations(output_path)

    expected_keys_order = build_expected_key_order(
        dataset_name=dataset_name,
        records=records,
        models=args.models,
        num_prompt_variants=args.num_prompt_variants,
        num_decodes=args.num_decodes,
        include_original=args.include_original,
    )

    missing_after = [key for key in expected_keys_order if key not in completed]

    compact_output_file(output_path, completed, expected_keys_order)

    write_dataset_summary(
        summary_path=summary_path,
        dataset_name=dataset_name,
        records=records,
        args=args,
        completed=completed,
        expected_total=len(expected_keys_order),
        failed_count=failed_count,
    )

    if failed_count:
        print(f"Warning: {failed_count} failed requests. See {error_path}")

    if missing_after:
        print(f"Warning: {len(missing_after)} generations still missing.")
        print(f"First missing keys: {missing_after[:10]}")
        print("Rerun the same command to continue.")
    else:
        print(f"Dataset {dataset_name} complete.")


def build_expected_key_order(
    dataset_name: str,
    records: Sequence[Dict[str, Any]],
    models: Sequence[str],
    num_prompt_variants: int,
    num_decodes: int,
    include_original: bool,
) -> List[str]:
    keys: List[str] = []

    for record in records:
        query_id = str(record.get("id"))
        dataset = str(record.get("dataset") or dataset_name)

        variants = extract_prompt_variants(
            record=record,
            num_prompt_variants=num_prompt_variants,
            include_original=include_original,
        )

        for variant in variants:
            variant_id = int(variant["prompt_variant_id"])

            for model in models:
                for decode_id in range(num_decodes):
                    keys.append(
                        make_generation_key(
                            dataset=dataset,
                            query_id=query_id,
                            model=model,
                            prompt_variant_id=variant_id,
                            decode_id=decode_id,
                        )
                    )

    return keys


def write_dataset_summary(
    summary_path: Path,
    dataset_name: str,
    records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    completed: Dict[str, Dict[str, Any]],
    expected_total: int,
    failed_count: int,
) -> None:
    by_model: Dict[str, int] = {m: 0 for m in args.models}
    by_dataset: Dict[str, int] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    cost_seen = False

    for record in completed.values():
        model = str(record.get("model"))
        by_model[model] = by_model.get(model, 0) + 1

        dataset = str(record.get("dataset"))
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1

        usage = record.get("usage") or {}
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            tokens = usage.get("total_tokens")
            cost = usage.get("cost")

            if isinstance(prompt_tokens, int):
                total_prompt_tokens += prompt_tokens
            if isinstance(completion_tokens, int):
                total_completion_tokens += completion_tokens
            if isinstance(tokens, int):
                total_tokens += tokens
            if isinstance(cost, (int, float)):
                total_cost += float(cost)
                cost_seen = True

    summary = {
        "dataset": dataset_name,
        "num_records": len(records),
        "models": list(args.models),
        "num_models": len(args.models),
        "num_prompt_variants": args.num_prompt_variants,
        "include_original": args.include_original,
        "num_decodes": args.num_decodes,
        "expected_total_generations": expected_total,
        "completed_generations": len(completed),
        "missing_generations": max(expected_total - len(completed), 0),
        "failed_count_this_run": failed_count,
        "by_model_completed": by_model,
        "by_dataset_completed": by_dataset,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost if cost_seen else None,
        },
        "request_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_workers": args.max_workers,
        },
        "files": {
            "input": args.input_filename,
            "output": args.output_filename,
            "errors": args.error_filename,
            "summary": args.summary_filename,
        },
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    write_json(summary, summary_path)


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gpqa", "math-500", "drop-800"],
        choices=["gpqa", "math-500", "drop-800"],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_POOL,
        help="OpenRouter model ids.",
    )

    parser.add_argument(
        "--input_filename",
        type=str,
        default="uncertainty_train_rewritten.jsonl",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="train_generations.jsonl",
    )
    parser.add_argument(
        "--error_filename",
        type=str,
        default="train_generation_errors.jsonl",
    )
    parser.add_argument(
        "--summary_filename",
        type=str,
        default="train_generation_summary.json",
    )

    parser.add_argument(
        "--num_prompt_variants",
        type=int,
        default=5,
        help="Number of rewritten prompt variants per query to use.",
    )
    parser.add_argument(
        "--include_original",
        action="store_true",
        help="Also sample the original question as prompt_variant_id=-1.",
    )
    parser.add_argument(
        "--num_decodes",
        type=int,
        default=5,
        help="Number of stochastic decoding samples per query/model/prompt variant.",
    )

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)

    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Number of parallel OpenRouter requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep after each successful request completion.",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key. Prefer setting OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--openrouter_referer",
        type=str,
        default=os.environ.get("OPENROUTER_HTTP_REFERER"),
    )
    parser.add_argument(
        "--openrouter_title",
        type=str,
        default=os.environ.get("OPENROUTER_X_TITLE", "distribution-aware-routing"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output/error files and regenerate everything.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately on the first failed request.",
    )
    parser.add_argument(
        "--save_messages",
        action="store_true",
        help="Save full chat messages in each generation record.",
    )
    parser.add_argument(
        "--limit_records",
        type=int,
        default=None,
        help="Debug option: only process the first N records per dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api_key."
        )

    if args.num_prompt_variants <= 0:
        raise ValueError("--num_prompt_variants must be positive.")
    if args.num_decodes <= 0:
        raise ValueError("--num_decodes must be positive.")
    if args.max_workers <= 0:
        raise ValueError("--max_workers must be positive.")
    if not args.models:
        raise ValueError("--models must contain at least one model.")

    print("Model pool:")
    for model in args.models:
        print(f"  - {model}")

    for dataset_name in args.datasets:
        process_dataset(dataset_name, args=args, api_key=args.api_key)

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset/rewrite_queries.py

Use OpenRouter GPT-4o to generate meaning-preserving query rewrites for:
  - GPQA
  - MATH-500
  - DROP-800

Features:
1. Parallel rewriting with ThreadPoolExecutor.
2. Resume support.
3. On every restart, checks each query's valid prompt_variants count.
   - If count >= --num_variants, skip.
   - If count < --num_variants, continue generating missing rewrites.
4. Existing partial rewrites are preserved and used as "avoid duplicates" examples.
5. Writes progress incrementally to all_rewritten.jsonl, so interrupted runs can resume.

Expected input layout:

  data/gpqa/
    all.jsonl
    uncertainty_train.jsonl
    test.jsonl

  data/math-500/
    all.jsonl
    uncertainty_train.jsonl
    test.jsonl

  data/drop-800/
    all.jsonl
    uncertainty_train.jsonl
    test.jsonl

Output layout:

  data/gpqa/
    all_rewritten.jsonl
    uncertainty_train_rewritten.jsonl
    test_rewritten.jsonl
    rewrite_errors.jsonl

  data/math-500/
    all_rewritten.jsonl
    uncertainty_train_rewritten.jsonl
    test_rewritten.jsonl
    rewrite_errors.jsonl

  data/drop-800/
    all_rewritten.jsonl
    uncertainty_train_rewritten.jsonl
    test_rewritten.jsonl
    rewrite_errors.jsonl

Usage:

  export OPENROUTER_API_KEY=xxx

  python dataset/rewrite_queries.py \
    --data_dir data \
    --datasets gpqa math-500 drop-800 \
    --num_variants 5 \
    --model openai/gpt-4o \
    --max_workers 8

If interrupted, rerun the same command. It will automatically continue only
records with fewer than --num_variants valid rewrites.
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# -----------------------------
# Basic IO
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

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


def append_jsonl(record: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def clean_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    text = str(x).strip()
    return text if text else None


# -----------------------------
# Resume helpers
# -----------------------------


def extract_variant_questions(
    record: Dict[str, Any],
    original_question: Optional[str] = None,
) -> List[str]:
    """
    Extract valid unique rewritten questions from a rewritten record.

    Supports:
      prompt_variants = [{"question": "..."}]
      prompt_variants = ["..."]
    """
    variants = record.get("prompt_variants", [])
    if not isinstance(variants, list):
        return []

    original_norm = (original_question or "").strip().lower()
    rewrites: List[str] = []
    seen = set()

    for item in variants:
        if isinstance(item, dict):
            text = clean_text(item.get("question"))
        elif isinstance(item, str):
            text = clean_text(item)
        else:
            text = None

        if not text:
            continue

        text = re.sub(r"\s+", " ", text).strip()
        key = text.lower()

        if original_norm and key == original_norm:
            continue
        if key in seen:
            continue

        seen.add(key)
        rewrites.append(text)

    return rewrites


def valid_variant_count(
    record: Dict[str, Any],
    original_question: Optional[str] = None,
) -> int:
    return len(extract_variant_questions(record, original_question=original_question))


def is_complete_rewritten_record(
    record: Dict[str, Any],
    num_variants: int,
    original_question: Optional[str] = None,
) -> bool:
    return valid_variant_count(record, original_question=original_question) >= num_variants


def load_best_rewrite_records(
    path: Path,
    original_question_by_id: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """
    Load existing all_rewritten.jsonl as an append log.

    If the same id appears multiple times, keep the version with the largest
    valid prompt_variants count. If tied, keep the later one.
    """
    best: Dict[str, Dict[str, Any]] = {}
    best_count: Dict[str, int] = {}

    if not path.exists():
        return best

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            record_id = str(record.get("id", ""))
            if not record_id:
                continue

            original_question = original_question_by_id.get(record_id)
            count = valid_variant_count(record, original_question=original_question)

            if record_id not in best or count >= best_count.get(record_id, -1):
                best[record_id] = record
                best_count[record_id] = count

    return best


# -----------------------------
# Prompt construction
# -----------------------------


def compact_text(text: Optional[str], max_chars: int) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]"


def dataset_instruction(record: Dict[str, Any]) -> str:
    dataset = record.get("dataset")
    task_type = record.get("task_type")
    score_type = record.get("score_type")

    if dataset == "gpqa":
        return (
            "This is a graduate-level multiple-choice science question. "
            "Rewrite only the question stem. Do not modify, reorder, remove, "
            "or paraphrase the answer choices. The correct option after rewriting "
            "must remain exactly the same."
        )

    if dataset == "math-500":
        return (
            "This is a free-form mathematics problem. Rewrite only the problem statement. "
            "Preserve every mathematical condition, number, variable, equation, unit, "
            "constraint, and requested quantity. The final mathematical answer must remain "
            "exactly the same. Do not simplify the problem, add hints, or reveal the answer."
        )

    if dataset == "drop-800":
        return (
            "This is a reading-comprehension question over a fixed passage. "
            "Rewrite only the question. Do not rewrite the passage. The rewritten question "
            "must be answerable from the same passage and must have the same gold answer. "
            "Do not add information that is not supported by the passage."
        )

    return (
        f"This is a dataset example with task_type={task_type} and score_type={score_type}. "
        "Rewrite only the query/question. Preserve the original answer target exactly."
    )


def build_messages(
    record: Dict[str, Any],
    num_new_rewrites: int,
    existing_rewrites: Sequence[str],
) -> List[Dict[str, str]]:
    question = record.get("question")
    context = record.get("context")
    choices = record.get("choices")
    answer = record.get("answer")
    answer_text = record.get("answer_text")
    answer_index = record.get("answer_index")
    category = record.get("category")
    task_type = record.get("task_type")
    score_type = record.get("score_type")

    payload = {
        "dataset": record.get("dataset"),
        "id": record.get("id"),
        "category": category,
        "task_type": task_type,
        "score_type": score_type,
        "original_question": question,
        "context_for_reference_only": compact_text(context, max_chars=4000),
        "choices_must_remain_unchanged": choices,
        "gold_answer_for_consistency_check_only": answer,
        "gold_answer_index_for_consistency_check_only": answer_index,
        "gold_answer_text_for_consistency_check_only": answer_text,
        "existing_rewrites_to_avoid": list(existing_rewrites),
        "num_new_rewrites": num_new_rewrites,
    }

    system = (
        "You are a careful dataset paraphrasing assistant for LLM routing experiments.\n"
        "Your job is to create meaning-preserving rewrites of input queries.\n\n"
        "Hard constraints:\n"
        "1. Do not change the correct answer.\n"
        "2. Do not make the query easier or harder.\n"
        "3. Do not add hints, explanations, chain-of-thought, or solution steps.\n"
        "4. Do not reveal or leak the gold answer.\n"
        "5. Do not change factual conditions, quantities, entities, dates, units, equations, or constraints.\n"
        "6. For multiple-choice examples, rewrite only the question stem; choices remain unchanged.\n"
        "7. For reading-comprehension examples, rewrite only the question; context remains unchanged.\n"
        "8. For math examples, preserve all mathematical semantics exactly.\n"
        "9. Produce natural but diverse rewrites.\n"
        "10. Avoid duplicating the original question and avoid duplicating existing rewrites.\n"
        "11. Output strict JSON only. No Markdown. No extra text.\n\n"
        'Required JSON schema:\n{"rewrites": ["rewrite 1", "rewrite 2", "..."]}'
    )

    user = (
        f"{dataset_instruction(record)}\n\n"
        f"Generate exactly {num_new_rewrites} new rewrites.\n"
        "The new rewrites must be different from the original question and from all existing rewrites.\n"
        "Return only strict JSON matching the required schema.\n\n"
        "Example payload:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# -----------------------------
# OpenRouter client
# -----------------------------


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")

    return json.loads(match.group(0))


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    referer: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
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
            f"{response.text[:1000]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]


# -----------------------------
# Rewrite validation and normalization
# -----------------------------


def normalize_rewrites(
    raw_rewrites: Any,
    original_question: str,
    existing_rewrites: Sequence[str],
    num_needed: int,
) -> List[str]:
    if not isinstance(raw_rewrites, list):
        raise ValueError(f"`rewrites` must be a list, got {type(raw_rewrites)}")

    existing_norm = {x.strip().lower() for x in existing_rewrites if x.strip()}
    original_norm = original_question.strip().lower()

    rewrites: List[str] = []
    seen = set(existing_norm)

    for item in raw_rewrites:
        if not isinstance(item, str):
            continue

        text = item.strip()
        text = re.sub(r"\s+", " ", text)

        if not text:
            continue

        key = text.lower()

        if key == original_norm:
            continue
        if key in seen:
            continue

        seen.add(key)
        rewrites.append(text)

    if len(rewrites) < num_needed:
        raise ValueError(
            f"Only got {len(rewrites)} valid new unique rewrites; expected {num_needed}"
        )

    return rewrites[:num_needed]


def merge_rewrites(
    original_question: str,
    existing_rewrites: Sequence[str],
    new_rewrites: Sequence[str],
    num_variants: int,
) -> List[str]:
    original_norm = original_question.strip().lower()

    merged: List[str] = []
    seen = set()

    for text in list(existing_rewrites) + list(new_rewrites):
        text = re.sub(r"\s+", " ", str(text).strip())
        if not text:
            continue

        key = text.lower()
        if key == original_norm:
            continue
        if key in seen:
            continue

        seen.add(key)
        merged.append(text)

        if len(merged) >= num_variants:
            break

    return merged


def build_rewritten_record(
    record: Dict[str, Any],
    rewrites: Sequence[str],
    previous_variant_count: int,
) -> Dict[str, Any]:
    out = dict(record)

    out["prompt_variants"] = []
    for i, rewritten_question in enumerate(rewrites):
        out["prompt_variants"].append(
            {
                "variant_id": i,
                "question": rewritten_question,
                "context": record.get("context"),
                "choices": record.get("choices"),
                "rewrite_method": "openrouter_gpt4o",
            }
        )

    out["rewrite_metadata"] = {
        "num_variants": len(rewrites),
        "previous_variant_count": previous_variant_count,
        "rewritten_field": "question",
        "context_changed": False,
        "choices_changed": False,
        "answer_changed": False,
        "resume_supported": True,
    }

    return out


def rewrite_one_record(
    record: Dict[str, Any],
    existing_rewrites: Sequence[str],
    args: argparse.Namespace,
    api_key: str,
) -> Dict[str, Any]:
    original_question = str(record.get("question", "")).strip()
    if not original_question:
        raise ValueError(f"Record {record.get('id')} has empty question")

    previous_count = len(existing_rewrites)
    num_needed = args.num_variants - previous_count

    if num_needed <= 0:
        final_rewrites = merge_rewrites(
            original_question=original_question,
            existing_rewrites=existing_rewrites,
            new_rewrites=[],
            num_variants=args.num_variants,
        )
        return build_rewritten_record(
            record=record,
            rewrites=final_rewrites,
            previous_variant_count=previous_count,
        )

    last_error: Optional[Exception] = None

    for attempt in range(args.max_retries):
        try:
            messages = build_messages(
                record=record,
                num_new_rewrites=num_needed,
                existing_rewrites=existing_rewrites,
            )

            raw_text = call_openrouter(
                messages=messages,
                model=args.model,
                api_key=api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                referer=args.openrouter_referer,
                title=args.openrouter_title,
            )

            obj = extract_json_object(raw_text)
            new_rewrites = normalize_rewrites(
                raw_rewrites=obj.get("rewrites"),
                original_question=original_question,
                existing_rewrites=existing_rewrites,
                num_needed=num_needed,
            )

            final_rewrites = merge_rewrites(
                original_question=original_question,
                existing_rewrites=existing_rewrites,
                new_rewrites=new_rewrites,
                num_variants=args.num_variants,
            )

            if len(final_rewrites) < args.num_variants:
                raise ValueError(
                    f"After merging, got {len(final_rewrites)} rewrites; "
                    f"expected {args.num_variants}"
                )

            return build_rewritten_record(
                record=record,
                rewrites=final_rewrites,
                previous_variant_count=previous_count,
            )

        except Exception as exc:
            last_error = exc
            sleep_s = args.retry_sleep * (2 ** attempt) + random.random()
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to rewrite record id={record.get('id')} after {args.max_retries} attempts. "
        f"Previous variants: {previous_count}. "
        f"Needed variants: {num_needed}. "
        f"Last error: {last_error}"
    )


# -----------------------------
# Dataset-level processing
# -----------------------------


def rewrite_all_file(dataset_dir: Path, args: argparse.Namespace, api_key: str) -> Path:
    input_path = dataset_dir / "all.jsonl"
    output_path = dataset_dir / "all_rewritten.jsonl"
    error_path = dataset_dir / "rewrite_errors.jsonl"

    records = read_jsonl(input_path)
    original_question_by_id = {
        str(record.get("id")): str(record.get("question", "")).strip()
        for record in records
    }

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if error_path.exists():
            error_path.unlink()
        best_by_id: Dict[str, Dict[str, Any]] = {}
    else:
        best_by_id = load_best_rewrite_records(
            output_path,
            original_question_by_id=original_question_by_id,
        )

    complete_count = 0
    incomplete_count = 0
    todo: List[Tuple[Dict[str, Any], List[str]]] = []

    for record in records:
        record_id = str(record.get("id"))
        original_question = original_question_by_id[record_id]
        existing_record = best_by_id.get(record_id)

        if existing_record is None:
            todo.append((record, []))
            continue

        existing_rewrites = extract_variant_questions(
            existing_record,
            original_question=original_question,
        )

        if len(existing_rewrites) >= args.num_variants:
            complete_count += 1
        else:
            incomplete_count += 1
            todo.append((record, existing_rewrites))

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total records: {len(records)}")
    print(f"Complete records: {complete_count}")
    print(f"Incomplete existing records: {incomplete_count}")
    print(f"Records to process: {len(todo)}")
    print(f"Max workers: {args.max_workers}")

    if not todo:
        ordered_records = []
        for record in records:
            record_id = str(record.get("id"))
            existing_record = best_by_id.get(record_id)
            if existing_record is not None:
                ordered_records.append(existing_record)

        write_jsonl(ordered_records, output_path)
        return output_path

    failures = 0

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_record: Dict[Any, Dict[str, Any]] = {}

            for record, existing_rewrites in todo:
                future = executor.submit(
                    rewrite_one_record,
                    record,
                    existing_rewrites,
                    args,
                    api_key,
                )
                future_to_record[future] = record

            for future in tqdm(
                as_completed(future_to_record),
                total=len(future_to_record),
                desc=f"Rewriting {dataset_dir.name}",
            ):
                record = future_to_record[future]
                record_id = str(record.get("id"))
                original_question = original_question_by_id[record_id]

                try:
                    rewritten = future.result()

                    count = valid_variant_count(
                        rewritten,
                        original_question=original_question,
                    )

                    if count < args.num_variants:
                        raise ValueError(
                            f"Record id={record_id} still has only {count} valid rewrites "
                            f"after rewriting."
                        )

                    append_jsonl(rewritten, output_path)
                    best_by_id[record_id] = rewritten

                    if args.sleep > 0:
                        time.sleep(args.sleep)

                except Exception as exc:
                    failures += 1
                    error_record = {
                        "id": record.get("id"),
                        "dataset": record.get("dataset"),
                        "error": str(exc),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    append_jsonl(error_record, error_path)

                    if args.fail_fast:
                        raise

    except KeyboardInterrupt:
        print(
            "\nInterrupted by user. Progress already appended to all_rewritten.jsonl. "
            "Rerun the same command to continue unfinished records."
        )
        raise

    ordered_best_records: List[Dict[str, Any]] = []
    still_incomplete: List[str] = []
    missing: List[str] = []

    for record in records:
        record_id = str(record.get("id"))
        original_question = original_question_by_id[record_id]
        best_record = best_by_id.get(record_id)

        if best_record is None:
            missing.append(record_id)
            continue

        ordered_best_records.append(best_record)

        if not is_complete_rewritten_record(
            best_record,
            num_variants=args.num_variants,
            original_question=original_question,
        ):
            still_incomplete.append(record_id)

    write_jsonl(ordered_best_records, output_path)

    if failures:
        print(f"Warning: {failures} records failed in {dataset_dir.name}. See {error_path}")

    if missing:
        print(f"Warning: {len(missing)} records missing rewrites in {dataset_dir.name}.")
        print(f"First missing ids: {missing[:10]}")

    if still_incomplete:
        print(
            f"Warning: {len(still_incomplete)} records still have fewer than "
            f"{args.num_variants} rewrites in {dataset_dir.name}."
        )
        print(f"First incomplete ids: {still_incomplete[:10]}")
        print("Rerun the same command to continue them.")

    return output_path


def materialize_split_from_all(
    dataset_dir: Path,
    split_name: str,
    rewritten_by_id: Dict[str, Dict[str, Any]],
    num_variants: int,
) -> None:
    split_path = dataset_dir / f"{split_name}.jsonl"
    output_path = dataset_dir / f"{split_name}_rewritten.jsonl"

    split_records = read_jsonl(split_path)
    out_records: List[Dict[str, Any]] = []
    missing_or_incomplete: List[str] = []

    for record in split_records:
        record_id = str(record.get("id"))
        original_question = str(record.get("question", "")).strip()
        rewritten = rewritten_by_id.get(record_id)

        if rewritten is None:
            missing_or_incomplete.append(record_id)
            continue

        if not is_complete_rewritten_record(
            rewritten,
            num_variants=num_variants,
            original_question=original_question,
        ):
            missing_or_incomplete.append(record_id)
            continue

        out_records.append(rewritten)

    if missing_or_incomplete:
        raise RuntimeError(
            f"{dataset_dir.name}/{split_name}: {len(missing_or_incomplete)} records are "
            f"missing complete rewrites. First ids: {missing_or_incomplete[:10]}. "
            f"Rerun the script to continue unfinished records."
        )

    write_jsonl(out_records, output_path)


def process_dataset(dataset_name: str, args: argparse.Namespace, api_key: str) -> None:
    dataset_dir = Path(args.data_dir) / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    all_path = dataset_dir / "all.jsonl"
    all_records = read_jsonl(all_path)
    original_question_by_id = {
        str(record.get("id")): str(record.get("question", "")).strip()
        for record in all_records
    }

    all_rewritten_path = rewrite_all_file(dataset_dir, args=args, api_key=api_key)
    all_rewritten = read_jsonl(all_rewritten_path)

    complete_by_id: Dict[str, Dict[str, Any]] = {}
    incomplete_ids: List[str] = []

    for rewritten in all_rewritten:
        record_id = str(rewritten.get("id"))
        original_question = original_question_by_id.get(record_id, "")

        if is_complete_rewritten_record(
            rewritten,
            num_variants=args.num_variants,
            original_question=original_question,
        ):
            complete_by_id[record_id] = rewritten
        else:
            incomplete_ids.append(record_id)

    missing_ids = [
        str(record.get("id"))
        for record in all_records
        if str(record.get("id")) not in complete_by_id
    ]

    if missing_ids:
        print(
            f"Dataset {dataset_name} is not fully complete yet: "
            f"{len(missing_ids)} records need more rewrites."
        )
        print(f"First unfinished ids: {missing_ids[:10]}")
        print("Split files will not be materialized until all records are complete.")
        return

    materialize_split_from_all(
        dataset_dir=dataset_dir,
        split_name="uncertainty_train",
        rewritten_by_id=complete_by_id,
        num_variants=args.num_variants,
    )
    materialize_split_from_all(
        dataset_dir=dataset_dir,
        split_name="test",
        rewritten_by_id=complete_by_id,
        num_variants=args.num_variants,
    )

    print(f"Finished dataset: {dataset_name}")


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
        "--num_variants",
        type=int,
        default=5,
        help="Number of rewritten query variants per original query.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="OpenRouter model id.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for rewrite diversity.",
    )

    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=2.0)

    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel OpenRouter requests.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help=(
            "Optional sleep after each successful completed future. "
            "Usually keep 0 when using parallel mode."
        ),
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
        help="Optional OpenRouter HTTP-Referer header.",
    )

    parser.add_argument(
        "--openrouter_title",
        type=str,
        default=os.environ.get("OPENROUTER_X_TITLE", "distribution-aware-routing"),
        help="Optional OpenRouter X-Title header.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing all_rewritten.jsonl and rewrite everything.",
    )

    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately on the first failed record.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api_key."
        )

    if args.num_variants <= 0:
        raise ValueError(f"--num_variants must be positive, got {args.num_variants}")

    if args.max_workers <= 0:
        raise ValueError(f"--max_workers must be positive, got {args.max_workers}")

    for dataset_name in args.datasets:
        print(f"\n=== Processing {dataset_name} ===")
        process_dataset(dataset_name, args=args, api_key=args.api_key)

    print("\nDone.")


if __name__ == "__main__":
    main()

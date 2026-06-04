#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare GPQA, MATH-500, and DROP-800 for distribution-aware LLM routing.

Current experimental design:
1. Use three compact datasets:
   - GPQA
   - MATH-500
   - DROP-800, sampled from DROP with a fixed random seed
2. For each prepared dataset, sample exactly 200 examples as the uncertainty/router
   training set, and use the remaining examples as the test set.
3. Normalize heterogeneous schemas into a shared JSONL format.
bd
Output directory layout:
   data/gpqa/
   data/math-500/
   data/drop-800/

Output files for each dataset:
   - all.jsonl
   - uncertainty_train.jsonl
   - test.jsonl
   - metadata.json

Recommended usage:
   python scripts/prepare_datasets.py \
       --output_dir data \
       --train_size 200 \
       --drop_max_records 800 \
       --seed 42

If GPQA access requires a token:
   HF_TOKEN=xxx python scripts/prepare_datasets.py
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm


# -----------------------------
# Generic utilities
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_int_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        value = x.strip()
    else:
        value = str(x).strip()
    return value if value else None


def get_first_available(row: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def load_hf_dataset(
    repo: str,
    config: Optional[str] = None,
    token: Optional[str] = None,
) -> DatasetDict:
    """
    Load a Hugging Face dataset, compatible with both newer and older datasets versions.
    """
    try:
        if config is None:
            ds = load_dataset(repo, token=token)
        else:
            ds = load_dataset(repo, config, token=token)
    except TypeError:
        if config is None:
            ds = load_dataset(repo, use_auth_token=token)
        else:
            ds = load_dataset(repo, config, use_auth_token=token)

    if isinstance(ds, Dataset):
        return DatasetDict({"train": ds})
    return ds


def add_source_split_column(ds_dict: DatasetDict) -> DatasetDict:
    """
    Preserve the original split after concatenation.
    """
    out = DatasetDict()
    for split_name, split_ds in ds_dict.items():
        out[split_name] = split_ds.map(
            lambda _: {"__source_split__": split_name},
            desc=f"Adding source split for {split_name}",
        )
    return out


def concatenate_selected_splits(
    ds: DatasetDict,
    split_names: Optional[Sequence[str]] = None,
) -> Tuple[Dataset, List[str]]:
    """
    Combine selected splits into one dataset.

    If split_names is None, empty, or ["all"], all available splits are combined.
    """
    available = list(ds.keys())

    if split_names is None or len(split_names) == 0 or split_names == ["all"]:
        selected = available
    else:
        missing = [s for s in split_names if s not in ds]
        if missing:
            raise ValueError(f"Missing splits {missing}. Available splits: {available}")
        selected = list(split_names)

    datasets = [ds[s] for s in selected]

    if len(datasets) == 1:
        return datasets[0], selected

    return concatenate_datasets(datasets), selected


def _group_key(record: Dict[str, Any], key: Optional[str]) -> str:
    if key is None:
        return "__all__"
    value = record.get(key)
    if value is None:
        return "__none__"
    if isinstance(value, str):
        return value if value else "__none__"
    return str(value)


def _allocate_stratified_counts(
    group_sizes: Dict[str, int],
    total: int,
) -> Dict[str, int]:
    """
    Allocate exactly `total` samples across groups approximately proportional to group size.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")

    n = sum(group_sizes.values())
    if total > n:
        raise ValueError(f"Cannot allocate {total} samples from only {n} records")

    raw = {g: total * size / n for g, size in group_sizes.items()}
    counts = {g: int(raw[g]) for g in group_sizes}

    # Ensure no group receives more samples than it contains.
    counts = {g: min(counts[g], group_sizes[g]) for g in counts}
    assigned = sum(counts.values())

    # Add remaining samples to groups with largest fractional parts and remaining capacity.
    remainders = sorted(
        group_sizes.keys(),
        key=lambda g: (raw[g] - int(raw[g]), group_sizes[g]),
        reverse=True,
    )
    while assigned < total:
        progressed = False
        for g in remainders:
            if counts[g] < group_sizes[g]:
                counts[g] += 1
                assigned += 1
                progressed = True
                if assigned == total:
                    break
        if not progressed:
            raise RuntimeError("Failed to allocate stratified counts")

    return counts


def sample_records(
    records: List[Dict[str, Any]],
    max_records: Optional[int],
    seed: int,
    stratify_key: Optional[str] = "category",
) -> List[Dict[str, Any]]:
    """
    Optionally sample a fixed-size subset before train/test splitting.

    This is mainly used to construct DROP-800 from the full DROP dataset.
    If max_records is None, negative, or >= len(records), all records are returned
    in a deterministic shuffled order.
    """
    rng = random.Random(seed)

    if max_records is None or max_records < 0 or max_records >= len(records):
        out = list(records)
        rng.shuffle(out)
        return out

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_group_key(record, stratify_key)].append(record)

    for group_records in groups.values():
        rng.shuffle(group_records)

    group_sizes = {g: len(v) for g, v in groups.items()}
    counts = _allocate_stratified_counts(group_sizes, max_records)

    selected: List[Dict[str, Any]] = []
    for g, count in counts.items():
        selected.extend(groups[g][:count])

    rng.shuffle(selected)
    return selected


def split_records_fixed_train_size(
    records: List[Dict[str, Any]],
    train_size: int,
    seed: int,
    stratify_key: Optional[str] = "category",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split records into exactly `train_size` uncertainty-training examples and the rest test.
    """
    if train_size <= 0:
        raise ValueError(f"train_size must be positive, got {train_size}")
    if train_size >= len(records):
        raise ValueError(
            f"train_size must be smaller than dataset size. "
            f"Got train_size={train_size}, num_records={len(records)}"
        )

    rng = random.Random(seed)

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[_group_key(record, stratify_key)].append(idx)

    for idxs in groups.values():
        rng.shuffle(idxs)

    group_sizes = {g: len(v) for g, v in groups.items()}
    counts = _allocate_stratified_counts(group_sizes, train_size)

    train_indices = set()
    for g, count in counts.items():
        train_indices.update(groups[g][:count])

    train_records = [record for idx, record in enumerate(records) if idx in train_indices]
    test_records = [record for idx, record in enumerate(records) if idx not in train_indices]

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def save_dataset_bundle(
    records: List[Dict[str, Any]],
    out_dir: Path,
    dataset_name: str,
    source_repo: str,
    source_config: Optional[str],
    source_splits: Sequence[str],
    train_size: int,
    seed: int,
    num_source_before_sampling: int,
    max_records: Optional[int] = None,
    stratify_key: Optional[str] = "category",
) -> None:
    ensure_dir(out_dir)

    train_records, test_records = split_records_fixed_train_size(
        records,
        train_size=train_size,
        seed=seed,
        stratify_key=stratify_key,
    )

    write_jsonl(records, out_dir / "all.jsonl")
    write_jsonl(train_records, out_dir / "uncertainty_train.jsonl")
    write_jsonl(test_records, out_dir / "test.jsonl")

    metadata = {
        "dataset": dataset_name,
        "source_repo": source_repo,
        "source_config": source_config,
        "source_splits": list(source_splits),
        "num_source_before_sampling": num_source_before_sampling,
        "max_records_after_sampling": max_records,
        "num_all": len(records),
        "num_uncertainty_train": len(train_records),
        "num_test": len(test_records),
        "train_size": train_size,
        "seed": seed,
        "stratify_key": stratify_key,
        "files": {
            "all": "all.jsonl",
            "uncertainty_train": "uncertainty_train.jsonl",
            "test": "test.jsonl",
        },
        "schema": {
            "id": "unique local id",
            "dataset": "dataset name",
            "source_split": "original Hugging Face split",
            "task_type": "task format, e.g. multiple_choice/free_form_math/reading_comprehension",
            "score_type": "recommended evaluation score type for a single model output",
            "question": "question text",
            "context": "passage/context, if any",
            "choices": "list of answer options for multiple-choice datasets, otherwise null",
            "answer": "canonical answer label, answer text, or answer list",
            "answer_index": "integer index for multiple-choice datasets, otherwise null",
            "answer_text": "canonical answer text when available",
            "category": "subject/domain/category/answer type when available",
            "metadata": "lightweight source metadata",
        },
    }

    write_json(metadata, out_dir / "metadata.json")


# -----------------------------
# GPQA normalization
# -----------------------------


def normalize_gpqa(
    ds: Dataset,
    source_splits: Sequence[str],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    GPQA expected columns in Idavidrein/gpqa:
      - Question
      - Correct Answer
      - Incorrect Answer 1
      - Incorrect Answer 2
      - Incorrect Answer 3

    We deterministically shuffle the answer options per example to avoid always
    placing the correct answer in the same position.
    """
    records = []

    for idx, row in enumerate(tqdm(ds, desc="Normalizing GPQA")):
        row = dict(row)

        question = clean_text(get_first_available(row, ["Question", "question"]))
        correct_answer = clean_text(
            get_first_available(row, ["Correct Answer", "correct_answer", "answer"])
        )

        incorrect_answers = []
        for key in [
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
            "incorrect_answer_1",
            "incorrect_answer_2",
            "incorrect_answer_3",
        ]:
            value = clean_text(row.get(key))
            if value:
                incorrect_answers.append(value)

        if question is None or correct_answer is None or len(incorrect_answers) < 3:
            raise ValueError(
                f"Unexpected GPQA row format at index {idx}. "
                f"Columns: {list(row.keys())}"
            )

        option_pairs = [(correct_answer, True)] + [(x, False) for x in incorrect_answers[:3]]

        local_id = f"gpqa_{idx:06d}"
        rng = random.Random(stable_int_hash(f"{local_id}_{seed}"))
        rng.shuffle(option_pairs)

        choices = [x[0] for x in option_pairs]
        answer_index = [x[1] for x in option_pairs].index(True)
        answer_letter = chr(ord("A") + answer_index)

        category = clean_text(
            get_first_available(
                row,
                [
                    "High-level domain",
                    "high_level_domain",
                    "domain",
                    "category",
                    "Subdomain",
                    "subdomain",
                ],
            )
        )

        record = {
            "id": local_id,
            "dataset": "gpqa",
            "source_split": clean_text(row.get("__source_split__")) or "combined",
            "task_type": "multiple_choice",
            "score_type": "binary_mcq_accuracy",
            "question": question,
            "context": None,
            "choices": choices,
            "answer": answer_letter,
            "answer_index": answer_index,
            "answer_text": correct_answer,
            "category": category,
            "metadata": {
                "source_splits_combined": list(source_splits),
            },
        }
        records.append(record)

    return records


# -----------------------------
# MATH-500 normalization
# -----------------------------


def extract_last_boxed_answer(solution: Optional[str]) -> Optional[str]:
    """
    Extract the content of the last \boxed{...} expression from a MATH-style solution.
    Falls back to None if no boxed expression is found.
    """
    if not solution:
        return None

    marker = r"\boxed"
    start = solution.rfind(marker)
    if start < 0:
        return None

    brace_start = solution.find("{", start)
    if brace_start < 0:
        return None

    depth = 0
    chars: List[str] = []
    for pos in range(brace_start, len(solution)):
        ch = solution[pos]
        if ch == "{":
            depth += 1
            if depth > 1:
                chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip() or None
            chars.append(ch)
        else:
            if depth >= 1:
                chars.append(ch)

    return None


def normalize_math_500(
    ds: Dataset,
    source_splits: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    MATH-500 expected columns in HuggingFaceH4/MATH-500 commonly include:
      - problem
      - solution
      - answer
      - subject
      - level

    The implementation is defensive and also accepts question/final_answer/category variants.
    """
    records = []

    for idx, row in enumerate(tqdm(ds, desc="Normalizing MATH-500")):
        row = dict(row)

        problem = clean_text(get_first_available(row, ["problem", "question", "prompt"]))
        solution = clean_text(get_first_available(row, ["solution", "cot", "rationale"]))
        answer = clean_text(
            get_first_available(row, ["answer", "final_answer", "ground_truth", "target"])
        )

        if answer is None:
            answer = extract_last_boxed_answer(solution)

        if problem is None or answer is None:
            raise ValueError(
                f"Unexpected MATH-500 row format at index {idx}. "
                f"Columns: {list(row.keys())}"
            )

        subject = clean_text(get_first_available(row, ["subject", "category", "type"]))
        level = clean_text(get_first_available(row, ["level", "difficulty"]))
        unique_id = clean_text(get_first_available(row, ["unique_id", "id"]))

        local_id = unique_id if unique_id else f"math500_{idx:06d}"

        record = {
            "id": local_id,
            "dataset": "math-500",
            "source_split": clean_text(row.get("__source_split__")) or "combined",
            "task_type": "free_form_math",
            "score_type": "binary_math_exact_match",
            "question": problem,
            "context": None,
            "choices": None,
            "answer": answer,
            "answer_index": None,
            "answer_text": answer,
            "category": subject,
            "metadata": {
                "solution": solution,
                "subject": subject,
                "level": level,
                "source_splits_combined": list(source_splits),
            },
        }
        records.append(record)

    return records


# -----------------------------
# DROP normalization
# -----------------------------


def parse_drop_answer_spans(answers_spans: Any) -> List[str]:
    """
    DROP answers_spans is usually:
      {"spans": [...], "types": [...]}

    Some HF conversions may only expose {"spans": [...]} or a plain list.
    """
    if answers_spans is None:
        return []

    if isinstance(answers_spans, dict):
        spans = answers_spans.get("spans", [])
        if spans is None:
            return []
        if isinstance(spans, list):
            return [clean_text(x) for x in spans if clean_text(x)]
        value = clean_text(spans)
        return [value] if value else []

    if isinstance(answers_spans, list):
        return [clean_text(x) for x in answers_spans if clean_text(x)]

    value = clean_text(answers_spans)
    return [value] if value else []


def infer_drop_answer_type(row: Dict[str, Any], answer_spans: Sequence[str]) -> str:
    """
    Infer a coarse DROP answer type for stratified sampling.
    This is intentionally lightweight and does not affect evaluation.
    """
    for key in ["answer_type", "type", "qtype", "question_type"]:
        value = clean_text(row.get(key))
        if value:
            return value

    if len(answer_spans) > 1:
        return "multi_span"
    if len(answer_spans) == 0:
        return "unknown"

    answer = answer_spans[0]
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", answer.replace(",", "")):
        return "number"
    if re.search(r"\b\d{4}\b|january|february|march|april|may|june|july|august|september|october|november|december", answer.lower()):
        return "date"
    return "span"


def normalize_drop(
    ds: Dataset,
    source_splits: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    DROP expected columns usually include:
      - section_id
      - query_id
      - passage
      - question
      - answers_spans
    """
    records = []

    for idx, row in enumerate(tqdm(ds, desc="Normalizing DROP")):
        row = dict(row)

        passage = clean_text(get_first_available(row, ["passage", "context"]))
        question = clean_text(get_first_available(row, ["question", "query"]))
        answer_spans = parse_drop_answer_spans(
            get_first_available(row, ["answers_spans", "answer_spans", "answers", "answer"])
        )

        if passage is None or question is None:
            raise ValueError(
                f"Unexpected DROP row format at index {idx}. Columns: {list(row.keys())}"
            )

        query_id = clean_text(row.get("query_id"))
        section_id = clean_text(row.get("section_id"))
        answer_type = infer_drop_answer_type(row, answer_spans)

        local_id = query_id if query_id else f"drop_{idx:06d}"

        record = {
            "id": local_id,
            "dataset": "drop-800",
            "source_split": clean_text(row.get("__source_split__")) or "combined",
            "task_type": "reading_comprehension",
            "score_type": "continuous_drop_f1",
            "question": question,
            "context": passage,
            "choices": None,
            "answer": answer_spans,
            "answer_index": None,
            "answer_text": answer_spans[0] if answer_spans else None,
            "category": answer_type,
            "metadata": {
                "section_id": section_id,
                "query_id": query_id,
                "answer_type": answer_type,
                "source_splits_combined": list(source_splits),
            },
        }
        records.append(record)

    return records


# -----------------------------
# Dataset preparation entry points
# -----------------------------


def prepare_gpqa(args: argparse.Namespace) -> None:
    repo = args.gpqa_repo
    config = args.gpqa_config

    ds_dict = load_hf_dataset(repo, config=config, token=args.hf_token)
    ds_dict = add_source_split_column(ds_dict)

    ds, source_splits = concatenate_selected_splits(ds_dict, split_names=args.gpqa_splits)
    records = normalize_gpqa(ds, source_splits=source_splits, seed=args.seed)
    num_source_before_sampling = len(records)
    records = sample_records(
        records,
        max_records=args.gpqa_max_records,
        seed=args.seed,
        stratify_key="category",
    )

    save_dataset_bundle(
        records=records,
        out_dir=Path(args.output_dir) / "gpqa",
        dataset_name="gpqa",
        source_repo=repo,
        source_config=config,
        source_splits=source_splits,
        train_size=args.train_size,
        seed=args.seed,
        num_source_before_sampling=num_source_before_sampling,
        max_records=args.gpqa_max_records,
        stratify_key="category",
    )


def prepare_math_500(args: argparse.Namespace) -> None:
    repo = args.math_500_repo
    config = args.math_500_config

    ds_dict = load_hf_dataset(repo, config=config, token=args.hf_token)
    ds_dict = add_source_split_column(ds_dict)

    ds, source_splits = concatenate_selected_splits(ds_dict, split_names=args.math_500_splits)
    records = normalize_math_500(ds, source_splits=source_splits)
    num_source_before_sampling = len(records)
    records = sample_records(
        records,
        max_records=args.math_500_max_records,
        seed=args.seed,
        stratify_key="category",
    )

    save_dataset_bundle(
        records=records,
        out_dir=Path(args.output_dir) / "math-500",
        dataset_name="math-500",
        source_repo=repo,
        source_config=config,
        source_splits=source_splits,
        train_size=args.train_size,
        seed=args.seed,
        num_source_before_sampling=num_source_before_sampling,
        max_records=args.math_500_max_records,
        stratify_key="category",
    )


def prepare_drop(args: argparse.Namespace) -> None:
    repo = args.drop_repo
    config = args.drop_config

    ds_dict = load_hf_dataset(repo, config=config, token=args.hf_token)
    ds_dict = add_source_split_column(ds_dict)

    ds, source_splits = concatenate_selected_splits(ds_dict, split_names=args.drop_splits)
    records = normalize_drop(ds, source_splits=source_splits)
    num_source_before_sampling = len(records)
    records = sample_records(
        records,
        max_records=args.drop_max_records,
        seed=args.seed,
        stratify_key="category",
    )

    save_dataset_bundle(
        records=records,
        out_dir=Path(args.output_dir) / "drop-800",
        dataset_name="drop-800",
        source_repo=repo,
        source_config=config,
        source_splits=source_splits,
        train_size=args.train_size,
        seed=args.seed,
        num_source_before_sampling=num_source_before_sampling,
        max_records=args.drop_max_records,
        stratify_key="category",
    )


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--train_size",
        type=int,
        default=200,
        help="Number of uncertainty/router-training examples per prepared dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Can also be set via HF_TOKEN env var.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gpqa", "math-500", "drop-800"],
        choices=["gpqa", "math-500", "drop-800"],
        help="Which datasets to prepare.",
    )

    # GPQA
    parser.add_argument("--gpqa_repo", type=str, default="Idavidrein/gpqa")
    parser.add_argument(
        "--gpqa_config",
        type=str,
        default="gpqa_main",
        help="GPQA config. Common choices: gpqa_main, gpqa_diamond, gpqa_extended.",
    )
    parser.add_argument(
        "--gpqa_splits",
        nargs="+",
        default=["all"],
        help="GPQA splits to combine before fixed-size train/test split.",
    )
    parser.add_argument(
        "--gpqa_max_records",
        type=int,
        default=-1,
        help="Optional cap for GPQA after normalization. -1 means use all available records.",
    )

    # MATH-500
    parser.add_argument(
        "--math_500_repo",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="Hugging Face repo for MATH-500.",
    )
    parser.add_argument(
        "--math_500_config",
        type=str,
        default=None,
        help="Optional MATH-500 config. Usually not needed.",
    )
    parser.add_argument(
        "--math_500_splits",
        nargs="+",
        default=["all"],
        help="MATH-500 splits to combine before fixed-size train/test split.",
    )
    parser.add_argument(
        "--math_500_max_records",
        type=int,
        default=-1,
        help="Optional cap for MATH-500 after normalization. -1 means use all available records.",
    )

    # DROP-800
    parser.add_argument("--drop_repo", type=str, default="ucinlp/drop")
    parser.add_argument(
        "--drop_config",
        type=str,
        default=None,
        help="Optional DROP config. Usually not needed.",
    )
    parser.add_argument(
        "--drop_splits",
        nargs="+",
        default=["train", "validation"],
        help="DROP splits to combine before sampling DROP-800.",
    )
    parser.add_argument(
        "--drop_max_records",
        type=int,
        default=800,
        help="Number of DROP records to sample before fixed-size train/test split.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(Path(args.output_dir))

    print(f"Output directory: {args.output_dir}")
    print(f"Train size per dataset: {args.train_size}")
    print(f"Seed: {args.seed}")
    print(f"Datasets: {args.datasets}")

    if "gpqa" in args.datasets:
        print("\nPreparing GPQA...")
        prepare_gpqa(args)

    if "math-500" in args.datasets:
        print("\nPreparing MATH-500...")
        prepare_math_500(args)

    if "drop-800" in args.datasets:
        print("\nPreparing DROP-800...")
        prepare_drop(args)

    print("\nDone.")


if __name__ == "__main__":
    main()

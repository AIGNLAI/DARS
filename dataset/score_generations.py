#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Score collected generation JSONL files for routing experiments.

This script is the reproducible replacement for notebook-only scoring. It keeps
the existing GPQA and DROP scoring semantics, but uses a more robust MATH answer
extractor and equivalence checker to reduce false negatives from common formats
such as ``The final answer is \boxed{25}``, LaTeX fractions, and simple symbolic
equivalent expressions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASETS = ("gpqa", "math-500", "drop-800")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def get_output_text(record: dict[str, Any]) -> str:
    for key in ("output_text", "response", "output", "text", "content", "message"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    choices = record.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
    return ""


def latex_command_spans(
    text: str,
    commands: Iterable[str],
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for command in commands:
        marker = "\\" + command
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            pos = idx + len(marker)
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text) or text[pos] != "{":
                start = idx + len(marker)
                continue

            depth = 0
            chars: list[str] = []
            end = None
            for cursor in range(pos, len(text)):
                ch = text[cursor]
                if ch == "{":
                    depth += 1
                    if depth > 1:
                        chars.append(ch)
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = cursor + 1
                        break
                    chars.append(ch)
                else:
                    if depth >= 1:
                        chars.append(ch)
            if end is not None:
                spans.append((idx, end, "".join(chars).strip()))
                start = end
            else:
                start = idx + len(marker)
    return sorted(spans, key=lambda item: item[0])


def last_latex_command_arg(text: str, commands: Iterable[str] = ("boxed", "fbox")) -> str:
    spans = latex_command_spans(str(text), commands)
    return spans[-1][2] if spans else ""


def unwrap_latex_commands(text: str, commands: Iterable[str] = ("boxed", "fbox")) -> str:
    out = str(text)
    while True:
        spans = latex_command_spans(out, commands)
        if not spans:
            return out
        start, end, content = spans[-1]
        out = out[:start] + content + out[end:]


def clean_answer_prefix(answer: str) -> str:
    out = str(answer).strip()
    out = out.strip(string.whitespace + "`*_")
    prefix_patterns = [
        r"^(?:therefore|thus|so)[,\s:]+",
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*",
        r"^(?:we\s+get|we\s+have|it\s+is)\s*",
    ]
    for _ in range(4):
        before = out
        out = out.strip(string.whitespace + "`*_")
        for pattern in prefix_patterns:
            out = re.sub(pattern, "", out, flags=re.IGNORECASE).strip()
        if out == before:
            break
    return out.strip()


def first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def extract_final_answer(text: str) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    matches = list(re.finditer(r"Final\s*Answer\s*:\s*", raw, flags=re.IGNORECASE))
    for match in reversed(matches):
        segment = raw[match.end() :].strip()
        first_line = first_nonempty_line(segment)
        if re.search(r"<[^>]+>", first_line):
            continue
        boxed = last_latex_command_arg(segment)
        if boxed:
            return clean_answer_prefix(boxed)
        answer = clean_answer_prefix(first_line)
        return answer.strip(" .。")

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.search(r"final\s+answer\s+is", line, flags=re.IGNORECASE):
            if "<answer>" in line.lower():
                continue
            boxed = last_latex_command_arg(line)
            if boxed:
                return clean_answer_prefix(boxed)
            return clean_answer_prefix(line).strip(" .。")

    boxed = last_latex_command_arg(raw)
    if boxed:
        return clean_answer_prefix(boxed)
    return clean_answer_prefix(lines[-1]).strip(" .。") if lines else ""


def extract_math_answer(text: str) -> str:
    raw = str(text or "")
    matches = list(re.finditer(r"Final\s*Answer\s*:\s*", raw, flags=re.IGNORECASE))
    for match in reversed(matches):
        segment = raw[match.end() :].strip()
        line = first_nonempty_line(segment)
        if re.search(r"<[^>]+>", line):
            continue
        if line:
            return clean_answer_prefix(unwrap_latex_commands(line)).strip(" .。")

    candidate = extract_final_answer(raw)
    if candidate:
        return clean_answer_prefix(unwrap_latex_commands(candidate)).strip(" .。")

    boxed = last_latex_command_arg(raw)
    if boxed:
        return clean_answer_prefix(boxed)
    return ""


def replace_simple_latex_frac(text: str, replacement: str = r"\1/\2") -> str:
    out = str(text)
    pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        new = pattern.sub(replacement, out)
        if new == out:
            return out
        out = new


def replace_simple_latex_sqrt(text: str) -> str:
    out = str(text)
    out = re.sub(r"\\sqrt\s*\[([^{}]+)\]\s*\{([^{}]+)\}", r"((\2)**(1/(\1)))", out)
    out = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", out)
    return out


def normalize_math_answer_text(answer: str) -> str:
    out = clean_answer_prefix(str(answer))
    out = unwrap_latex_commands(out)
    out = replace_simple_latex_frac(out)
    out = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", out)
    replacements = {
        "$": "",
        "\\(": "",
        "\\)": "",
        "\\[": "",
        "\\]": "",
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "\\;": "",
        "\\:": "",
        "\\cdot": "*",
        "\\times": "*",
        "\\pi": "pi",
        "π": "pi",
        "−": "-",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = re.sub(r"\s+", "", out)
    out = out.strip(".。,，;:")
    return out.lower()


def strip_balanced_outer(text: str) -> str:
    pairs = {"(": ")", "[": "]", "{": "}"}
    out = str(text)
    changed = True
    while changed and len(out) >= 2:
        changed = False
        left = out[0]
        right = pairs.get(left)
        if right and out[-1] == right:
            depth = 0
            balanced_outer = True
            for idx, ch in enumerate(out):
                if ch == left:
                    depth += 1
                elif ch == right:
                    depth -= 1
                    if depth == 0 and idx != len(out) - 1:
                        balanced_outer = False
                        break
            if balanced_outer:
                out = out[1:-1]
                changed = True
    return out


def normalized_math_variants(answer: str) -> set[str]:
    base = normalize_math_answer_text(answer)
    variants = {base, strip_balanced_outer(base)}
    variants.update({item.replace("\\", "") for item in list(variants)})
    variants.update({item.replace("*", "") for item in list(variants)})
    return {item for item in variants if item}


def math_to_sympy_expr(answer: str) -> str:
    out = clean_answer_prefix(str(answer))
    out = unwrap_latex_commands(out)
    out = replace_simple_latex_frac(out, replacement=r"(\1)/(\2)")
    out = replace_simple_latex_sqrt(out)
    out = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", out)
    replacements = {
        "$": "",
        "\\(": "",
        "\\)": "",
        "\\[": "",
        "\\]": "",
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "\\;": "",
        "\\:": "",
        "\\cdot": "*",
        "\\times": "*",
        "\\pi": "pi",
        "π": "pi",
        "−": "-",
        "^": "**",
        "{": "(",
        "}": ")",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = out.replace(",", "")
    out = re.sub(r"\s+", "", out)
    out = out.strip(".;:")
    return out


def sympy_equal(pred: str, gold: str) -> bool:
    try:
        import sympy as sp
    except ImportError:
        return False

    pred_expr = math_to_sympy_expr(pred)
    gold_expr = math_to_sympy_expr(gold)
    allowed = re.compile(r"^[0-9a-zA-Z_+\-*/().]+$")
    if not pred_expr or not gold_expr:
        return False
    if not allowed.fullmatch(pred_expr) or not allowed.fullmatch(gold_expr):
        return False

    locals_map = {"pi": sp.pi, "sqrt": sp.sqrt}
    try:
        p = sp.sympify(pred_expr, locals=locals_map)
        g = sp.sympify(gold_expr, locals=locals_map)
        diff = sp.simplify(p - g)
        if diff == 0:
            return True
        return bool(abs(float(sp.N(diff))) <= 1e-8 * max(1.0, abs(float(sp.N(g)))))
    except Exception:
        return False


def numeric_value(answer: str) -> float | None:
    expr = math_to_sympy_expr(answer)
    if not expr:
        return None
    try:
        return float(expr)
    except Exception:
        pass
    match = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", expr)
    if match:
        denominator = float(match.group(2))
        if abs(denominator) > 1e-12:
            return float(match.group(1)) / denominator
    return None


def math_equal(pred_answer: str, gold_answer: str) -> bool:
    pred_variants = normalized_math_variants(pred_answer)
    gold_variants = normalized_math_variants(gold_answer)
    if pred_variants.intersection(gold_variants):
        return True

    pred_value = numeric_value(pred_answer)
    gold_value = numeric_value(gold_answer)
    if pred_value is not None and gold_value is not None:
        return abs(pred_value - gold_value) <= 1e-8 * max(1.0, abs(gold_value))

    return sympy_equal(pred_answer, gold_answer)


def gold_letter(record: dict[str, Any]) -> str | None:
    answer = record.get("answer")
    if isinstance(answer, str) and re.fullmatch(r"[A-Z]", answer.strip(), flags=re.IGNORECASE):
        return answer.strip().upper()
    index = record.get("answer_index")
    if index is not None:
        try:
            return chr(ord("A") + int(index))
        except Exception:
            return None
    return None


def extract_mcq_letter(answer_text: str) -> str | None:
    if not answer_text:
        return None
    match = re.search(r"\b([A-D])\b", str(answer_text), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def normalize_drop_answer(answer: str) -> str:
    out = str(answer).lower()
    out = re.sub(r"\b(a|an|the)\b", " ", out)
    out = re.sub(r"[^a-z0-9\.\- ]", " ", out)
    return " ".join(out.split())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_drop_answer(pred).split()
    gold_tokens = normalize_drop_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def drop_gold_answers(record: dict[str, Any]) -> list[str]:
    answer = record.get("answer")
    if isinstance(answer, list):
        golds = [str(item) for item in answer if str(item).strip()]
    elif isinstance(answer, dict):
        spans = answer.get("spans", [])
        golds = [str(item) for item in spans if str(item).strip()]
    elif answer is not None:
        golds = [str(answer)]
    else:
        golds = []

    answer_text = record.get("answer_text")
    if answer_text:
        golds.append(str(answer_text))

    out = []
    seen = set()
    for gold in golds:
        key = normalize_drop_answer(gold)
        if key and key not in seen:
            seen.add(key)
            out.append(gold)
    return out


def score_record(record: dict[str, Any], split: str) -> dict[str, Any]:
    dataset = record.get("dataset")
    output_text = get_output_text(record)
    out = dict(record)

    if dataset == "gpqa":
        pred = extract_final_answer(output_text)
        pred_letter = extract_mcq_letter(pred)
        gold = gold_letter(record)
        score = float(pred_letter == gold) if gold else math.nan
        out.update(
            {
                "predicted_answer": pred_letter,
                "gold_answer": gold,
                "score": score,
                "score_type": "binary_mcq_accuracy",
            }
        )

    elif dataset == "math-500":
        pred = extract_math_answer(output_text)
        gold = str(record.get("answer_text") or record.get("answer") or "")
        out.update(
            {
                "predicted_answer": pred,
                "gold_answer": gold,
                "score": float(math_equal(pred, gold)),
                "score_type": "binary_math_exact_match",
            }
        )

    elif dataset == "drop-800":
        pred = extract_final_answer(output_text)
        golds = drop_gold_answers(record)
        f1s = [token_f1(pred, gold) for gold in golds]
        score = max(f1s) if f1s else math.nan
        em = (
            float(any(normalize_drop_answer(pred) == normalize_drop_answer(gold) for gold in golds))
            if golds
            else math.nan
        )
        out.update(
            {
                "predicted_answer": pred,
                "gold_answer": golds,
                "em": em,
                "score": score,
                "score_type": "continuous_drop_f1",
            }
        )

    else:
        pred = extract_final_answer(output_text)
        out.update(
            {
                "predicted_answer": pred,
                "gold_answer": record.get("answer"),
                "score": math.nan,
                "score_type": "unknown",
            }
        )

    usage = record.get("usage") or {}
    out["output_text_extracted"] = output_text
    out["parse_success"] = bool(out.get("predicted_answer"))
    out["cost"] = safe_float(record.get("cost"), default=safe_float(usage.get("cost")))
    out["split"] = split
    return out


def score_file(input_path: Path, output_path: Path, split: str) -> dict[str, Any]:
    records = read_jsonl(input_path)
    scored = [score_record(record, split=split) for record in records]
    write_jsonl(scored, output_path)

    old_scored = sum(1 for record in records if "score" in record)
    changed = 0
    if old_scored:
        for old, new in zip(records, scored):
            old_score = safe_float(old.get("score"))
            new_score = safe_float(new.get("score"))
            if math.isnan(old_score) and math.isnan(new_score):
                continue
            if old_score != new_score:
                changed += 1

    finite_scores = [safe_float(record.get("score")) for record in scored]
    finite_scores = [score for score in finite_scores if not math.isnan(score)]
    parse_rate = sum(bool(record.get("parse_success")) for record in scored) / max(1, len(scored))
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(scored),
        "mean_score": sum(finite_scores) / max(1, len(finite_scores)),
        "parse_rate": parse_rate,
        "old_scored_rows": old_scored,
        "score_changed_rows": changed,
    }


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument(
        "--input-suffix",
        default="_generations.jsonl",
        help="Input file suffix after split name, e.g. _generations.jsonl.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_scored_generations.jsonl",
        help="Output file suffix after split name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for dataset in args.datasets:
        for split in args.splits:
            input_path = args.data_dir / dataset / f"{split}{args.input_suffix}"
            if args.output_dir is None:
                output_path = args.data_dir / dataset / f"{split}{args.output_suffix}"
            else:
                output_path = args.output_dir / dataset / f"{split}{args.output_suffix}"
            summaries.append(score_file(input_path, output_path, split=split))

    for summary in summaries:
        print(
            f"{summary['output']}: rows={summary['rows']} "
            f"mean_score={summary['mean_score']:.6f} "
            f"parse_rate={summary['parse_rate']:.6f} "
            f"score_changed_rows={summary['score_changed_rows']}"
        )


if __name__ == "__main__":
    main()

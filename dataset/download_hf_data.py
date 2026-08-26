#!/usr/bin/env python3
"""Download and validate the scored DARS benchmark from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = ("gpqa", "math-500", "drop-800")
REQUIRED_FILES = ("train_scored_generations.jsonl", "test_scored_generations.jsonl")
DEFAULT_REPO_ID = "AIGNLAI/DARS"


def nonempty_jsonl(path: Path) -> bool:
    """Return whether a file contains at least one valid JSON object."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return isinstance(json.loads(line), dict)
    except (OSError, json.JSONDecodeError):
        return False
    return False


def validate_download(output_dir: Path) -> None:
    missing: list[Path] = []
    invalid: list[Path] = []
    for dataset in DATASETS:
        for filename in REQUIRED_FILES:
            path = output_dir / dataset / filename
            if not path.is_file():
                missing.append(path)
            elif not nonempty_jsonl(path):
                invalid.append(path)

    if missing or invalid:
        details = []
        if missing:
            details.append("missing: " + ", ".join(str(path) for path in missing))
        if invalid:
            details.append(
                "empty or invalid: " + ", ".join(str(path) for path in invalid)
            )
        raise RuntimeError("DARS data validation failed (" + "; ".join(details) + ")")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face branch, tag, or commit. Pin a commit for archival runs.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload files even if they are already present in the local cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'huggingface-hub'. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id}@{args.revision} to {args.output_dir.resolve()}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=[
            "README.md",
            "gpqa/*.json*",
            "math-500/*.json*",
            "drop-800/*.json*",
        ],
        force_download=args.force_download,
    )
    validate_download(args.output_dir)
    print("DARS data download complete; all scored train/test files are valid.")


if __name__ == "__main__":
    main()

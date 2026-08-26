"""Regression tests for shared routing utilities and paper defaults."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from methods.mlp import (
    DEFAULT_RISK_BETA,
    RouterConfig,
    get_risk_beta,
    load_scored_data,
    split_test_records,
)


class MethodUtilitiesTest(unittest.TestCase):
    def test_cost_is_normalized_within_each_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            costs = {
                "gpqa": {"train": 10.0, "test": 20.0},
                "math-500": {"train": 100.0, "test": 200.0},
            }
            for dataset, split_costs in costs.items():
                dataset_dir = root / dataset
                dataset_dir.mkdir(parents=True)
                for split, cost in split_costs.items():
                    record = {
                        "dataset": dataset,
                        "split": split,
                        "query_id": f"{dataset}-{split}",
                        "model": "model-a",
                        "score": 1.0,
                        "cost": cost,
                    }
                    path = dataset_dir / f"{split}_scored_generations.jsonl"
                    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            frame = load_scored_data(root, ["gpqa", "math-500"])
            normalized = {
                dataset: group.sort_values("cost")["routing_cost"].tolist()
                for dataset, group in frame.groupby("dataset")
            }
            self.assertEqual(normalized["gpqa"], [0.0, 1.0])
            self.assertEqual(normalized["math-500"], [0.0, 1.0])

    def test_paper_risk_default_is_used(self) -> None:
        self.assertEqual(DEFAULT_RISK_BETA, 0.2)
        self.assertEqual(get_risk_beta("gpqa", RouterConfig()), 0.2)
        self.assertEqual(get_risk_beta("math-500", RouterConfig()), 0.2)

    def test_missing_test_view_is_rejected(self) -> None:
        import pandas as pd

        records = pd.DataFrame(
            [{"observation_type": "prompt_variation", "prompt_variant_id": 0}]
        )
        with self.assertRaisesRegex(ValueError, "decoding_variation"):
            split_test_records(records)


if __name__ == "__main__":
    unittest.main()

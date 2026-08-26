#!/usr/bin/env python3
"""kNNRouter retrieval baseline for single-shot and DARS supervision.

The implementation follows the DARS paper appendix and predicts each model's
score, cost, and uncertainty from inverse-distance-weighted nearest training
queries.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from .mlp import (
        DATASET_DISPLAY,
        DATASET_RISK_BETA,
        DATASETS,
        DEFAULT_LOCAL_ENCODER,
        DEFAULT_RISK_BETA,
        QueryEncoder,
        build_quality_table,
        build_query_text_table,
        build_test_baselines,
        evaluate_predictions,
        load_scored_data,
        ordered_models,
        pivot_targets,
        sample_single_point_quality_table,
        select_dataset,
        split_test_records,
        to_dense_array,
    )
except ImportError:
    from mlp import (
        DATASET_DISPLAY,
        DATASET_RISK_BETA,
        DATASETS,
        DEFAULT_LOCAL_ENCODER,
        DEFAULT_RISK_BETA,
        QueryEncoder,
        build_quality_table,
        build_query_text_table,
        build_test_baselines,
        evaluate_predictions,
        load_scored_data,
        ordered_models,
        pivot_targets,
        sample_single_point_quality_table,
        select_dataset,
        split_test_records,
        to_dense_array,
    )


@dataclass(frozen=True)
class KNNConfig:
    cost_weight: float = 0.05
    risk_beta: float | None = None
    seed: int = 42
    single_point_runs: int = 100
    k: int = 16
    eps: float = 1e-8
    batch_size: int = 512
    max_features: int = 20000
    use_context: bool = True
    device: str = "auto"


class KNNRouter:
    def __init__(
        self,
        x_train: Any,
        target_score: np.ndarray,
        target_cost: np.ndarray,
        target_std: np.ndarray,
        config: KNNConfig,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.x_train = torch.from_numpy(
            np.asarray(to_dense_array(x_train), dtype=np.float32)
        ).to(self.device)
        self.target_score = torch.from_numpy(
            np.asarray(target_score, dtype=np.float32)
        ).to(self.device)
        self.target_cost = torch.from_numpy(
            np.asarray(target_cost, dtype=np.float32)
        ).to(self.device)
        self.target_std = torch.from_numpy(np.asarray(target_std, dtype=np.float32)).to(
            self.device
        )
        self.stats = {
            "train_examples": int(self.x_train.shape[0]),
            "k": int(min(config.k, self.x_train.shape[0])),
            "eps": float(config.eps),
            "device": str(self.device),
        }

    def predict(self, x_test: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = torch.from_numpy(np.asarray(to_dense_array(x_test), dtype=np.float32)).to(
            self.device
        )
        k = int(min(self.config.k, self.x_train.shape[0]))
        score_batches = []
        cost_batches = []
        std_batches = []
        with torch.no_grad():
            for start in range(0, x.shape[0], self.config.batch_size):
                xb = x[start : start + self.config.batch_size]
                distances = torch.cdist(xb, self.x_train)
                values, indices = torch.topk(
                    distances, k, dim=1, largest=False, sorted=True
                )
                zero_mask = values <= 1e-12
                any_zero = zero_mask.any(dim=1, keepdim=True)
                inv = 1.0 / (values + self.config.eps)
                weights = torch.where(any_zero.expand(-1, k), zero_mask.float(), inv)
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

                score_batches.append(
                    (weights.unsqueeze(-1) * self.target_score[indices]).sum(dim=1)
                )
                cost_batches.append(
                    (weights.unsqueeze(-1) * self.target_cost[indices]).sum(dim=1)
                )
                std_batches.append(
                    (weights.unsqueeze(-1) * self.target_std[indices]).sum(dim=1)
                )

        return (
            torch.cat(score_batches, dim=0).cpu().numpy(),
            torch.cat(cost_batches, dim=0).cpu().numpy(),
            torch.cat(std_batches, dim=0).cpu().numpy(),
        )


def resolve_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_risk_beta(dataset: str, config: KNNConfig) -> float:
    if config.risk_beta is not None:
        return float(config.risk_beta)
    return float(DATASET_RISK_BETA.get(dataset, DEFAULT_RISK_BETA))


def router_targets(
    qtarget: pd.DataFrame,
    query_meta: pd.DataFrame,
    models: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = pivot_targets(
        qtarget, query_meta, ["mean_score", "mean_cost", "score_std"], models
    )
    return (
        np.clip(targets["mean_score"], 0.0, 1.0).astype(np.float32),
        np.clip(targets["mean_cost"], 0.0, 1.0).astype(np.float32),
        np.clip(targets["score_std"], 0.0, 1.0).astype(np.float32),
    )


def predict_knn(
    router: KNNRouter,
    x_test: Any,
    models: Sequence[str],
    risk_beta: float,
    router_name: str,
    test_meta: pd.DataFrame,
    config: KNNConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_score, pred_cost, pred_std = router.predict(x_test)
    pred_score = np.clip(pred_score, 0.0, 1.0)
    pred_cost = np.clip(pred_cost, 0.0, 1.0)
    pred_std = np.clip(pred_std, 0.0, 1.0)
    pred_utility = pred_score - config.cost_weight * pred_cost
    pred_risk_utility = pred_utility - float(risk_beta) * pred_std
    selected = pred_risk_utility.argmax(axis=1)

    predictions = test_meta[["dataset", "dataset_display", "query_id"]].copy()
    predictions["router"] = router_name
    predictions["pred_model"] = [models[index] for index in selected]

    rows = []
    for row_index, row in test_meta.reset_index(drop=True).iterrows():
        for model_index, model_name in enumerate(models):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "dataset_display": row["dataset_display"],
                    "query_id": row["query_id"],
                    "router": router_name,
                    "model": model_name,
                    "pred_score": float(pred_score[row_index, model_index]),
                    "pred_cost": float(pred_cost[row_index, model_index]),
                    "pred_score_std": float(pred_std[row_index, model_index]),
                    "pred_utility": float(pred_utility[row_index, model_index]),
                    "pred_risk_utility": float(
                        pred_risk_utility[row_index, model_index]
                    ),
                    "risk_beta": float(risk_beta),
                    "selected": bool(model_index == selected[row_index]),
                }
            )
    return predictions, pd.DataFrame(rows)


def add_summary_metadata(
    summary: dict[str, Any],
    dataset: str,
    mode: str,
    run: int | None,
    risk_beta: float,
    stats: dict[str, Any],
    config: KNNConfig,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
        "mode": mode,
        "run": run,
        "cost_weight": config.cost_weight,
        "risk_beta": risk_beta,
        **stats,
        **summary,
    }


def summarize_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "dataset",
        "dataset_display",
        "mode",
        "cost_weight",
        "risk_beta",
        "k",
        "feature_backend",
    ]
    aggregate = (
        summary_df.groupby(grouping, dropna=False)
        .agg(
            n_routers=("mode", "size"),
            rewrite_score_mean=("rewrite_score", "mean"),
            rewrite_score_std=("rewrite_score", "std"),
            decoding_score_mean=("decoding_score", "mean"),
            decoding_score_std=("decoding_score", "std"),
            chosen_decoding_instability_mean=("decoding_score_std", "mean"),
            train_examples_mean=("train_examples", "mean"),
            n_eval_mean=("n_eval", "mean"),
            rewrite_best_model=("rewrite_best_model", "first"),
            rewrite_best_model_score=("rewrite_best_model_score", "first"),
            rewrite_oracle_score=("rewrite_oracle_score", "first"),
            decoding_best_model=("decoding_best_model", "first"),
            decoding_best_model_score=("decoding_best_model_score", "first"),
            decoding_oracle_score=("decoding_oracle_score", "first"),
        )
        .reset_index()
    )
    return aggregate.fillna({"rewrite_score_std": 0.0, "decoding_score_std": 0.0})


def run_experiment(
    data_dir: Path,
    datasets: Sequence[str],
    output_dir: Path,
    mode: str,
    feature_backend: str,
    local_encoder_path: Path,
    config: KNNConfig,
) -> pd.DataFrame:
    df = load_scored_data(data_dir, datasets)
    train_records = df[df["split"] == "train"].copy()
    test_records = df[df["split"] == "test"].copy()
    models = ordered_models(train_records)
    train_meta = build_query_text_table(train_records, use_context=config.use_context)
    test_meta = build_query_text_table(test_records, use_context=config.use_context)

    encoder = QueryEncoder(feature_backend, local_encoder_path, config.max_features)
    x_train, x_test = encoder.fit_transform(train_meta["text"], test_meta["text"])

    rewrite_records, decoding_records = split_test_records(test_records)
    train_distribution_qtable = build_quality_table(train_records, config.cost_weight)
    rewrite_qtable = build_quality_table(rewrite_records, config.cost_weight)
    decoding_qtable = build_quality_table(decoding_records, config.cost_weight)
    baselines = build_test_baselines(rewrite_qtable, decoding_qtable)

    summaries = []
    prediction_frames = []
    model_prediction_frames = []
    evaluation_frames = []
    for dataset in datasets:
        ds_train_meta, ds_x_train = select_dataset(train_meta, x_train, dataset)
        ds_test_meta, ds_x_test = select_dataset(test_meta, x_test, dataset)
        if ds_train_meta.empty or ds_test_meta.empty:
            continue

        if mode in {"distribution", "both"}:
            risk_beta = get_risk_beta(dataset, config)
            target = train_distribution_qtable[
                train_distribution_qtable["dataset"] == dataset
            ].copy()
            target_score, target_cost, target_std = router_targets(
                target, ds_train_meta, models
            )
            router = KNNRouter(
                ds_x_train, target_score, target_cost, target_std, config
            )
            predictions, model_predictions = predict_knn(
                router,
                ds_x_test,
                models,
                risk_beta=risk_beta,
                router_name="knn_router_distribution",
                test_meta=ds_test_meta,
                config=config,
            )
            summary, detail = evaluate_predictions(
                predictions, rewrite_qtable, decoding_qtable
            )
            summaries.append(
                add_summary_metadata(
                    summary,
                    dataset,
                    "distribution",
                    None,
                    risk_beta,
                    router.stats,
                    config,
                )
            )
            prediction_frames.append(predictions)
            model_prediction_frames.append(model_predictions)
            evaluation_frames.append(detail)

        if mode in {"single-point", "both"}:
            for run in range(config.single_point_runs):
                single_qtable = sample_single_point_quality_table(
                    train_records[train_records["dataset"] == dataset],
                    seed=config.seed + run,
                    cost_weight=config.cost_weight,
                )
                target_score, target_cost, target_std = router_targets(
                    single_qtable, ds_train_meta, models
                )
                router = KNNRouter(
                    ds_x_train, target_score, target_cost, target_std, config
                )
                predictions, model_predictions = predict_knn(
                    router,
                    ds_x_test,
                    models,
                    risk_beta=0.0,
                    router_name=f"knn_router_single_point_{run:03d}",
                    test_meta=ds_test_meta,
                    config=config,
                )
                summary, detail = evaluate_predictions(
                    predictions, rewrite_qtable, decoding_qtable
                )
                summaries.append(
                    add_summary_metadata(
                        summary, dataset, "single-point", run, 0.0, router.stats, config
                    )
                )
                prediction_frames.append(predictions)
                model_prediction_frames.append(model_predictions)
                evaluation_frames.append(detail)

    if not summaries:
        raise ValueError(
            "No kNN routers were evaluated. Check dataset names and input files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df["feature_backend"] = encoder.name
    summary_df = summary_df.merge(
        baselines, on=["dataset", "dataset_display"], how="left", validate="many_to_one"
    )
    summary_df.to_csv(output_dir / "knn_router_summary.csv", index=False)
    summarize_runs(summary_df).to_csv(
        output_dir / "knn_router_summary_by_mode.csv", index=False
    )
    baselines.to_csv(output_dir / "knn_router_test_baselines.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "knn_router_predictions.csv", index=False
    )
    pd.concat(model_prediction_frames, ignore_index=True).to_csv(
        output_dir / "knn_router_model_predictions.csv", index=False
    )
    pd.concat(evaluation_frames, ignore_index=True).to_csv(
        output_dir / "knn_router_evaluation_detail.csv", index=False
    )
    return summary_df


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "outputs" / "knn_router"
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--mode", choices=("single-point", "distribution", "both"), default="both"
    )
    parser.add_argument(
        "--feature-backend",
        choices=("auto", "tfidf", "sentence-transformer"),
        default="auto",
    )
    parser.add_argument(
        "--local-encoder-path", type=Path, default=DEFAULT_LOCAL_ENCODER
    )
    parser.add_argument("--single-point-runs", type=int, default=100)
    parser.add_argument("--cost-weight", type=float, default=0.05)
    parser.add_argument(
        "--risk-beta",
        type=float,
        default=None,
        help=f"Override risk penalty beta (paper default: {DEFAULT_RISK_BETA}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-context", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.single_point_runs <= 0 or args.k <= 0 or args.batch_size <= 0:
        raise ValueError("--single-point-runs, --k, and --batch-size must be positive.")
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive.")
    if args.cost_weight < 0 or (args.risk_beta is not None and args.risk_beta < 0):
        raise ValueError("--cost-weight and --risk-beta must be non-negative.")
    if args.eps <= 0:
        raise ValueError("--eps must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = KNNConfig(
        cost_weight=args.cost_weight,
        risk_beta=args.risk_beta,
        seed=args.seed,
        single_point_runs=args.single_point_runs,
        k=args.k,
        eps=args.eps,
        batch_size=args.batch_size,
        max_features=args.max_features,
        use_context=not args.no_context,
        device=args.device,
    )
    summary = run_experiment(
        args.data_dir,
        args.datasets,
        args.output_dir,
        args.mode,
        args.feature_backend,
        args.local_encoder_path,
        config,
    )
    report = summarize_runs(summary)
    print(
        report[
            [
                "dataset_display",
                "mode",
                "risk_beta",
                "k",
                "n_routers",
                "rewrite_score_mean",
                "decoding_score_mean",
                "n_eval_mean",
                "feature_backend",
            ]
        ].to_string(index=False)
    )
    baseline_columns = [
        "dataset_display",
        "rewrite_best_model",
        "rewrite_best_model_score",
        "rewrite_oracle_score",
        "decoding_best_model",
        "decoding_best_model_score",
        "decoding_oracle_score",
    ]
    print("\nTest score baselines:")
    print(
        summary[baseline_columns]
        .drop_duplicates(subset=["dataset_display"])
        .sort_values("dataset_display")
        .to_string(index=False)
    )
    print(f"\nSaved kNNRouter outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

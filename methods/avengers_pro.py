#!/usr/bin/env python3
"""AvengersPro cluster router for single-shot and DARS supervision.

The implementation follows the DARS paper appendix: it clusters training
queries, stores model-wise score, cost, and risk statistics for every cluster,
and routes test queries by aggregating their nearest clusters.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
class AvengersProConfig:
    cost_weight: float = 0.05
    risk_beta: float | None = None
    seed: int = 42
    single_point_runs: int = 100
    n_clusters: int = 32
    multi_cluster: int = 3
    max_iter: int = 300
    max_features: int = 20000
    use_context: bool = True


class AvengersProRouter:
    def __init__(
        self,
        x_train: Any,
        target_score: np.ndarray,
        target_cost: np.ndarray,
        target_std: np.ndarray,
        config: AvengersProConfig,
    ) -> None:
        self.config = config
        self.x_train = np.asarray(to_dense_array(x_train), dtype=np.float32)
        self.target_score = np.asarray(target_score, dtype=np.float32)
        self.target_cost = np.asarray(target_cost, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.n_clusters = int(min(config.n_clusters, self.x_train.shape[0]))
        self.multi_cluster = int(min(config.multi_cluster, self.n_clusters))
        self.kmeans: KMeans | None = None
        self.cluster_score: np.ndarray | None = None
        self.cluster_cost: np.ndarray | None = None
        self.cluster_std: np.ndarray | None = None
        self.stats: dict[str, Any] = {}

    def fit(self) -> None:
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.config.seed,
            n_init=10,
            max_iter=self.config.max_iter,
        )
        cluster_id = self.kmeans.fit_predict(self.x_train)
        n_models = self.target_score.shape[1]
        global_score = np.nanmean(self.target_score, axis=0)
        global_cost = np.nanmean(self.target_cost, axis=0)
        global_std = np.nanmean(self.target_std, axis=0)

        self.cluster_score = np.zeros((self.n_clusters, n_models), dtype=np.float32)
        self.cluster_cost = np.zeros((self.n_clusters, n_models), dtype=np.float32)
        self.cluster_std = np.zeros((self.n_clusters, n_models), dtype=np.float32)
        cluster_sizes = []
        for cluster in range(self.n_clusters):
            mask = cluster_id == cluster
            cluster_sizes.append(int(mask.sum()))
            if mask.any():
                self.cluster_score[cluster] = np.nanmean(
                    self.target_score[mask], axis=0
                )
                self.cluster_cost[cluster] = np.nanmean(self.target_cost[mask], axis=0)
                self.cluster_std[cluster] = np.nanmean(self.target_std[mask], axis=0)
            else:
                self.cluster_score[cluster] = global_score
                self.cluster_cost[cluster] = global_cost
                self.cluster_std[cluster] = global_std

        self.stats = {
            "train_examples": int(self.x_train.shape[0]),
            "n_clusters": int(self.n_clusters),
            "multi_cluster": int(self.multi_cluster),
            "empty_clusters": int(sum(size == 0 for size in cluster_sizes)),
            "min_cluster_size": int(min(cluster_sizes) if cluster_sizes else 0),
            "max_cluster_size": int(max(cluster_sizes) if cluster_sizes else 0),
        }

    def predict(self, x_test: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.kmeans is None or self.cluster_score is None:
            raise RuntimeError("Call fit() before predict().")
        features = np.asarray(to_dense_array(x_test), dtype=np.float32)
        centers = np.asarray(self.kmeans.cluster_centers_, dtype=np.float32)
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        top_clusters = np.argpartition(distances, self.multi_cluster - 1, axis=1)[
            :, : self.multi_cluster
        ]
        pred_score = self.cluster_score[top_clusters].mean(axis=1)
        pred_cost = self.cluster_cost[top_clusters].mean(axis=1)
        pred_std = self.cluster_std[top_clusters].mean(axis=1)
        return pred_score, pred_cost, pred_std


def get_risk_beta(dataset: str, config: AvengersProConfig) -> float:
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


def predict_avengers_pro(
    router: AvengersProRouter,
    x_test: Any,
    models: Sequence[str],
    risk_beta: float,
    router_name: str,
    test_meta: pd.DataFrame,
    config: AvengersProConfig,
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
    config: AvengersProConfig,
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
        "n_clusters",
        "multi_cluster",
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
            empty_clusters_mean=("empty_clusters", "mean"),
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
    config: AvengersProConfig,
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
            router = AvengersProRouter(
                ds_x_train, target_score, target_cost, target_std, config
            )
            router.fit()
            predictions, model_predictions = predict_avengers_pro(
                router,
                ds_x_test,
                models,
                risk_beta,
                "avengers_pro_distribution",
                ds_test_meta,
                config,
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
                run_config = replace(config, seed=config.seed + run)
                router = AvengersProRouter(
                    ds_x_train, target_score, target_cost, target_std, run_config
                )
                router.fit()
                predictions, model_predictions = predict_avengers_pro(
                    router,
                    ds_x_test,
                    models,
                    0.0,
                    f"avengers_pro_single_point_{run:03d}",
                    ds_test_meta,
                    config,
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
            "No AvengersPro routers were trained. Check dataset names and input files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df["feature_backend"] = encoder.name
    summary_df = summary_df.merge(
        baselines, on=["dataset", "dataset_display"], how="left", validate="many_to_one"
    )
    summary_df.to_csv(output_dir / "avengers_pro_summary.csv", index=False)
    summarize_runs(summary_df).to_csv(
        output_dir / "avengers_pro_summary_by_mode.csv", index=False
    )
    baselines.to_csv(output_dir / "avengers_pro_test_baselines.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "avengers_pro_predictions.csv", index=False
    )
    pd.concat(model_prediction_frames, ignore_index=True).to_csv(
        output_dir / "avengers_pro_model_predictions.csv", index=False
    )
    pd.concat(evaluation_frames, ignore_index=True).to_csv(
        output_dir / "avengers_pro_evaluation_detail.csv", index=False
    )
    return summary_df


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "outputs" / "avengers_pro"
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
    parser.add_argument("--n-clusters", type=int, default=32)
    parser.add_argument("--multi-cluster", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--no-context", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.single_point_runs <= 0:
        raise ValueError("--single-point-runs must be positive.")
    if args.n_clusters <= 0 or args.multi_cluster <= 0 or args.max_iter <= 0:
        raise ValueError(
            "--n-clusters, --multi-cluster, and --max-iter must be positive."
        )
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive.")
    if args.cost_weight < 0 or (args.risk_beta is not None and args.risk_beta < 0):
        raise ValueError("--cost-weight and --risk-beta must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = AvengersProConfig(
        cost_weight=args.cost_weight,
        risk_beta=args.risk_beta,
        seed=args.seed,
        single_point_runs=args.single_point_runs,
        n_clusters=args.n_clusters,
        multi_cluster=args.multi_cluster,
        max_iter=args.max_iter,
        max_features=args.max_features,
        use_context=not args.no_context,
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
                "n_clusters",
                "multi_cluster",
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
    print(f"\nSaved AvengersPro outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

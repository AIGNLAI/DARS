#!/usr/bin/env python3
"""Causal regret-minimization routers for distribution-aware LLM routing.

This script follows ``methods/mlp.py`` for local data loading, text encoding,
test splitting, and score reporting. The router itself follows the CausalRouter
reference: learn model choices from augmented ``(query embedding, lambda)``
examples by minimizing regret over observed model utility.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .mlp import (
        DATASET_DISPLAY,
        DATASET_RISK_BETA,
        DATASETS,
        DEFAULT_LOCAL_ENCODER,
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
class CausalConfig:
    cost_weight: float = 0.05
    risk_beta: float | None = None
    seed: int = 42
    single_point_runs: int = 100
    hidden_layers: tuple[int, ...] = (256, 256)
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    temperature: float = 1.0
    lambda_max: float = 0.10
    num_lambdas: int = 11
    max_features: int = 20000
    use_context: bool = True
    causal_loss: str = "softmax"
    device: str = "auto"


class RouterNet(nn.Module):
    """MLP logits network for a query embedding with one lambda feature."""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int], output_dim: int) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(size) for size in hidden_layers]]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def resolve_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_risk_beta(dataset: str, config: CausalConfig) -> float:
    if config.risk_beta is not None:
        return float(config.risk_beta)
    return float(DATASET_RISK_BETA.get(dataset, 0.10))


def seed_training(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_augmented_training(
    features: np.ndarray,
    performance: np.ndarray,
    cost: np.ndarray,
    lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand query rows over lambda values and return observed utilities."""
    features = np.asarray(features, dtype=np.float32)
    performance = np.asarray(performance, dtype=np.float32)
    cost = np.asarray(cost, dtype=np.float32)
    lambdas = np.asarray(lambdas, dtype=np.float32)

    n_queries, feature_dim = features.shape
    n_lambdas = int(lambdas.shape[0])
    utility = performance[None, :, :] - lambdas[:, None, None] * cost[None, :, :]
    labels = utility.argmax(axis=2).reshape(-1).astype(np.int64)

    features_rep = np.repeat(features[None, :, :], n_lambdas, axis=0)
    lambda_rep = np.repeat(lambdas[:, None], n_queries, axis=1)
    augmented = np.concatenate(
        [features_rep.reshape(-1, feature_dim), lambda_rep.reshape(-1, 1)],
        axis=1,
    )
    return augmented.astype(np.float32), labels, utility.reshape(-1, utility.shape[2]).astype(
        np.float32
    )


def causal_targets(
    qtarget: pd.DataFrame,
    query_meta: pd.DataFrame,
    models: Sequence[str],
    risk_beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build score, cost, and risk-adjusted performance matrices."""
    columns = ["mean_score", "mean_cost", "score_std"]
    targets = pivot_targets(qtarget, query_meta, columns, models)
    score = np.clip(targets["mean_score"], 0.0, 1.0)
    cost = np.clip(targets["mean_cost"], 0.0, 1.0)
    score_std = np.clip(targets["score_std"], 0.0, 1.0)
    risk_performance = score - float(risk_beta) * score_std
    return score, cost, risk_performance


def train_causal_router(
    features: Any,
    performance: np.ndarray,
    cost: np.ndarray,
    seed: int,
    config: CausalConfig,
) -> tuple[RouterNet, dict[str, float | int | str]]:
    """Fit an RM-Softmax or RM-Classification router."""
    seed_training(seed)
    device = resolve_device(config.device)
    features_dense = np.asarray(to_dense_array(features), dtype=np.float32)
    lambda_max = max(float(config.lambda_max), float(config.cost_weight))
    lambdas = np.linspace(0.0, lambda_max, int(config.num_lambdas), dtype=np.float32)
    x_aug, labels, utilities = build_augmented_training(features_dense, performance, cost, lambdas)

    x_tensor = torch.from_numpy(x_aug).to(device)
    label_tensor = torch.from_numpy(labels).long().to(device)
    utility_tensor = torch.from_numpy(utilities).to(device)
    model = RouterNet(
        input_dim=x_aug.shape[1],
        hidden_layers=config.hidden_layers,
        output_dim=performance.shape[1],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ce_loss = nn.CrossEntropyLoss()
    n_samples = int(x_tensor.shape[0])

    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, config.batch_size):
            index = order[start : start + config.batch_size]
            logits = model(x_tensor[index])
            if config.causal_loss == "classification":
                loss = ce_loss(logits, label_tensor[index])
            elif config.causal_loss == "softmax":
                scaled_logits = logits / float(config.temperature)
                probs = F.softmax(scaled_logits, dim=1)
                batch_utility = utility_tensor[index]
                best_utility = batch_utility.max(dim=1, keepdim=True).values
                loss = torch.sum(probs * (best_utility - batch_utility), dim=1).mean()
            else:
                raise ValueError(f"Unsupported causal loss: {config.causal_loss}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        chosen = logits.argmax(dim=1)
        train_accuracy = (chosen == label_tensor).float().mean().item()
        row_index = torch.arange(n_samples, device=device)
        chosen_utility = utility_tensor[row_index, chosen]
        best_utility = utility_tensor.max(dim=1).values
        avg_regret = (best_utility - chosen_utility).mean().item()

    stats: dict[str, float | int | str] = {
        "train_augmented_rows": n_samples,
        "train_target_accuracy": float(train_accuracy),
        "train_avg_regret": float(avg_regret),
        "lambda_max": float(lambda_max),
        "num_lambdas": int(config.num_lambdas),
        "causal_loss": config.causal_loss,
        "device": str(device),
    }
    return model, stats


def predict_choices(
    model: RouterNet,
    features: Any,
    models: Sequence[str],
    lambda_value: float,
    router_name: str,
    test_meta: pd.DataFrame,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route test queries and keep one probability row per query/model."""
    features_dense = np.asarray(to_dense_array(features), dtype=np.float32)
    lambda_column = np.full((features_dense.shape[0], 1), float(lambda_value), dtype=np.float32)
    router_features = np.concatenate([features_dense, lambda_column], axis=1)
    x_tensor = torch.from_numpy(router_features).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        probabilities = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    choices = probabilities.argmax(axis=1)

    predictions = test_meta[["dataset", "dataset_display", "query_id"]].copy()
    predictions["router"] = router_name
    predictions["pred_model"] = [models[index] for index in choices]

    probability_rows = []
    for row_index, row in test_meta.reset_index(drop=True).iterrows():
        for model_index, model_name in enumerate(models):
            probability_rows.append(
                {
                    "dataset": row["dataset"],
                    "dataset_display": row["dataset_display"],
                    "query_id": row["query_id"],
                    "router": router_name,
                    "model": model_name,
                    "choice_probability": float(probabilities[row_index, model_index]),
                    "selected": bool(model_index == choices[row_index]),
                    "lambda": float(lambda_value),
                }
            )
    return predictions, pd.DataFrame(probability_rows)


def add_summary_metadata(
    summary: dict[str, Any],
    dataset: str,
    mode: str,
    run: int | None,
    risk_beta: float,
    stats: dict[str, float | int | str],
    config: CausalConfig,
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
        "causal_loss",
        "lambda_max",
        "num_lambdas",
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
            train_avg_regret_mean=("train_avg_regret", "mean"),
            train_target_accuracy_mean=("train_target_accuracy", "mean"),
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
    config: CausalConfig,
) -> pd.DataFrame:
    df = load_scored_data(data_dir, datasets)
    train_records = df[df["split"] == "train"].copy()
    test_records = df[df["split"] == "test"].copy()
    models = ordered_models(train_records)
    train_meta = build_query_text_table(train_records, use_context=config.use_context)
    test_meta = build_query_text_table(test_records, use_context=config.use_context)

    encoder = QueryEncoder(
        backend=feature_backend,
        local_encoder_path=local_encoder_path,
        max_features=config.max_features,
    )
    x_train, x_test = encoder.fit_transform(train_meta["text"], test_meta["text"])

    rewrite_records, decoding_records = split_test_records(test_records)
    train_distribution_qtable = build_quality_table(train_records, config.cost_weight)
    rewrite_qtable = build_quality_table(rewrite_records, config.cost_weight)
    decoding_qtable = build_quality_table(decoding_records, config.cost_weight)
    baselines = build_test_baselines(rewrite_qtable, decoding_qtable)

    summaries = []
    prediction_frames = []
    probability_frames = []
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
            _, target_cost, target_performance = causal_targets(
                target, ds_train_meta, models, risk_beta
            )
            model, stats = train_causal_router(
                ds_x_train,
                target_performance,
                target_cost,
                seed=config.seed,
                config=config,
            )
            router_name = f"causal_{config.causal_loss}_distribution"
            predictions, probabilities = predict_choices(
                model,
                ds_x_test,
                models,
                lambda_value=config.cost_weight,
                router_name=router_name,
                test_meta=ds_test_meta,
                device=resolve_device(config.device),
            )
            summary, detail = evaluate_predictions(predictions, rewrite_qtable, decoding_qtable)
            summaries.append(
                add_summary_metadata(
                    summary,
                    dataset,
                    "distribution",
                    run=None,
                    risk_beta=risk_beta,
                    stats=stats,
                    config=config,
                )
            )
            prediction_frames.append(predictions)
            probability_frames.append(probabilities)
            evaluation_frames.append(detail)

        if mode in {"single-point", "both"}:
            for run in range(config.single_point_runs):
                single_qtable = sample_single_point_quality_table(
                    train_records[train_records["dataset"] == dataset],
                    seed=config.seed + run,
                    cost_weight=config.cost_weight,
                )
                _, target_cost, target_performance = causal_targets(
                    single_qtable, ds_train_meta, models, risk_beta=0.0
                )
                model, stats = train_causal_router(
                    ds_x_train,
                    target_performance,
                    target_cost,
                    seed=config.seed + run,
                    config=config,
                )
                router_name = f"causal_{config.causal_loss}_single_point_{run:03d}"
                predictions, probabilities = predict_choices(
                    model,
                    ds_x_test,
                    models,
                    lambda_value=config.cost_weight,
                    router_name=router_name,
                    test_meta=ds_test_meta,
                    device=resolve_device(config.device),
                )
                summary, detail = evaluate_predictions(predictions, rewrite_qtable, decoding_qtable)
                summaries.append(
                    add_summary_metadata(
                        summary,
                        dataset,
                        "single-point",
                        run=run,
                        risk_beta=0.0,
                        stats=stats,
                        config=config,
                    )
                )
                prediction_frames.append(predictions)
                probability_frames.append(probabilities)
                evaluation_frames.append(detail)

    if not summaries:
        raise ValueError("No causal routers were trained. Check dataset names and input files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df["feature_backend"] = encoder.name
    summary_df = summary_df.merge(
        baselines,
        on=["dataset", "dataset_display"],
        how="left",
        validate="many_to_one",
    )
    summary_df.to_csv(output_dir / "causalrouter_summary.csv", index=False)
    summarize_runs(summary_df).to_csv(
        output_dir / "causalrouter_summary_by_mode.csv", index=False
    )
    baselines.to_csv(output_dir / "causalrouter_test_baselines.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "causalrouter_predictions.csv", index=False
    )
    pd.concat(probability_frames, ignore_index=True).to_csv(
        output_dir / "causalrouter_model_probabilities.csv", index=False
    )
    pd.concat(evaluation_frames, ignore_index=True).to_csv(
        output_dir / "causalrouter_evaluation_detail.csv", index=False
    )
    return summary_df


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers or any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError("hidden layers must be positive comma-separated integers")
    return layers


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "analysis_outputs_causalrouter",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--mode",
        choices=("single-point", "distribution", "both"),
        default="both",
    )
    parser.add_argument(
        "--causal-loss",
        choices=("softmax", "classification"),
        default="softmax",
    )
    parser.add_argument(
        "--feature-backend",
        choices=("auto", "tfidf", "sentence-transformer"),
        default="auto",
    )
    parser.add_argument("--local-encoder-path", type=Path, default=DEFAULT_LOCAL_ENCODER)
    parser.add_argument("--single-point-runs", type=int, default=100)
    parser.add_argument("--cost-weight", type=float, default=0.05)
    parser.add_argument(
        "--risk-beta",
        type=float,
        default=None,
        help="Override distribution risk penalty beta. Default follows mlp.py per dataset.",
    )
    parser.add_argument("--lambda-max", type=float, default=0.10)
    parser.add_argument("--num-lambdas", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-layers", type=parse_hidden_layers, default=(256, 256))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-context", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.single_point_runs <= 0:
        raise ValueError("--single-point-runs must be positive.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.num_lambdas <= 0:
        raise ValueError("--num-lambdas must be positive.")
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.cost_weight < 0 or args.lambda_max < 0:
        raise ValueError("--cost-weight and --lambda-max must be non-negative.")
    if args.risk_beta is not None and args.risk_beta < 0:
        raise ValueError("--risk-beta must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = CausalConfig(
        cost_weight=args.cost_weight,
        risk_beta=args.risk_beta,
        seed=args.seed,
        single_point_runs=args.single_point_runs,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        lambda_max=args.lambda_max,
        num_lambdas=args.num_lambdas,
        max_features=args.max_features,
        use_context=not args.no_context,
        causal_loss=args.causal_loss,
        device=args.device,
    )
    summary = run_experiment(
        data_dir=args.data_dir,
        datasets=args.datasets,
        output_dir=args.output_dir,
        mode=args.mode,
        feature_backend=args.feature_backend,
        local_encoder_path=args.local_encoder_path,
        config=config,
    )
    report = summarize_runs(summary)
    report_columns = [
        "dataset_display",
        "mode",
        "causal_loss",
        "risk_beta",
        "n_routers",
        "rewrite_score_mean",
        "rewrite_score_std",
        "decoding_score_mean",
        "decoding_score_std",
        "train_avg_regret_mean",
        "n_eval_mean",
        "feature_backend",
    ]
    print(report[report_columns].to_string(index=False))

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
    print(f"\nSaved causal router outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MIRT-style interpretable router for distribution-aware LLM routing.

This adapts the LAMDA-ORBIT MIRT reference into the local scored-generation
format. It learns query ability vectors and per-model discrimination/difficulty
parameters, then routes by predicted score minus cost and risk penalties.
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

DATASET_RISK_BETA = {
    "gpqa": 0.2,
    "math-500": 0.05,
    "drop-800": 0.15,
}
@dataclass(frozen=True)
class MIRTConfig:
    cost_weight: float = 0.05
    risk_beta: float | None = None
    seed: int = 42
    single_point_runs: int = 100
    latent_dim: int = 16
    hidden_layers: tuple[int, ...] = (128,)
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    cost_loss_weight: float = 1.0
    std_loss_weight: float = 0.5
    max_features: int = 20000
    use_context: bool = True
    device: str = "auto"


class MIRTHead(nn.Module):
    """Multi-dimensional IRT head: sigmoid(theta(x) dot a_m - b_m)."""

    def __init__(
        self,
        input_dim: int,
        num_models: int,
        latent_dim: int,
        hidden_layers: Sequence[int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim, *[int(size) for size in hidden_layers]]
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.theta_net = nn.Sequential(*layers)
        self.discrimination = nn.Parameter(torch.randn(num_models, latent_dim) * 0.02)
        self.difficulty = nn.Parameter(torch.zeros(num_models))

    def logits(self, query_features: torch.Tensor) -> torch.Tensor:
        theta = self.theta_net(query_features)
        return theta @ self.discrimination.t() - self.difficulty.unsqueeze(0)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(query_features))


class MIRTRouter:
    def __init__(self, model: "MIRTNet", config: MIRTConfig, stats: dict[str, Any]) -> None:
        self.model = model
        self.config = config
        self.stats = stats


class MIRTNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_models: int,
        latent_dim: int,
        hidden_layers: Sequence[int],
    ) -> None:
        super().__init__()
        self.score_head = MIRTHead(input_dim, num_models, latent_dim, hidden_layers)
        self.cost_head = MIRTHead(input_dim, num_models, latent_dim, hidden_layers)
        self.std_head = MIRTHead(input_dim, num_models, latent_dim, hidden_layers)

    def forward(self, query_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score_logits = self.score_head.logits(query_features)
        cost = self.cost_head(query_features)
        std = self.std_head(query_features)
        return score_logits, cost, std


def resolve_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_training(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_risk_beta(dataset: str, config: MIRTConfig) -> float:
    if config.risk_beta is not None:
        return float(config.risk_beta)
    return float(DATASET_RISK_BETA.get(dataset, 0.10))


def router_targets(
    qtarget: pd.DataFrame,
    query_meta: pd.DataFrame,
    models: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = pivot_targets(qtarget, query_meta, ["mean_score", "mean_cost", "score_std"], models)
    return (
        np.clip(targets["mean_score"], 0.0, 1.0).astype(np.float32),
        np.clip(targets["mean_cost"], 0.0, 1.0).astype(np.float32),
        np.clip(targets["score_std"], 0.0, 1.0).astype(np.float32),
    )


def train_mirt_model(
    x_train: Any,
    target_score: np.ndarray,
    target_cost: np.ndarray,
    target_std: np.ndarray,
    seed: int,
    config: MIRTConfig,
) -> MIRTRouter:
    seed_training(seed)
    device = resolve_device(config.device)
    features = np.asarray(to_dense_array(x_train), dtype=np.float32)
    score = np.asarray(target_score, dtype=np.float32)
    cost = np.asarray(target_cost, dtype=np.float32)
    std = np.asarray(target_std, dtype=np.float32)
    if score.shape != cost.shape or score.shape != std.shape or score.shape[0] != features.shape[0]:
        raise ValueError("MIRT targets must align with training features.")

    model = MIRTNet(
        input_dim=features.shape[1],
        num_models=score.shape[1],
        latent_dim=config.latent_dim,
        hidden_layers=config.hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    x_tensor = torch.from_numpy(features).to(device)
    score_tensor = torch.from_numpy(score).to(device)
    cost_tensor = torch.from_numpy(cost).to(device)
    std_tensor = torch.from_numpy(std).to(device)

    n_examples = features.shape[0]
    batch_size = min(config.batch_size, n_examples)
    last_loss = 0.0
    last_score_loss = 0.0
    last_cost_loss = 0.0
    last_std_loss = 0.0
    model.train()
    for _ in range(config.epochs):
        permutation = torch.randperm(n_examples, device=device)
        epoch_loss = 0.0
        epoch_score_loss = 0.0
        epoch_cost_loss = 0.0
        epoch_std_loss = 0.0
        for start in range(0, n_examples, batch_size):
            index = permutation[start : start + batch_size]
            xb = x_tensor[index]
            y_score = score_tensor[index]
            y_cost = cost_tensor[index]
            y_std = std_tensor[index]

            optimizer.zero_grad()
            score_logits, pred_cost, pred_std = model(xb)
            score_loss = bce(score_logits, y_score)
            cost_loss = mse(pred_cost, y_cost)
            std_loss = mse(pred_std, y_std)
            loss = (
                score_loss
                + config.cost_loss_weight * cost_loss
                + config.std_loss_weight * std_loss
            )
            loss.backward()
            optimizer.step()

            batch_n = int(index.shape[0])
            epoch_loss += float(loss.detach().item()) * batch_n
            epoch_score_loss += float(score_loss.detach().item()) * batch_n
            epoch_cost_loss += float(cost_loss.detach().item()) * batch_n
            epoch_std_loss += float(std_loss.detach().item()) * batch_n

        last_loss = epoch_loss / max(1, n_examples)
        last_score_loss = epoch_score_loss / max(1, n_examples)
        last_cost_loss = epoch_cost_loss / max(1, n_examples)
        last_std_loss = epoch_std_loss / max(1, n_examples)

    model.eval()
    stats = {
        "train_examples": int(n_examples),
        "train_loss": float(last_loss),
        "train_score_loss": float(last_score_loss),
        "train_cost_loss": float(last_cost_loss),
        "train_std_loss": float(last_std_loss),
        "latent_dim": int(config.latent_dim),
        "device": str(device),
    }
    return MIRTRouter(model=model, config=config, stats=stats)


def predict_mirt(
    router: MIRTRouter,
    x_test: Any,
    models: Sequence[str],
    risk_beta: float,
    router_name: str,
    test_meta: pd.DataFrame,
    config: MIRTConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = resolve_device(config.device)
    features = np.asarray(to_dense_array(x_test), dtype=np.float32)
    x_tensor = torch.from_numpy(features).to(device)
    router.model.eval()
    with torch.no_grad():
        score_logits, pred_cost_tensor, pred_std_tensor = router.model(x_tensor)
        pred_score = torch.sigmoid(score_logits).cpu().numpy().astype(np.float32)
        pred_cost = pred_cost_tensor.cpu().numpy().astype(np.float32)
        pred_std = pred_std_tensor.cpu().numpy().astype(np.float32)

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
                    "pred_risk_utility": float(pred_risk_utility[row_index, model_index]),
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
    config: MIRTConfig,
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
        "latent_dim",
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
            train_loss_mean=("train_loss", "mean"),
            train_score_loss_mean=("train_score_loss", "mean"),
            train_cost_loss_mean=("train_cost_loss", "mean"),
            train_std_loss_mean=("train_std_loss", "mean"),
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
    config: MIRTConfig,
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
            target_score, target_cost, target_std = router_targets(target, ds_train_meta, models)
            router = train_mirt_model(
                ds_x_train, target_score, target_cost, target_std, config.seed, config
            )
            predictions, model_predictions = predict_mirt(
                router,
                ds_x_test,
                models,
                risk_beta,
                "mirt_distribution",
                ds_test_meta,
                config,
            )
            summary, detail = evaluate_predictions(predictions, rewrite_qtable, decoding_qtable)
            summaries.append(
                add_summary_metadata(
                    summary, dataset, "distribution", None, risk_beta, router.stats, config
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
                router = train_mirt_model(
                    ds_x_train, target_score, target_cost, target_std, config.seed + run, config
                )
                predictions, model_predictions = predict_mirt(
                    router,
                    ds_x_test,
                    models,
                    0.0,
                    f"mirt_single_point_{run:03d}",
                    ds_test_meta,
                    config,
                )
                summary, detail = evaluate_predictions(predictions, rewrite_qtable, decoding_qtable)
                summaries.append(
                    add_summary_metadata(
                        summary, dataset, "single-point", run, 0.0, router.stats, config
                    )
                )
                prediction_frames.append(predictions)
                model_prediction_frames.append(model_predictions)
                evaluation_frames.append(detail)

    if not summaries:
        raise ValueError("No MIRT routers were trained. Check dataset names and input files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df["feature_backend"] = encoder.name
    summary_df = summary_df.merge(
        baselines, on=["dataset", "dataset_display"], how="left", validate="many_to_one"
    )
    summary_df.to_csv(output_dir / "mirt_summary.csv", index=False)
    summarize_runs(summary_df).to_csv(output_dir / "mirt_summary_by_mode.csv", index=False)
    baselines.to_csv(output_dir / "mirt_test_baselines.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "mirt_predictions.csv", index=False
    )
    pd.concat(model_prediction_frames, ignore_index=True).to_csv(
        output_dir / "mirt_model_predictions.csv", index=False
    )
    pd.concat(evaluation_frames, ignore_index=True).to_csv(
        output_dir / "mirt_evaluation_detail.csv", index=False
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
    parser.add_argument("--output-dir", type=Path, default=project_dir / "analysis_outputs_mirt")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--mode", choices=("single-point", "distribution", "both"), default="both")
    parser.add_argument(
        "--feature-backend",
        choices=("auto", "tfidf", "sentence-transformer"),
        default="auto",
    )
    parser.add_argument("--local-encoder-path", type=Path, default=DEFAULT_LOCAL_ENCODER)
    parser.add_argument("--single-point-runs", type=int, default=100)
    parser.add_argument("--cost-weight", type=float, default=0.05)
    parser.add_argument("--risk-beta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-layers", type=parse_hidden_layers, default=(128,))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--cost-loss-weight", type=float, default=1.0)
    parser.add_argument("--std-loss-weight", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-context", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.single_point_runs <= 0:
        raise ValueError("--single-point-runs must be positive.")
    if args.latent_dim <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--latent-dim, --epochs, and --batch-size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive.")
    if args.cost_weight < 0 or (args.risk_beta is not None and args.risk_beta < 0):
        raise ValueError("--cost-weight and --risk-beta must be non-negative.")
    if args.weight_decay < 0 or args.cost_loss_weight < 0 or args.std_loss_weight < 0:
        raise ValueError("loss weights and --weight-decay must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = MIRTConfig(
        cost_weight=args.cost_weight,
        risk_beta=args.risk_beta,
        seed=args.seed,
        single_point_runs=args.single_point_runs,
        latent_dim=args.latent_dim,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        cost_loss_weight=args.cost_loss_weight,
        std_loss_weight=args.std_loss_weight,
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
                "latent_dim",
                "n_routers",
                "rewrite_score_mean",
                "decoding_score_mean",
                "train_loss_mean",
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
    print(f"\nSaved MIRT outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

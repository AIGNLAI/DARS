#!/usr/bin/env python3
"""EmbedLLM-style compact model-embedding router.

This script follows ``mlp.py`` for data loading, feature extraction, test
splitting, and reporting. It adapts the LAMDA-ORBIT EmbedLLM reference into the
local scored-generation format by learning a query projection and one compact
embedding per candidate model.
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


@dataclass(frozen=True)
class EmbedLLMConfig:
    cost_weight: float = 0.05
    risk_beta: float | None = None
    seed: int = 42
    single_point_runs: int = 100
    embed_dim: int = 256
    epochs: int = 20
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    alpha: float = 0.05
    cost_loss_weight: float = 1.0
    max_features: int = 20000
    use_context: bool = True
    device: str = "auto"


class EmbedLLMNet(nn.Module):
    """Predict query-model performance with compact learned model embeddings."""

    def __init__(self, input_dim: int, num_models: int, embed_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(input_dim, embed_dim)
        self.model_embed = nn.Embedding(num_models, embed_dim)
        self.model_bias = nn.Parameter(torch.zeros(num_models))
        self.cost_head = nn.Linear(embed_dim, 1)
        nn.init.normal_(self.model_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        query_features: torch.Tensor,
        add_noise: bool = False,
        alpha: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_hidden = self.query_proj(query_features)
        model_ids = torch.arange(self.model_embed.num_embeddings, device=query_features.device)
        model_hidden = self.model_embed(model_ids)

        if add_noise and alpha > 0:
            query_hidden = query_hidden + alpha * torch.randn_like(query_hidden)
            model_hidden = model_hidden + alpha * torch.randn_like(model_hidden)

        logits = query_hidden @ model_hidden.t() + self.model_bias.unsqueeze(0)
        model_cost = self.cost_head(model_hidden).squeeze(1)
        pred_cost = model_cost.unsqueeze(0).expand(query_features.shape[0], -1)
        return logits, pred_cost


class EmbedLLMRouter:
    def __init__(self, model: EmbedLLMNet, config: EmbedLLMConfig, stats: dict[str, Any]) -> None:
        self.model = model
        self.config = config
        self.stats = stats


def resolve_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_training(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_risk_beta(dataset: str, config: EmbedLLMConfig) -> float:
    if config.risk_beta is not None:
        return float(config.risk_beta)
    return float(DATASET_RISK_BETA.get(dataset, 0.10))


def embed_targets(
    qtarget: pd.DataFrame,
    query_meta: pd.DataFrame,
    models: Sequence[str],
    risk_beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = pivot_targets(qtarget, query_meta, ["mean_score", "mean_cost", "score_std"], models)
    score = np.clip(targets["mean_score"], 0.0, 1.0).astype(np.float32)
    cost = np.clip(targets["mean_cost"], 0.0, 1.0).astype(np.float32)
    score_std = np.clip(targets["score_std"], 0.0, 1.0).astype(np.float32)
    risk_score = np.clip(score - float(risk_beta) * score_std, 0.0, 1.0).astype(np.float32)
    return score, cost, risk_score


def train_embedllm_model(
    x_train: Any,
    target_score: np.ndarray,
    target_cost: np.ndarray,
    seed: int,
    config: EmbedLLMConfig,
) -> EmbedLLMRouter:
    seed_training(seed)
    device = resolve_device(config.device)
    features = np.asarray(to_dense_array(x_train), dtype=np.float32)
    score = np.asarray(target_score, dtype=np.float32)
    cost = np.asarray(target_cost, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("Training features must be a 2D matrix.")
    if score.shape != cost.shape or score.shape[0] != features.shape[0]:
        raise ValueError("Target score/cost matrices must align with training features.")

    model = EmbedLLMNet(
        input_dim=features.shape[1],
        num_models=score.shape[1],
        embed_dim=config.embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    x_tensor = torch.from_numpy(features).to(device)
    score_tensor = torch.from_numpy(np.clip(score, 0.0, 1.0)).to(device)
    cost_tensor = torch.from_numpy(np.clip(cost, 0.0, 1.0)).to(device)

    n_examples = features.shape[0]
    batch_size = min(config.batch_size, n_examples)
    last_loss = 0.0
    last_perf_loss = 0.0
    last_cost_loss = 0.0
    model.train()
    for _ in range(config.epochs):
        permutation = torch.randperm(n_examples, device=device)
        epoch_loss = 0.0
        epoch_perf_loss = 0.0
        epoch_cost_loss = 0.0
        for start in range(0, n_examples, batch_size):
            index = permutation[start : start + batch_size]
            xb = x_tensor[index]
            y_score = score_tensor[index]
            y_cost = cost_tensor[index]

            optimizer.zero_grad()
            logits, pred_cost = model(xb, add_noise=True, alpha=config.alpha)
            perf_loss = bce(logits, y_score)
            cost_loss = mse(pred_cost, y_cost)
            loss = perf_loss + config.cost_loss_weight * cost_loss
            loss.backward()
            optimizer.step()

            batch_n = int(index.shape[0])
            epoch_loss += float(loss.detach().item()) * batch_n
            epoch_perf_loss += float(perf_loss.detach().item()) * batch_n
            epoch_cost_loss += float(cost_loss.detach().item()) * batch_n

        last_loss = epoch_loss / max(1, n_examples)
        last_perf_loss = epoch_perf_loss / max(1, n_examples)
        last_cost_loss = epoch_cost_loss / max(1, n_examples)

    model.eval()
    stats = {
        "train_examples": int(n_examples),
        "train_loss": float(last_loss),
        "train_perf_loss": float(last_perf_loss),
        "train_cost_loss": float(last_cost_loss),
        "embed_dim": int(config.embed_dim),
        "alpha": float(config.alpha),
        "cost_loss_weight": float(config.cost_loss_weight),
        "device": str(device),
    }
    return EmbedLLMRouter(model=model, config=config, stats=stats)


def predict_embedllm(
    router: EmbedLLMRouter,
    x_test: Any,
    models: Sequence[str],
    risk_beta: float,
    router_name: str,
    test_meta: pd.DataFrame,
    config: EmbedLLMConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = resolve_device(config.device)
    features = np.asarray(to_dense_array(x_test), dtype=np.float32)
    x_tensor = torch.from_numpy(features).to(device)
    router.model.eval()
    with torch.no_grad():
        logits, pred_cost_tensor = router.model(x_tensor, add_noise=False)
        pred_score = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        pred_cost = pred_cost_tensor.cpu().numpy().astype(np.float32)

    pred_score = np.clip(pred_score, 0.0, 1.0)
    pred_cost = np.clip(pred_cost, 0.0, 1.0)
    pred_std = np.zeros_like(pred_score, dtype=np.float32)
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
    config: EmbedLLMConfig,
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
        "embed_dim",
        "alpha",
        "cost_loss_weight",
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
            train_perf_loss_mean=("train_perf_loss", "mean"),
            train_cost_loss_mean=("train_cost_loss", "mean"),
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
    config: EmbedLLMConfig,
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
            _, target_cost, target_score = embed_targets(target, ds_train_meta, models, risk_beta)
            router = train_embedllm_model(
                ds_x_train,
                target_score,
                target_cost,
                seed=config.seed,
                config=config,
            )
            router_name = "embedllm_distribution"
            predictions, model_predictions = predict_embedllm(
                router,
                ds_x_test,
                models,
                risk_beta=risk_beta,
                router_name=router_name,
                test_meta=ds_test_meta,
                config=config,
            )
            summary, detail = evaluate_predictions(predictions, rewrite_qtable, decoding_qtable)
            summaries.append(
                add_summary_metadata(
                    summary,
                    dataset,
                    "distribution",
                    run=None,
                    risk_beta=risk_beta,
                    stats=router.stats,
                    config=config,
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
                _, target_cost, target_score = embed_targets(
                    single_qtable,
                    ds_train_meta,
                    models,
                    risk_beta=0.0,
                )
                router = train_embedllm_model(
                    ds_x_train,
                    target_score,
                    target_cost,
                    seed=config.seed + run,
                    config=config,
                )
                router_name = f"embedllm_single_point_{run:03d}"
                predictions, model_predictions = predict_embedllm(
                    router,
                    ds_x_test,
                    models,
                    risk_beta=0.0,
                    router_name=router_name,
                    test_meta=ds_test_meta,
                    config=config,
                )
                summary, detail = evaluate_predictions(predictions, rewrite_qtable, decoding_qtable)
                summaries.append(
                    add_summary_metadata(
                        summary,
                        dataset,
                        "single-point",
                        run=run,
                        risk_beta=0.0,
                        stats=router.stats,
                        config=config,
                    )
                )
                prediction_frames.append(predictions)
                model_prediction_frames.append(model_predictions)
                evaluation_frames.append(detail)

    if not summaries:
        raise ValueError("No EmbedLLM routers were trained. Check dataset names and input files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df["feature_backend"] = encoder.name
    summary_df = summary_df.merge(
        baselines,
        on=["dataset", "dataset_display"],
        how="left",
        validate="many_to_one",
    )
    summary_df.to_csv(output_dir / "embedllm_summary.csv", index=False)
    summarize_runs(summary_df).to_csv(output_dir / "embedllm_summary_by_mode.csv", index=False)
    baselines.to_csv(output_dir / "embedllm_test_baselines.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "embedllm_predictions.csv", index=False
    )
    pd.concat(model_prediction_frames, ignore_index=True).to_csv(
        output_dir / "embedllm_model_predictions.csv", index=False
    )
    pd.concat(evaluation_frames, ignore_index=True).to_csv(
        output_dir / "embedllm_evaluation_detail.csv", index=False
    )
    return summary_df


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "analysis_outputs_embedllm",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--mode",
        choices=("single-point", "distribution", "both"),
        default="both",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cost-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-context", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.single_point_runs <= 0:
        raise ValueError("--single-point-runs must be positive.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.embed_dim <= 0:
        raise ValueError("--embed-dim must be positive.")
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive.")
    if args.cost_weight < 0 or (args.risk_beta is not None and args.risk_beta < 0):
        raise ValueError("--cost-weight and --risk-beta must be non-negative.")
    if args.weight_decay < 0 or args.alpha < 0 or args.cost_loss_weight < 0:
        raise ValueError("--weight-decay, --alpha, and --cost-loss-weight must be non-negative.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = EmbedLLMConfig(
        cost_weight=args.cost_weight,
        risk_beta=args.risk_beta,
        seed=args.seed,
        single_point_runs=args.single_point_runs,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        cost_loss_weight=args.cost_loss_weight,
        max_features=args.max_features,
        use_context=not args.no_context,
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
        "risk_beta",
        "embed_dim",
        "n_routers",
        "rewrite_score_mean",
        "rewrite_score_std",
        "decoding_score_mean",
        "decoding_score_std",
        "train_loss_mean",
        "train_examples_mean",
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
    print(f"\nSaved EmbedLLM outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

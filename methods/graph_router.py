#!/usr/bin/env python3
"""GraphRouter edge predictor for single-shot and DARS supervision.

The implementation follows the DARS paper appendix: each query-model pair is
an edge, and the router predicts edge-level score, cost, and uncertainty from
query features and model descriptions.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

try:
    from .mlp import (
        DATASET_DISPLAY,
        DATASET_RISK_BETA,
        DATASETS,
        DEFAULT_LOCAL_ENCODER,
        DEFAULT_RISK_BETA,
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


MODEL_DESCRIPTIONS = {
    "google/gemma-3-12b-it": "Google Gemma 3 12B instruction tuned open model.",
    "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2 24B instruction tuned model.",
    "qwen/qwen3-32b": "Qwen3 32B general reasoning and instruction model.",
    "meta-llama/llama-3.3-70b-instruct": "Meta Llama 3.3 70B instruction tuned model.",
    "google/gemini-2.5-flash-lite": "Google Gemini 2.5 Flash Lite efficient model.",
    "deepseek/deepseek-chat-v3.1": "DeepSeek Chat V3.1 general chat and reasoning model.",
}


@dataclass(frozen=True)
class GraphConfig:
    cost_weight: float = 0.05
    risk_beta: float | None = None
    seed: int = 42
    single_point_runs: int = 100
    hidden_layers: tuple[int, ...] = (256, 128)
    model_feature_dim: int = 128
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_features: int = 20000
    use_context: bool = True
    description_source: str = "generic"
    device: str = "auto"


class GraphTextEncoder:
    """Encode query and model-description text into the same feature space."""

    def __init__(
        self,
        backend: str,
        local_encoder_path: Path,
        max_features: int,
    ) -> None:
        self.backend = backend
        self.local_encoder_path = local_encoder_path
        self.max_features = max_features
        self.name = ""
        self._encoder: Any = None
        self._vectorizer: TfidfVectorizer | None = None

    def fit_transform(
        self,
        train_texts: Sequence[str],
        test_texts: Sequence[str],
        model_texts: Sequence[str],
    ) -> tuple[Any, Any, np.ndarray]:
        if (
            self.backend in {"auto", "sentence-transformer"}
            and self.local_encoder_path.exists()
        ):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                if self.backend == "sentence-transformer":
                    raise
            else:
                self._encoder = SentenceTransformer(str(self.local_encoder_path))
                self.name = str(self.local_encoder_path)
                return (
                    self._encode_dense(train_texts),
                    self._encode_dense(test_texts),
                    self._encode_dense(model_texts),
                )

        if self.backend == "sentence-transformer":
            raise FileNotFoundError(
                f"Local sentence encoder not found: {self.local_encoder_path}"
            )

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=1,
            ngram_range=(1, 2),
        )
        self.name = "tfidf"
        train_features = self._vectorizer.fit_transform(
            list(train_texts) + list(model_texts)
        )
        n_train = len(train_texts)
        x_train = train_features[:n_train]
        model_features = train_features[n_train:].toarray().astype(np.float32)
        return x_train, self._vectorizer.transform(test_texts), model_features

    def _encode_dense(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._encoder.encode(
                list(texts),
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )


class EdgeGraphNet(nn.Module):
    """Predict query-model edge targets from query and LLM node features."""

    def __init__(
        self,
        query_dim: int,
        model_dim: int,
        model_feature_dim: int,
        hidden_layers: Sequence[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(query_dim, model_feature_dim)
        self.model_proj = nn.Linear(model_dim, model_feature_dim)
        edge_input_dim = model_feature_dim * 4
        layers: list[nn.Module] = []
        dims = [edge_input_dim, *[int(size) for size in hidden_layers]]
        for in_dim, out_dim in pairwise(dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.edge_mlp = nn.Sequential(*layers)

    def forward(
        self, query_features: torch.Tensor, model_features: torch.Tensor
    ) -> torch.Tensor:
        query_hidden = torch.relu(self.query_proj(query_features))
        model_hidden = torch.relu(self.model_proj(model_features))
        n_queries = int(query_hidden.shape[0])
        n_models = int(model_hidden.shape[0])

        query_edge = query_hidden[:, None, :].expand(n_queries, n_models, -1)
        model_edge = model_hidden[None, :, :].expand(n_queries, n_models, -1)
        edge_features = torch.cat(
            [
                query_edge,
                model_edge,
                query_edge * model_edge,
                torch.abs(query_edge - model_edge),
            ],
            dim=2,
        )
        return self.edge_mlp(edge_features)


def resolve_device(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_training(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_risk_beta(dataset: str, config: GraphConfig) -> float:
    if config.risk_beta is not None:
        return float(config.risk_beta)
    return float(DATASET_RISK_BETA.get(dataset, DEFAULT_RISK_BETA))


def model_description_texts(models: Sequence[str], source: str) -> list[str]:
    if source == "name":
        return [model for model in models]
    if source == "generic":
        return [
            f"{model}. {MODEL_DESCRIPTIONS.get(model, 'Large language model.')}"
            for model in models
        ]
    raise ValueError(f"Unsupported description source: {source}")


def graph_targets(
    qtarget: pd.DataFrame,
    query_meta: pd.DataFrame,
    models: Sequence[str],
) -> np.ndarray:
    targets = pivot_targets(
        qtarget, query_meta, ["mean_score", "mean_cost", "score_std"], models
    )
    stacked = np.stack(
        [
            np.clip(targets["mean_score"], 0.0, 1.0),
            np.clip(targets["mean_cost"], 0.0, 1.0),
            np.clip(targets["score_std"], 0.0, 1.0),
        ],
        axis=2,
    )
    return stacked.astype(np.float32)


def train_graph_model(
    x_train: Any,
    model_features: np.ndarray,
    targets: np.ndarray,
    seed: int,
    config: GraphConfig,
) -> tuple[EdgeGraphNet, dict[str, float | int | str]]:
    seed_training(seed)
    device = resolve_device(config.device)
    query_features = np.asarray(to_dense_array(x_train), dtype=np.float32)
    model_features = np.asarray(model_features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    query_tensor = torch.from_numpy(query_features).to(device)
    model_tensor = torch.from_numpy(model_features).to(device)
    target_tensor = torch.from_numpy(targets).to(device)

    model = EdgeGraphNet(
        query_dim=query_features.shape[1],
        model_dim=model_features.shape[1],
        model_feature_dim=config.model_feature_dim,
        hidden_layers=config.hidden_layers,
        output_dim=3,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    n_queries = int(query_tensor.shape[0])

    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(n_queries, device=device)
        for start in range(0, n_queries, config.batch_size):
            index = order[start : start + config.batch_size]
            prediction = model(query_tensor[index], model_tensor)
            loss = criterion(prediction, target_tensor[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction = model(query_tensor, model_tensor)
        train_mse = criterion(prediction, target_tensor).item()

    stats: dict[str, float | int | str] = {
        "train_mse": float(train_mse),
        "train_edges": int(targets.shape[0] * targets.shape[1]),
        "device": str(device),
    }
    return model, stats


def predict_graph_router(
    model: EdgeGraphNet,
    x_test: Any,
    model_features: np.ndarray,
    models: Sequence[str],
    risk_beta: float,
    router_name: str,
    test_meta: pd.DataFrame,
    config: GraphConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = resolve_device(config.device)
    query_features = np.asarray(to_dense_array(x_test), dtype=np.float32)
    query_tensor = torch.from_numpy(query_features).to(device)
    model_tensor = torch.from_numpy(np.asarray(model_features, dtype=np.float32)).to(
        device
    )

    model.eval()
    with torch.no_grad():
        raw_prediction = model(query_tensor, model_tensor).cpu().numpy()
    pred_score = np.clip(raw_prediction[:, :, 0], 0.0, 1.0)
    pred_cost = np.clip(raw_prediction[:, :, 1], 0.0, 1.0)
    pred_std = np.clip(raw_prediction[:, :, 2], 0.0, 1.0)
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
    stats: dict[str, float | int | str],
    config: GraphConfig,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
        "mode": mode,
        "run": run,
        "cost_weight": config.cost_weight,
        "risk_beta": risk_beta,
        "description_source": config.description_source,
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
        "description_source",
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
            train_mse_mean=("train_mse", "mean"),
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
    config: GraphConfig,
) -> pd.DataFrame:
    df = load_scored_data(data_dir, datasets)
    train_records = df[df["split"] == "train"].copy()
    test_records = df[df["split"] == "test"].copy()
    models = ordered_models(train_records)
    train_meta = build_query_text_table(train_records, use_context=config.use_context)
    test_meta = build_query_text_table(test_records, use_context=config.use_context)
    description_texts = model_description_texts(models, config.description_source)

    encoder = GraphTextEncoder(
        backend=feature_backend,
        local_encoder_path=local_encoder_path,
        max_features=config.max_features,
    )
    x_train, x_test, model_features = encoder.fit_transform(
        train_meta["text"], test_meta["text"], description_texts
    )

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
            targets = graph_targets(target, ds_train_meta, models)
            graph_model, stats = train_graph_model(
                ds_x_train,
                model_features,
                targets,
                seed=config.seed,
                config=config,
            )
            router_name = "graph_router_distribution"
            predictions, model_predictions = predict_graph_router(
                graph_model,
                ds_x_test,
                model_features,
                models,
                risk_beta=risk_beta,
                router_name=router_name,
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
                    run=None,
                    risk_beta=risk_beta,
                    stats=stats,
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
                targets = graph_targets(single_qtable, ds_train_meta, models)
                graph_model, stats = train_graph_model(
                    ds_x_train,
                    model_features,
                    targets,
                    seed=config.seed + run,
                    config=config,
                )
                router_name = f"graph_router_single_point_{run:03d}"
                predictions, model_predictions = predict_graph_router(
                    graph_model,
                    ds_x_test,
                    model_features,
                    models,
                    risk_beta=0.0,
                    router_name=router_name,
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
                        "single-point",
                        run=run,
                        risk_beta=0.0,
                        stats=stats,
                        config=config,
                    )
                )
                prediction_frames.append(predictions)
                model_prediction_frames.append(model_predictions)
                evaluation_frames.append(detail)

    if not summaries:
        raise ValueError(
            "No graph routers were trained. Check dataset names and input files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df["feature_backend"] = encoder.name
    summary_df = summary_df.merge(
        baselines,
        on=["dataset", "dataset_display"],
        how="left",
        validate="many_to_one",
    )
    summary_df.to_csv(output_dir / "graph_router_summary.csv", index=False)
    summarize_runs(summary_df).to_csv(
        output_dir / "graph_router_summary_by_mode.csv", index=False
    )
    baselines.to_csv(output_dir / "graph_router_test_baselines.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "graph_router_predictions.csv", index=False
    )
    pd.concat(model_prediction_frames, ignore_index=True).to_csv(
        output_dir / "graph_router_model_predictions.csv", index=False
    )
    pd.concat(evaluation_frames, ignore_index=True).to_csv(
        output_dir / "graph_router_evaluation_detail.csv", index=False
    )
    return summary_df


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers or any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError(
            "hidden layers must be positive comma-separated integers"
        )
    return layers


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "graph_router",
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
    parser.add_argument(
        "--local-encoder-path", type=Path, default=DEFAULT_LOCAL_ENCODER
    )
    parser.add_argument(
        "--description-source", choices=("generic", "name"), default="generic"
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
    parser.add_argument("--hidden-layers", type=parse_hidden_layers, default=(256, 128))
    parser.add_argument("--model-feature-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-context", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.single_point_runs <= 0:
        raise ValueError("--single-point-runs must be positive.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.model_feature_dim <= 0:
        raise ValueError("--model-feature-dim must be positive.")
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive.")
    if args.cost_weight < 0 or (args.risk_beta is not None and args.risk_beta < 0):
        raise ValueError("--cost-weight and --risk-beta must be non-negative.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = GraphConfig(
        cost_weight=args.cost_weight,
        risk_beta=args.risk_beta,
        seed=args.seed,
        single_point_runs=args.single_point_runs,
        hidden_layers=args.hidden_layers,
        model_feature_dim=args.model_feature_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_features=args.max_features,
        use_context=not args.no_context,
        description_source=args.description_source,
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
        "n_routers",
        "rewrite_score_mean",
        "rewrite_score_std",
        "decoding_score_mean",
        "decoding_score_std",
        "train_mse_mean",
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
    print(f"\nSaved GraphRouter outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

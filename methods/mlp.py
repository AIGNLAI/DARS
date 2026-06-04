#!/usr/bin/env python3
"""MLP routers for single-point and risk-aware distribution supervision.

The scored generation files under ``data/<dataset>`` already contain one row
per model observation. Training records contain repeated prompt and decoding
observations; test records separate rewritten prompts from original-query
decoding observations through ``observation_type``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = ("gpqa", "math-500", "drop-800")
MODEL_ORDER = (
    "google/gemma-3-12b-it",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3-32b",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-chat-v3.1",
)
DATASET_DISPLAY = {
    "gpqa": "GPQA",
    "math-500": "MATH-500",
    "drop-800": "DROP-800",
}
# MATH labels are more sensitive to uncertainty over-penalization in the
# notebook experiments, so keep its default risk weight weaker.
DATASET_RISK_BETA = {
    "gpqa": 0.1,
    "math-500": 0.05,
    "drop-800": 0.15,
}
DEFAULT_LOCAL_ENCODER = Path("/data/laign/code/LAMDA-ORBIT/all-MiniLM-L6-v2")
JSONL_FIELDS = (
    "dataset",
    "split",
    "query_id",
    "model",
    "score",
    "cost",
    "prompt_variant_id",
    "decode_id",
    "observation_type",
    "original_question",
    "input_question",
    "question",
    "prompt",
    "context",
    "passage",
)


@dataclass(frozen=True)
class RouterConfig:
    cost_weight: float = 0.0
    risk_beta: float | None = None
    seed: int = 42
    single_point_runs: int = 100
    hidden_layers: tuple[int, ...] = (128, 64)
    max_iter: int = 800
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    max_features: int = 20000
    use_context: bool = True


def read_scored_jsonl(path: Path, dataset: str, split: str) -> pd.DataFrame:
    """Read only fields needed for routing from a scored JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc

            row = {field: record.get(field) for field in JSONL_FIELDS}
            row["dataset"] = row["dataset"] or dataset
            row["split"] = row["split"] or split
            rows.append(row)

    if not rows:
        raise ValueError(f"No scored records found in {path}")
    return pd.DataFrame(rows)


def load_scored_data(data_dir: Path, datasets: Sequence[str]) -> pd.DataFrame:
    """Load train and test scored generations for each selected dataset."""
    frames = []
    for dataset in datasets:
        for split in ("train", "test"):
            path = data_dir / dataset / f"{split}_scored_generations.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing scored generations: {path}")
            frames.append(read_scored_jsonl(path, dataset=dataset, split=split))

    df = pd.concat(frames, ignore_index=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df = df.dropna(subset=["dataset", "query_id", "model", "score", "cost"]).copy()
    if df.empty:
        raise ValueError("No records with finite score and cost were loaded.")

    df["dataset_display"] = df["dataset"].map(DATASET_DISPLAY).fillna(df["dataset"])
    cmin = float(df["cost"].min())
    cmax = float(df["cost"].max())
    df["routing_cost"] = 0.0 if cmax <= cmin else (df["cost"] - cmin) / (cmax - cmin)
    return df


def first_nonempty(record: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_query_text_table(split_df: pd.DataFrame, use_context: bool) -> pd.DataFrame:
    """Return one original-query text feature row for each dataset/query pair."""
    rows = []
    for (dataset, query_id), group in split_df.groupby(["dataset", "query_id"], sort=False):
        record = group.iloc[0].to_dict()
        question = first_nonempty(
            record, ("original_question", "input_question", "question", "prompt")
        )
        context = first_nonempty(record, ("context", "passage"))
        text = f"{context[:1200]}\nQuestion: {question}" if use_context and context else question
        rows.append(
            {
                "dataset": dataset,
                "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
                "query_id": query_id,
                "text": text,
            }
        )
    return pd.DataFrame(rows)


def ordered_models(df: pd.DataFrame) -> list[str]:
    """Keep the notebook model order and append any new model names deterministically."""
    present = {str(model) for model in df["model"].dropna().unique()}
    known = [model for model in MODEL_ORDER if model in present]
    unknown = sorted(present.difference(known))
    if not known and not unknown:
        raise ValueError("Cannot train a router without model records.")
    return known + unknown


class QueryEncoder:
    """Encode query text with a local sentence model or a TF-IDF fallback."""

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

    def fit_transform(self, train_texts: Sequence[str], test_texts: Sequence[str]) -> tuple[Any, Any]:
        if self.backend in {"auto", "sentence-transformer"} and self.local_encoder_path.exists():
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                if self.backend == "sentence-transformer":
                    raise
            else:
                self._encoder = SentenceTransformer(str(self.local_encoder_path))
                self.name = str(self.local_encoder_path)
                return self._encode_dense(train_texts), self._encode_dense(test_texts)

        if self.backend == "sentence-transformer":
            raise FileNotFoundError(f"Local sentence encoder not found: {self.local_encoder_path}")

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=1,
            ngram_range=(1, 2),
        )
        self.name = "tfidf"
        return self._vectorizer.fit_transform(train_texts), self._vectorizer.transform(test_texts)

    def _encode_dense(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._encoder.encode(
                list(texts),
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        )


def to_dense_array(features: Any) -> np.ndarray:
    return features.toarray() if hasattr(features, "toarray") else np.asarray(features)


def build_quality_table(records: pd.DataFrame, cost_weight: float) -> pd.DataFrame:
    """Aggregate repeated observations into query/model distribution targets."""
    keys = ["dataset", "dataset_display", "query_id", "model"]
    qtable = (
        records.groupby(keys, sort=False)
        .agg(
            mean_score=("score", "mean"),
            score_std=("score", "std"),
            mean_cost=("routing_cost", "mean"),
            mean_raw_cost=("cost", "mean"),
            n_obs=("score", "size"),
        )
        .reset_index()
    )
    qtable["score_std"] = qtable["score_std"].fillna(0.0)
    qtable["utility"] = qtable["mean_score"] - cost_weight * qtable["mean_cost"]
    return qtable


def get_risk_beta(dataset: str, config: RouterConfig) -> float:
    """Return an explicit beta override or the dataset-tuned default."""
    if config.risk_beta is not None:
        return float(config.risk_beta)
    return float(DATASET_RISK_BETA.get(dataset, 0.10))


def sample_single_point_quality_table(
    train_records: pd.DataFrame,
    seed: int,
    cost_weight: float,
) -> pd.DataFrame:
    """Sample one observation for each training query/model pair."""
    rng = np.random.default_rng(seed)
    keys = ["dataset", "dataset_display", "query_id", "model"]
    rows = []
    for key, group in train_records.groupby(keys, sort=False):
        sampled = group.iloc[int(rng.integers(0, len(group)))]
        score = float(sampled["score"])
        routing_cost = float(sampled["routing_cost"])
        rows.append(
            {
                "dataset": key[0],
                "dataset_display": key[1],
                "query_id": key[2],
                "model": key[3],
                "mean_score": score,
                "score_std": 0.0,
                "mean_cost": routing_cost,
                "mean_raw_cost": float(sampled["cost"]),
                "n_obs": 1,
                "utility": score - cost_weight * routing_cost,
            }
        )
    return pd.DataFrame(rows)


def pivot_targets(
    qtarget: pd.DataFrame,
    query_meta: pd.DataFrame,
    target_columns: Sequence[str],
    models: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return query-aligned multi-model target matrices for selected columns."""
    targets: dict[str, np.ndarray] = {}
    for column in target_columns:
        pivoted = qtarget.pivot_table(
            index="query_id",
            columns="model",
            values=column,
            aggfunc="mean",
        )
        pivoted = pivoted.reindex(index=query_meta["query_id"], columns=models)
        model_means = qtarget.groupby("model")[column].mean()
        for model in models:
            if model in model_means.index:
                pivoted[model] = pivoted[model].fillna(float(model_means[model]))
        fallback = float(qtarget[column].mean()) if qtarget[column].notna().any() else 0.0
        targets[column] = pivoted.fillna(fallback).to_numpy(dtype=float)
    return targets


def fit_regressor(
    features: Any,
    targets: np.ndarray,
    seed: int,
    config: RouterConfig,
) -> Any:
    model = make_pipeline(
        StandardScaler(with_mean=True),
        MLPRegressor(
            hidden_layer_sizes=config.hidden_layers,
            activation="relu",
            solver="adam",
            alpha=config.alpha,
            learning_rate_init=config.learning_rate_init,
            max_iter=config.max_iter,
            early_stopping=False,
            random_state=seed,
        ),
    )
    model.fit(to_dense_array(features), np.asarray(targets, dtype=float))
    return model


def train_predict_router(
    qtarget: pd.DataFrame,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    x_train: Any,
    x_test: Any,
    models: Sequence[str],
    seed: int,
    config: RouterConfig,
    risk_aware: bool,
    risk_beta: float,
    router_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train MLP regressors and select the model with the best routing utility."""
    target_columns = ["mean_score", "mean_cost"]
    if risk_aware:
        target_columns.append("score_std")
    targets = pivot_targets(qtarget, train_meta, target_columns, models)

    score_model = fit_regressor(x_train, targets["mean_score"], seed=seed, config=config)
    cost_model = fit_regressor(x_train, targets["mean_cost"], seed=seed + 991, config=config)
    predicted_score = np.clip(score_model.predict(to_dense_array(x_test)), 0.0, 1.0)
    predicted_cost = np.clip(cost_model.predict(to_dense_array(x_test)), 0.0, 1.0)

    if risk_aware:
        risk_model = fit_regressor(x_train, targets["score_std"], seed=seed + 1991, config=config)
        predicted_std = np.clip(risk_model.predict(to_dense_array(x_test)), 0.0, 1.0)
    else:
        predicted_std = np.zeros_like(predicted_score)

    predicted_utility = predicted_score - config.cost_weight * predicted_cost
    predicted_risk_utility = predicted_utility - risk_beta * predicted_std
    selected_indices = predicted_risk_utility.argmax(axis=1)

    predictions = test_meta[["dataset", "dataset_display", "query_id"]].copy()
    predictions["router"] = router_name
    predictions["pred_model"] = [models[index] for index in selected_indices]

    model_predictions = []
    for row_index, row in test_meta.reset_index(drop=True).iterrows():
        for model_index, model in enumerate(models):
            model_predictions.append(
                {
                    "dataset": row["dataset"],
                    "dataset_display": row["dataset_display"],
                    "query_id": row["query_id"],
                    "router": router_name,
                    "model": model,
                    "pred_score": float(predicted_score[row_index, model_index]),
                    "pred_cost": float(predicted_cost[row_index, model_index]),
                    "pred_score_std": float(predicted_std[row_index, model_index]),
                    "pred_utility": float(predicted_utility[row_index, model_index]),
                    "pred_risk_utility": float(predicted_risk_utility[row_index, model_index]),
                    "risk_beta": float(risk_beta),
                }
            )
    return predictions, pd.DataFrame(model_predictions)


def select_dataset(
    meta: pd.DataFrame,
    features: Any,
    dataset: str,
) -> tuple[pd.DataFrame, Any]:
    mask = meta["dataset"].to_numpy() == dataset
    return meta.loc[mask].reset_index(drop=True), features[mask]


def split_test_records(test_records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rewrite tests from original-query decoding tests."""
    observation_type = test_records.get(
        "observation_type", pd.Series(index=test_records.index, dtype="object")
    ).fillna("")
    rewrite = test_records[observation_type == "prompt_variation"].copy()
    decoding = test_records[observation_type == "decoding_variation"].copy()

    prompt_ids = pd.to_numeric(test_records.get("prompt_variant_id"), errors="coerce")
    if rewrite.empty and prompt_ids.notna().any():
        rewrite = test_records[prompt_ids >= 0].copy()
    if decoding.empty and prompt_ids.notna().any():
        decoding = test_records[prompt_ids < 0].copy()
    return (rewrite if not rewrite.empty else test_records.copy()), (
        decoding if not decoding.empty else test_records.copy()
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
    rewrite_qtable: pd.DataFrame,
    decoding_qtable: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score a routing policy on rewritten and original-query test views."""
    rewrite_lookup = rewrite_qtable.set_index(["dataset", "query_id", "model"])
    decoding_lookup = decoding_qtable.set_index(["dataset", "query_id", "model"])
    rows = []
    for record in predictions.itertuples(index=False):
        key = (record.dataset, record.query_id, record.pred_model)
        if key not in rewrite_lookup.index or key not in decoding_lookup.index:
            continue
        rewrite = rewrite_lookup.loc[key]
        decoding = decoding_lookup.loc[key]
        rows.append(
            {
                "dataset": record.dataset,
                "dataset_display": record.dataset_display,
                "query_id": record.query_id,
                "router": record.router,
                "pred_model": record.pred_model,
                "rewrite_score": float(rewrite["mean_score"]),
                "rewrite_cost": float(rewrite["mean_cost"]),
                "rewrite_n_obs": int(rewrite["n_obs"]),
                "decoding_score": float(decoding["mean_score"]),
                "decoding_cost": float(decoding["mean_cost"]),
                "decoding_score_std": float(decoding["score_std"]),
                "decoding_n_obs": int(decoding["n_obs"]),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("Predictions could not be matched to test observations.")
    summary = {
        "rewrite_score": float(detail["rewrite_score"].mean()),
        "rewrite_cost": float(detail["rewrite_cost"].mean()),
        "rewrite_observations_per_query": float(detail["rewrite_n_obs"].mean()),
        "decoding_score": float(detail["decoding_score"].mean()),
        "decoding_cost": float(detail["decoding_cost"].mean()),
        "decoding_score_std": float(detail["decoding_score_std"].mean()),
        "decoding_observations_per_query": float(detail["decoding_n_obs"].mean()),
        "n_eval": int(len(detail)),
    }
    return summary, detail


def score_baselines(qtable: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return best fixed-model and per-query Oracle scores for one test view."""
    fixed_model_scores = (
        qtable.groupby(["dataset", "dataset_display", "model"], sort=False)["mean_score"]
        .mean()
        .reset_index()
    )
    best_indices = fixed_model_scores.groupby("dataset")["mean_score"].idxmax()
    best_fixed = fixed_model_scores.loc[
        best_indices, ["dataset", "dataset_display", "model", "mean_score"]
    ].rename(
        columns={
            "model": f"{prefix}_best_model",
            "mean_score": f"{prefix}_best_model_score",
        }
    )

    oracle = (
        qtable.groupby(["dataset", "dataset_display", "query_id"], sort=False)["mean_score"]
        .max()
        .groupby(["dataset", "dataset_display"], sort=False)
        .mean()
        .reset_index(name=f"{prefix}_oracle_score")
    )
    return best_fixed.merge(oracle, on=["dataset", "dataset_display"], how="inner")


def build_test_baselines(rewrite_qtable: pd.DataFrame, decoding_qtable: pd.DataFrame) -> pd.DataFrame:
    """Build score baselines for rewritten prompts and original-query decoding."""
    rewrite = score_baselines(rewrite_qtable, "rewrite")
    decoding = score_baselines(decoding_qtable, "decoding")
    return rewrite.merge(decoding, on=["dataset", "dataset_display"], how="inner")


def add_summary_metadata(
    summary: dict[str, Any],
    dataset: str,
    mode: str,
    run: int | None,
    config: RouterConfig,
    risk_beta: float,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
        "mode": mode,
        "run": run,
        "cost_weight": config.cost_weight,
        "risk_beta": risk_beta,
        **summary,
    }


def summarize_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated single-point runs while keeping distribution rows comparable."""
    grouping = [
        "dataset",
        "dataset_display",
        "mode",
        "cost_weight",
        "risk_beta",
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
    return aggregate.fillna(
        {
            "rewrite_score_std": 0.0,
            "decoding_score_std": 0.0,
        }
    )


def run_experiment(
    data_dir: Path,
    datasets: Sequence[str],
    output_dir: Path,
    mode: str,
    feature_backend: str,
    local_encoder_path: Path,
    config: RouterConfig,
) -> pd.DataFrame:
    """Load data, train selected routers, evaluate, and write CSV outputs."""
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
            router_name = "risk_aware_distribution"
            risk_beta = get_risk_beta(dataset, config)
            predictions, model_predictions = train_predict_router(
                qtarget=train_distribution_qtable[
                    train_distribution_qtable["dataset"] == dataset
                ],
                train_meta=ds_train_meta,
                test_meta=ds_test_meta,
                x_train=ds_x_train,
                x_test=ds_x_test,
                models=models,
                seed=config.seed,
                config=config,
                risk_aware=True,
                risk_beta=risk_beta,
                router_name=router_name,
            )
            summary, detail = evaluate_predictions(predictions, rewrite_qtable, decoding_qtable)
            summaries.append(
                add_summary_metadata(
                    summary,
                    dataset,
                    "distribution",
                    run=None,
                    config=config,
                    risk_beta=risk_beta,
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
                router_name = f"single_point_{run:03d}"
                predictions, model_predictions = train_predict_router(
                    qtarget=single_qtable,
                    train_meta=ds_train_meta,
                    test_meta=ds_test_meta,
                    x_train=ds_x_train,
                    x_test=ds_x_test,
                    models=models,
                    seed=config.seed + run,
                    config=config,
                    risk_aware=False,
                    risk_beta=0.0,
                    router_name=router_name,
                )
                summary, detail = evaluate_predictions(predictions, rewrite_qtable, decoding_qtable)
                summaries.append(
                    add_summary_metadata(
                        summary,
                        dataset,
                        "single-point",
                        run=run,
                        config=config,
                        risk_beta=0.0,
                    )
                )
                prediction_frames.append(predictions)
                model_prediction_frames.append(model_predictions)
                evaluation_frames.append(detail)

    if not summaries:
        raise ValueError("No routers were trained. Check dataset names and input files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df["feature_backend"] = encoder.name
    summary_df = summary_df.merge(
        baselines,
        on=["dataset", "dataset_display"],
        how="left",
        validate="many_to_one",
    )
    summary_df.to_csv(output_dir / "mlp_summary.csv", index=False)
    summarize_runs(summary_df).to_csv(output_dir / "mlp_summary_by_mode.csv", index=False)
    baselines.to_csv(output_dir / "mlp_test_baselines.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "mlp_predictions.csv", index=False
    )
    pd.concat(model_prediction_frames, ignore_index=True).to_csv(
        output_dir / "mlp_model_predictions.csv", index=False
    )
    pd.concat(evaluation_frames, ignore_index=True).to_csv(
        output_dir / "mlp_evaluation_detail.csv", index=False
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
    parser.add_argument("--output-dir", type=Path, default=project_dir / "analysis_outputs_mlp")
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
        help="Override risk penalty beta. Default uses 0.02 for MATH-500 and 0.10 otherwise.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-layers", type=parse_hidden_layers, default=(128, 64))
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--no-context", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.single_point_runs <= 0:
        raise ValueError("--single-point-runs must be positive.")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive.")
    if args.max_features <= 0:
        raise ValueError("--max-features must be positive.")
    if args.cost_weight < 0 or (args.risk_beta is not None and args.risk_beta < 0):
        raise ValueError("--cost-weight and --risk-beta must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = RouterConfig(
        cost_weight=args.cost_weight,
        risk_beta=args.risk_beta,
        seed=args.seed,
        single_point_runs=args.single_point_runs,
        hidden_layers=args.hidden_layers,
        max_iter=args.max_iter,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        max_features=args.max_features,
        use_context=not args.no_context,
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
    columns = [
        "dataset_display",
        "mode",
        "risk_beta",
        "n_routers",
        "rewrite_score_mean",
        "rewrite_score_std",
        "decoding_score_mean",
        "decoding_score_std",
        "chosen_decoding_instability_mean",
        "n_eval_mean",
        "feature_backend",
    ]
    print(summarize_runs(summary)[columns].to_string(index=False))
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
    print(f"\nSaved MLP outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

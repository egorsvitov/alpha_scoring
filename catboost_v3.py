from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from catboost_baseline import FEATURE_DIR, ROOT, SAMPLE_SUBMISSION, TEST_DATA, TRAIN_DATA, TRAIN_TARGET, make_train_valid, reduce_memory
from catboost_v2 import (
    NOMINAL_COLUMNS,
    align_train_test,
    build_nominal_features,
    category_code,
    flatten_columns,
    read_source,
)


RECENT_WINDOWS = (3, 5, 10)
RECENT_STATS = ("mean", "max", "std")


SEARCH_CONFIGS = (
    {"name": "d5_lr05", "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 6.0},
    {"name": "d6_lr03", "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 6.0},
    {"name": "d6_lr05", "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
    {"name": "d7_lr03", "depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 10.0},
    {"name": "d7_lr05", "depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 15.0},
    {"name": "d8_lr03", "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 15.0},
    {"name": "d6_reg30", "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 30.0},
    {"name": "d7_rs05", "depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 10.0, "random_strength": 0.5},
    {"name": "d7_rs2", "depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 10.0, "random_strength": 2.0},
    {"name": "d6_bag02", "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 10.0, "bagging_temperature": 0.2},
    {"name": "d6_bag1", "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 10.0, "bagging_temperature": 1.0},
    {"name": "d6_no_weights", "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 10.0, "class_weights": "none"},
)


def build_recent_numeric_features(
    recent_df: pd.DataFrame,
    numeric_cols: list[str],
    full_features: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    recent = recent_df.groupby("id", sort=False)[numeric_cols].agg(list(RECENT_STATS))
    recent.columns = [f"recent{window}_{col}_{stat}" for col, stat in recent.columns.to_flat_index()]

    deltas: dict[str, pd.Series] = {}
    for col in numeric_cols:
        recent_mean = f"recent{window}_{col}_mean"
        full_mean = f"{col}_mean"
        if full_mean in full_features.columns:
            deltas[f"recent{window}_{col}_mean_delta_full"] = recent[recent_mean] - full_features[full_mean]
    return pd.concat([recent, pd.DataFrame(deltas, index=recent.index)], axis=1)


def build_recent_nominal_features(recent_df: pd.DataFrame, window: int) -> pd.DataFrame:
    grouped = recent_df.groupby("id", sort=False)
    result = pd.DataFrame(index=grouped.size().index)
    for col in NOMINAL_COLUMNS:
        if col not in recent_df.columns:
            continue
        counts = pd.crosstab(recent_df["id"], recent_df[col], dropna=False).reindex(result.index, fill_value=0)
        shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        shares.columns = [f"recent{window}_{col}_share_{category_code(value)}" for value in shares.columns]
        result = result.join(shares.astype(np.float32))
        result[f"recent{window}_{col}_nunique"] = grouped[col].nunique(dropna=True)
        result[f"recent{window}_{col}_mode_share"] = shares.max(axis=1).astype(np.float32)
    return result


def build_features(path: Path, cache_path: Path, force: bool = False, max_rows: int | None = None) -> pd.DataFrame:
    if cache_path.exists() and not force:
        print(f"Load cached features: {cache_path}")
        return pd.read_parquet(cache_path)

    df = read_source(path, max_rows=max_rows).sort_values(["id", "rn"])
    nominal_cols = [col for col in NOMINAL_COLUMNS if col in df.columns]
    numeric_cols = [col for col in df.columns if col not in {"id", "rn", *nominal_cols}]
    grouped = df.groupby("id", sort=False)
    print(f"Rows: {len(df)}, ids: {df['id'].nunique()}, numeric columns: {len(numeric_cols)}")

    print("Build full-history numeric aggregations")
    features = grouped[numeric_cols].agg(["mean", "sum", "min", "max", "std", "nunique"])
    features.columns = flatten_columns(features.columns)

    print("Build full-history nominal features")
    features = features.join(build_nominal_features(df, grouped), how="left")

    rn_features = grouped["rn"].agg(["count", "min", "max", "mean"])
    rn_features.columns = ["products_count", "rn_min", "rn_max", "rn_mean"]
    features = features.join(rn_features)

    first_rows = df.drop_duplicates("id", keep="first").set_index("id")[numeric_cols]
    first_rows.columns = [f"first_{col}" for col in numeric_cols]
    last_rows = df.drop_duplicates("id", keep="last").set_index("id")[numeric_cols]
    last_rows.columns = [f"last_{col}" for col in numeric_cols]
    features = features.join(first_rows).join(last_rows)

    for window in RECENT_WINDOWS:
        print(f"Build features for last {window} products")
        recent_df = grouped.tail(window)
        recent_numeric = build_recent_numeric_features(recent_df, numeric_cols, features, window)
        recent_nominal = build_recent_nominal_features(recent_df, window)
        features = features.join(recent_numeric).join(recent_nominal)

    features = features.reset_index()
    features = reduce_memory(features.replace([np.inf, -np.inf], np.nan))
    FEATURE_DIR.mkdir(exist_ok=True)
    features.to_parquet(cache_path, index=False)
    print(f"Saved features: {cache_path}, shape={features.shape}")
    return features


def model_params(args: argparse.Namespace, overrides: dict[str, object] | None = None) -> dict[str, object]:
    overrides = overrides or {}
    class_weights = overrides.get("class_weights", args.class_weights)
    params: dict[str, object] = {
        "iterations": args.iterations,
        "learning_rate": overrides.get("learning_rate", args.learning_rate),
        "depth": overrides.get("depth", args.depth),
        "l2_leaf_reg": overrides.get("l2_leaf_reg", args.l2_leaf_reg),
        "random_strength": overrides.get("random_strength", args.random_strength),
        "bagging_temperature": overrides.get("bagging_temperature", args.bagging_temperature),
        "border_count": args.border_count,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": args.random_seed,
        "verbose": args.verbose,
        "allow_writing_files": False,
        "task_type": args.task_type,
    }
    if class_weights == "balanced":
        params["auto_class_weights"] = "Balanced"
    if args.task_type == "GPU":
        params["devices"] = args.devices
    return params


def fit_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    args: argparse.Namespace,
    use_best_model: bool,
    overrides: dict[str, object] | None = None,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**model_params(args, overrides))
    eval_set = Pool(x_valid, y_valid) if x_valid is not None and y_valid is not None else None
    model.fit(
        Pool(x_train, y_train),
        eval_set=eval_set,
        use_best_model=use_best_model,
        early_stopping_rounds=args.early_stopping_rounds if eval_set is not None else None,
    )
    return model


def run_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, config in enumerate(SEARCH_CONFIGS, start=1):
        print(f"\nSearch {index}/{len(SEARCH_CONFIGS)}: {json.dumps(config, ensure_ascii=True)}")
        model = fit_model(x_train, y_train, x_valid, y_valid, args, True, config)
        prediction = model.predict_proba(x_valid)[:, 1]
        row = {
            **config,
            "auc": roc_auc_score(y_valid, prediction),
            "best_iteration": model.get_best_iteration(),
        }
        rows.append(row)
        results = pd.DataFrame(rows).sort_values("auc", ascending=False)
        results.to_csv(args.search_output, index=False)
        print(results.to_string(index=False))
        print(f"Saved search results: {args.search_output}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CatBoost V3 with recent-product aggregations and parameter search.")
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--train-target", type=Path, default=TRAIN_TARGET)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=6.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=1.0)
    parser.add_argument("--border-count", type=int, default=128)
    parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=250)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--search-output", type=Path, default=ROOT / "catboost_v3_search.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "submission_catboost_v3.csv")
    parser.add_argument("--model-output", type=Path, default=ROOT / "catboost_v3.cbm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_suffix = f"_{args.max_rows}" if args.max_rows is not None else ""
    train_features = build_features(
        args.train_data,
        FEATURE_DIR / f"train_features_v3{cache_suffix}.parquet",
        force=args.force_features,
        max_rows=args.max_rows,
    )
    target = pd.read_csv(args.train_target)
    x_train, x_valid, y_train, y_valid, valid_ids = make_train_valid(
        train_features, target, valid_fraction=args.valid_fraction
    )
    print(f"Train shape: {x_train.shape}, valid shape: {x_valid.shape}")
    print(f"Valid id range: {int(valid_ids.min())}..{int(valid_ids.max())}")

    if args.search:
        run_search(x_train, y_train, x_valid, y_valid, args)
        return

    model = fit_model(x_train, y_train, x_valid, y_valid, args, True)
    valid_prediction = model.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_prediction)
    best_iteration = model.get_best_iteration()
    print(f"Validation ROC-AUC: {auc:.6f}")
    print(f"Best iteration: {best_iteration}")
    if args.validate_only:
        return

    test_features = build_features(
        args.test_data,
        FEATURE_DIR / f"test_features_v3{cache_suffix}.parquet",
        force=args.force_features,
        max_rows=args.max_rows,
    )
    _, x_test = align_train_test(train_features, test_features)
    full_data = train_features.merge(target, on="id", how="inner").sort_values("id").reset_index(drop=True)
    x_full = full_data.drop(columns=["id", "flag"])
    y_full = full_data["flag"]
    if best_iteration is not None and best_iteration > 0:
        args.iterations = best_iteration + 1
    final_model = fit_model(x_full, y_full, None, None, args, False)
    final_model.save_model(args.model_output)

    sample = pd.read_csv(args.sample_submission)
    prediction = final_model.predict_proba(x_test)[:, 1]
    test_prediction = pd.DataFrame({"id": test_features["id"].values, "flag": prediction})
    submission = sample[["id"]].merge(test_prediction, on="id", how="left")
    if submission["flag"].isna().any():
        raise RuntimeError(f"Submission has missing predictions: {int(submission['flag'].isna().sum())}")
    submission["flag"] = submission["flag"].clip(0.0, 1.0)
    submission.to_csv(args.output, index=False)
    print(f"Saved model: {args.model_output}")
    print(f"Saved submission: {args.output}, shape={submission.shape}")


if __name__ == "__main__":
    main()

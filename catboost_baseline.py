from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
TRAIN_DATA = ROOT / "train_data.parquet"
TEST_DATA = ROOT / "test_data.parquet"
TRAIN_TARGET = ROOT / "train_target.csv"
SAMPLE_SUBMISSION = ROOT / "sample_submission.csv"
FEATURE_DIR = ROOT / "features"


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max <= np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_max <= np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_max <= np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
            elif np.iinfo(np.int8).min <= c_min <= c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif np.iinfo(np.int16).min <= c_min <= c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif np.iinfo(np.int32).min <= c_min <= c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(np.float32)
    return df


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    paym_cols = [col for col in df.columns if col.startswith("enc_paym_")]
    overdue_cols = [
        col
        for col in ("pre_loans5", "pre_loans530", "pre_loans3060", "pre_loans6090", "pre_loans90")
        if col in df.columns
    ]
    zero_overdue_cols = [
        col
        for col in (
            "is_zero_loans_5",
            "is_zero_loans_530",
            "is_zero_loans_3060",
            "is_zero_loans_6090",
            "is_zero_loans90",
        )
        if col in df.columns
    ]

    if paym_cols:
        paym = df[paym_cols]
        df["paym_mean"] = paym.mean(axis=1).astype(np.float32)
        df["paym_sum"] = paym.sum(axis=1).astype(np.int16)
        df["paym_min"] = paym.min(axis=1).astype(np.int16)
        df["paym_max"] = paym.max(axis=1).astype(np.int16)
        df["paym_std"] = paym.std(axis=1).fillna(0).astype(np.float32)
        df["paym_bad_cnt"] = (paym >= 3).sum(axis=1).astype(np.int16)
        df["paym_zero_cnt"] = (paym == 0).sum(axis=1).astype(np.int16)

    if overdue_cols:
        overdue = df[overdue_cols]
        df["overdue_sum"] = overdue.sum(axis=1).astype(np.int16)
        df["overdue_max"] = overdue.max(axis=1).astype(np.int16)
        df["overdue_nonzero_cnt"] = (overdue > 0).sum(axis=1).astype(np.int16)

    if zero_overdue_cols:
        df["zero_overdue_flags_sum"] = df[zero_overdue_cols].sum(axis=1).astype(np.int16)

    return df


def read_source(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    print(f"Read parquet: {path}")
    df = pd.read_parquet(path)
    if max_rows is not None:
        df = df.head(max_rows).copy()
        print(f"Use first rows only: {len(df)}")
    return reduce_memory(add_row_features(df))


def flatten_columns(columns: pd.MultiIndex) -> list[str]:
    return [f"{col}_{stat}" for col, stat in columns.to_flat_index()]


def build_features(
    path: Path,
    cache_path: Path,
    force: bool = False,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if cache_path.exists() and not force:
        print(f"Load cached features: {cache_path}")
        return pd.read_parquet(cache_path)

    df = read_source(path, max_rows=max_rows).sort_values(["id", "rn"])
    feature_cols = [col for col in df.columns if col not in {"id", "rn"}]
    print(f"Rows: {len(df)}, ids: {df['id'].nunique()}, row feature columns: {len(feature_cols)}")

    grouped = df.groupby("id", sort=False)

    print("Build group aggregations")
    agg_map = {col: ["mean", "sum", "min", "max", "std", "nunique"] for col in feature_cols}
    features = grouped.agg(agg_map)
    features.columns = flatten_columns(features.columns)
    features = features.reset_index()

    print("Add rn/count features")
    rn_features = grouped["rn"].agg(["count", "min", "max", "mean"]).reset_index()
    rn_features.columns = ["id", "products_count", "rn_min", "rn_max", "rn_mean"]
    features = features.merge(rn_features, on="id", how="left")

    print("Add first/last product features")
    first_rows = df.drop_duplicates("id", keep="first")[["id", *feature_cols]].copy()
    first_rows = first_rows.rename(columns={col: f"first_{col}" for col in feature_cols})
    last_rows = df.drop_duplicates("id", keep="last")[["id", *feature_cols]].copy()
    last_rows = last_rows.rename(columns={col: f"last_{col}" for col in feature_cols})
    features = features.merge(first_rows, on="id", how="left")
    features = features.merge(last_rows, on="id", how="left")

    features = reduce_memory(features.replace([np.inf, -np.inf], np.nan))
    FEATURE_DIR.mkdir(exist_ok=True)
    features.to_parquet(cache_path, index=False)
    print(f"Saved features: {cache_path}, shape={features.shape}")
    return features


def make_train_valid(
    train_features: pd.DataFrame,
    target: pd.DataFrame,
    valid_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index]:
    data = train_features.merge(target, on="id", how="inner").sort_values("id").reset_index(drop=True)
    split = int(len(data) * (1.0 - valid_fraction))
    train_part = data.iloc[:split].copy()
    valid_part = data.iloc[split:].copy()

    valid_ids = valid_part["id"].copy()
    y_train = train_part.pop("flag")
    y_valid = valid_part.pop("flag")
    x_train = train_part.drop(columns=["id"])
    x_valid = valid_part.drop(columns=["id"])
    return x_train, x_valid, y_train, y_valid, valid_ids


def align_train_test(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_cols = [col for col in train_features.columns if col != "id"]
    missing = sorted(set(train_cols) - set(test_features.columns))
    if missing:
        raise RuntimeError(f"Test features miss {len(missing)} train columns, first: {missing[:5]}")
    return train_features[train_cols], test_features[train_cols]


def fit_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame | None,
    y_valid: pd.Series | None,
    args: argparse.Namespace,
    use_best_model: bool,
) -> CatBoostClassifier:
    params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_seed": args.random_seed,
        "verbose": args.verbose,
        "allow_writing_files": False,
        "task_type": args.task_type,
    }
    if args.task_type == "GPU":
        params["devices"] = args.devices

    model = CatBoostClassifier(**params)
    eval_set = Pool(x_valid, y_valid) if x_valid is not None and y_valid is not None else None
    model.fit(
        Pool(x_train, y_train),
        eval_set=eval_set,
        use_best_model=use_best_model,
        early_stopping_rounds=args.early_stopping_rounds if eval_set is not None else None,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Full CatBoost baseline for credit scoring.")
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--train-target", type=Path, default=TRAIN_TARGET)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=6.0)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None, help="Use only first N parquet rows for server smoke runs.")
    parser.add_argument("--validate-only", action="store_true", help="Stop after validation and do not build test/submission.")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "submission_catboost.csv")
    parser.add_argument("--model-output", type=Path, default=ROOT / "catboost_baseline.cbm")
    args = parser.parse_args()

    cache_suffix = f"_{args.max_rows}" if args.max_rows is not None else ""
    train_features = build_features(
        args.train_data,
        FEATURE_DIR / f"train_features_full{cache_suffix}.parquet",
        force=args.force_features,
        max_rows=args.max_rows,
    )
    target = pd.read_csv(args.train_target)
    print(f"Target shape: {target.shape}, mean: {target['flag'].mean():.6f}")

    x_train, x_valid, y_train, y_valid, valid_ids = make_train_valid(
        train_features,
        target,
        valid_fraction=args.valid_fraction,
    )
    print(f"Train shape: {x_train.shape}, valid shape: {x_valid.shape}")
    print(f"Valid id range: {int(valid_ids.min())}..{int(valid_ids.max())}")

    model = fit_model(x_train, y_train, x_valid, y_valid, args, use_best_model=True)
    valid_pred = model.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    best_iteration = model.get_best_iteration()
    print(f"Validation ROC-AUC: {auc:.6f}")
    print(f"Best iteration: {best_iteration}")

    if args.validate_only:
        return

    test_features = build_features(
        args.test_data,
        FEATURE_DIR / f"test_features_full{cache_suffix}.parquet",
        force=args.force_features,
        max_rows=args.max_rows,
    )
    x_train_full, x_test = align_train_test(train_features, test_features)

    print("Refit on all train ids")
    full_data = train_features.merge(target, on="id", how="inner").sort_values("id").reset_index(drop=True)
    y_full = full_data["flag"]
    x_full = full_data[x_train_full.columns]
    if best_iteration is not None and best_iteration > 0:
        args.iterations = best_iteration + 1
    final_model = fit_model(x_full, y_full, None, None, args, use_best_model=False)
    final_model.save_model(args.model_output)
    print(f"Saved model: {args.model_output}")

    sample = pd.read_csv(args.sample_submission)
    pred = final_model.predict_proba(x_test)[:, 1]
    test_pred = pd.DataFrame({"id": test_features["id"].values, "flag": pred})
    submission = sample[["id"]].merge(test_pred, on="id", how="left")

    if submission["flag"].isna().any():
        missing = int(submission["flag"].isna().sum())
        raise RuntimeError(f"Submission has missing predictions: {missing}")

    submission["flag"] = submission["flag"].clip(0.0, 1.0)
    submission.to_csv(args.output, index=False)
    print(f"Saved submission: {args.output}, shape={submission.shape}")


if __name__ == "__main__":
    main()

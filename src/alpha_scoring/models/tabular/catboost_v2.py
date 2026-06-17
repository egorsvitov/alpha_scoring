from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from alpha_scoring.models.tabular.catboost_baseline import (
    FEATURE_DIR,
    ROOT,
    SAMPLE_SUBMISSION,
    TEST_DATA,
    TRAIN_DATA,
    TRAIN_TARGET,
    fit_model,
    make_train_valid,
    reduce_memory,
)


NOMINAL_COLUMNS = (
    "enc_loans_account_holder_type",
    "enc_loans_credit_status",
    "enc_loans_account_cur",
    "enc_loans_credit_type",
)
PAYMENT_BAD_THRESHOLD = 3


def payment_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("enc_paym_")]
    return sorted(cols, key=lambda col: int(col.rsplit("_", 1)[1]))


def longest_true_run(values: np.ndarray) -> np.ndarray:
    longest = np.zeros(values.shape[0], dtype=np.int16)
    current = np.zeros(values.shape[0], dtype=np.int16)
    for column in values.T:
        current = np.where(column, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def add_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    paym_cols = payment_columns(df)
    if not paym_cols:
        return df

    paym = df[paym_cols]
    values = paym.to_numpy(dtype=np.float32, copy=False)
    bad = values >= PAYMENT_BAD_THRESHOLD

    df["paym_mean"] = np.nanmean(values, axis=1).astype(np.float32)
    df["paym_sum"] = np.nansum(values, axis=1).astype(np.float32)
    df["paym_min"] = np.nanmin(values, axis=1).astype(np.float32)
    df["paym_max"] = np.nanmax(values, axis=1).astype(np.float32)
    df["paym_std"] = np.nanstd(values, axis=1).astype(np.float32)
    df["paym_bad_cnt"] = bad.sum(axis=1).astype(np.int16)
    df["paym_zero_cnt"] = (values == 0).sum(axis=1).astype(np.int16)
    df["paym_change_cnt"] = (values[:, 1:] != values[:, :-1]).sum(axis=1).astype(np.int16)
    df["paym_bad_longest_run"] = longest_true_run(bad)
    df["paym_latest"] = values[:, 0]
    df["paym_oldest"] = values[:, -1]
    df["paym_latest_minus_oldest"] = values[:, 0] - values[:, -1]

    month_index = np.arange(values.shape[1], dtype=np.float32)
    centered_month = month_index - month_index.mean()
    denominator = float(np.square(centered_month).sum())
    if denominator > 0:
        centered_values = values - np.nanmean(values, axis=1, keepdims=True)
        df["paym_trend"] = np.nansum(centered_values * centered_month, axis=1) / denominator

    for window in (3, 6, 12):
        if values.shape[1] < window:
            continue
        recent = values[:, :window]
        recent_bad = recent >= PAYMENT_BAD_THRESHOLD
        prefix = f"paym_recent_{window}"
        df[f"{prefix}_mean"] = np.nanmean(recent, axis=1).astype(np.float32)
        df[f"{prefix}_max"] = np.nanmax(recent, axis=1).astype(np.float32)
        df[f"{prefix}_bad_cnt"] = recent_bad.sum(axis=1).astype(np.int16)
        df[f"{prefix}_bad_rate"] = recent_bad.mean(axis=1).astype(np.float32)

        older = values[:, window:]
        if older.shape[1]:
            df[f"{prefix}_vs_older"] = (
                np.nanmean(recent, axis=1) - np.nanmean(older, axis=1)
            ).astype(np.float32)

    return df


def add_overdue_features(df: pd.DataFrame) -> pd.DataFrame:
    overdue_cols = [
        col
        for col in ("pre_loans5", "pre_loans530", "pre_loans3060", "pre_loans6090", "pre_loans90")
        if col in df.columns
    ]
    zero_cols = [
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
    if overdue_cols:
        overdue = df[overdue_cols]
        df["overdue_sum"] = overdue.sum(axis=1).astype(np.int16)
        df["overdue_max"] = overdue.max(axis=1).astype(np.int16)
        df["overdue_nonzero_cnt"] = (overdue > 0).sum(axis=1).astype(np.int16)
    if zero_cols:
        df["zero_overdue_flags_sum"] = df[zero_cols].sum(axis=1).astype(np.int16)
    return df


def read_source(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    print(f"Read parquet: {path}")
    df = pd.read_parquet(path)
    if max_rows is not None:
        df = df.head(max_rows).copy()
        print(f"Use first rows only: {len(df)}")
    df = add_payment_features(df)
    df = add_overdue_features(df)
    return reduce_memory(df)


def flatten_columns(columns: pd.MultiIndex) -> list[str]:
    return [f"{col}_{stat}" for col, stat in columns.to_flat_index()]


def category_code(value: object) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p").replace("-", "neg")


def category_indicators(values: pd.Series, prefix: str) -> pd.DataFrame:
    indicators = pd.get_dummies(values, dummy_na=True, dtype=np.uint8)
    indicators.columns = [f"{prefix}_{category_code(value)}" for value in indicators.columns]
    return indicators


def build_nominal_features(df: pd.DataFrame, grouped: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    result = pd.DataFrame(index=grouped.size().index)

    for col in NOMINAL_COLUMNS:
        if col not in df.columns:
            continue

        counts = pd.crosstab(df["id"], df[col], dropna=False).reindex(result.index, fill_value=0)
        counts.columns = [f"{col}_count_{category_code(value)}" for value in counts.columns]
        shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        shares.columns = [name.replace("_count_", "_share_") for name in counts.columns]

        mode_share = counts.max(axis=1) / counts.sum(axis=1).replace(0, np.nan)
        entropy = -(shares * np.log(shares.where(shares > 0))).sum(axis=1)
        changes = grouped[col].agg(lambda values: values.ne(values.shift()).sum() - 1)
        first_indicators = category_indicators(grouped[col].first(), f"{col}_first")
        last_indicators = category_indicators(grouped[col].last(), f"{col}_last")

        result = result.join(counts.astype(np.int32))
        result = result.join(shares.astype(np.float32))
        result = result.join(first_indicators).join(last_indicators)
        result[f"{col}_nunique"] = grouped[col].nunique(dropna=True)
        result[f"{col}_mode_share"] = mode_share.astype(np.float32)
        result[f"{col}_entropy"] = entropy.astype(np.float32)
        result[f"{col}_change_cnt"] = changes.clip(lower=0).astype(np.int16)

    return result


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
    nominal_cols = [col for col in NOMINAL_COLUMNS if col in df.columns]
    numeric_cols = [col for col in df.columns if col not in {"id", "rn", *nominal_cols}]
    grouped = df.groupby("id", sort=False)
    print(f"Rows: {len(df)}, ids: {df['id'].nunique()}, numeric columns: {len(numeric_cols)}")

    print("Build numeric aggregations")
    features = grouped[numeric_cols].agg(["mean", "sum", "min", "max", "std", "nunique"])
    features.columns = flatten_columns(features.columns)

    print("Build nominal count/share features")
    features = features.join(build_nominal_features(df, grouped), how="left")

    print("Add rn/count features")
    rn_features = grouped["rn"].agg(["count", "min", "max", "mean"])
    rn_features.columns = ["products_count", "rn_min", "rn_max", "rn_mean"]
    features = features.join(rn_features)

    print("Add first/last numeric product features")
    first_rows = df.drop_duplicates("id", keep="first").set_index("id")[numeric_cols]
    first_rows.columns = [f"first_{col}" for col in numeric_cols]
    last_rows = df.drop_duplicates("id", keep="last").set_index("id")[numeric_cols]
    last_rows.columns = [f"last_{col}" for col in numeric_cols]
    features = features.join(first_rows).join(last_rows)

    features = features.reset_index()
    features = reduce_memory(features.replace([np.inf, -np.inf], np.nan))
    FEATURE_DIR.mkdir(exist_ok=True)
    features.to_parquet(cache_path, index=False)
    print(f"Saved features: {cache_path}, shape={features.shape}")
    return features


def align_train_test(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_features.drop(columns="id").copy()
    test = test_features.drop(columns="id").copy()
    for col in train.columns.difference(test.columns):
        test[col] = 0
    extra_test_cols = test.columns.difference(train.columns)
    if len(extra_test_cols):
        print(f"Ignore {len(extra_test_cols)} test-only category columns")
    return train, test.reindex(columns=train.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="CatBoost V2 with nominal and temporal features.")
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--train-target", type=Path, default=TRAIN_TARGET)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=6.0)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=250)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "submission_catboost_v2.csv")
    parser.add_argument("--model-output", type=Path, default=ROOT / "catboost_v2.cbm")
    args = parser.parse_args()

    cache_suffix = f"_{args.max_rows}" if args.max_rows is not None else ""
    train_features = build_features(
        args.train_data,
        FEATURE_DIR / f"train_features_v2{cache_suffix}.parquet",
        force=args.force_features,
        max_rows=args.max_rows,
    )
    target = pd.read_csv(args.train_target)
    print(f"Target shape: {target.shape}, mean: {target['flag'].mean():.6f}")

    x_train, x_valid, y_train, y_valid, valid_ids = make_train_valid(
        train_features, target, valid_fraction=args.valid_fraction
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
        FEATURE_DIR / f"test_features_v2{cache_suffix}.parquet",
        force=args.force_features,
        max_rows=args.max_rows,
    )
    _, x_test = align_train_test(train_features, test_features)
    full_data = train_features.merge(target, on="id", how="inner").sort_values("id").reset_index(drop=True)
    x_full = full_data.drop(columns=["id", "flag"])
    y_full = full_data["flag"]
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
        raise RuntimeError(f"Submission has missing predictions: {int(submission['flag'].isna().sum())}")
    submission["flag"] = submission["flag"].clip(0.0, 1.0)
    submission.to_csv(args.output, index=False)
    print(f"Saved submission: {args.output}, shape={submission.shape}")


if __name__ == "__main__":
    main()

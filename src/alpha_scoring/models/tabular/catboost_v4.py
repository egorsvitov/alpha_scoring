from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from alpha_scoring.models.tabular.catboost_baseline import FEATURE_DIR, ROOT, TRAIN_DATA, TRAIN_TARGET
from alpha_scoring.models.tabular.catboost_v3 import fit_model


BIN_COLUMNS = {
    "pre_since_opened": range(20),
    "pre_since_confirmed": range(18),
    "pre_loans_credit_limit": range(20),
    "pre_loans_outstanding": range(1, 6),
    "pre_loans_total_overdue": range(2),
    "pre_loans_max_overdue_sum": range(4),
    "pre_loans90": (2, 3, 8, 10, 13, 14, 19),
    "pre_util": range(20),
    "pre_over2limit": range(20),
    "pre_maxover2limit": range(20),
}

STATUS_NUMERIC_COLUMNS = (
    "pre_loans_credit_limit",
    "pre_loans_outstanding",
    "pre_loans_total_overdue",
    "pre_loans_max_overdue_sum",
    "pre_loans90",
    "pre_util",
    "pre_over2limit",
    "pre_maxover2limit",
)


def payment_expressions() -> list[pl.Expr]:
    paym = [f"enc_paym_{index}" for index in range(25)]
    return [
        pl.mean_horizontal(paym).alias("v4_paym_mean"),
        pl.max_horizontal(paym).alias("v4_paym_max"),
        pl.col("enc_paym_0").alias("v4_paym_latest"),
        pl.sum_horizontal([(pl.col(col) >= 3).cast(pl.UInt8) for col in paym]).alias("v4_paym_ge3_count"),
    ]


def aggregation_expressions() -> list[pl.Expr]:
    expressions: list[pl.Expr] = []

    for col, values in BIN_COLUMNS.items():
        for value in values:
            indicator = (pl.col(col) == value).cast(pl.Float32)
            expressions.extend(
                [
                    indicator.mean().alias(f"v4_{col}_bin_{value}_share"),
                    (pl.col(col).last() == value).cast(pl.UInt8).alias(f"v4_{col}_last_bin_{value}"),
                ]
            )

    for status in range(7):
        status_mask = pl.col("enc_loans_credit_status") == status
        expressions.append(status_mask.cast(pl.Float32).mean().alias(f"v4_status_{status}_share"))
        for col in STATUS_NUMERIC_COLUMNS:
            conditioned = pl.when(status_mask).then(pl.col(col))
            expressions.extend(
                [
                    conditioned.mean().cast(pl.Float32).alias(f"v4_status_{status}_{col}_mean"),
                    conditioned.max().alias(f"v4_status_{status}_{col}_max"),
                ]
            )
        expressions.extend(
            [
                pl.when(status_mask).then(pl.col("v4_paym_mean")).mean().cast(pl.Float32).alias(
                    f"v4_status_{status}_paym_mean"
                ),
                pl.when(status_mask).then(pl.col("v4_paym_max")).max().alias(f"v4_status_{status}_paym_max"),
                pl.when(status_mask).then(pl.col("v4_paym_ge3_count")).mean().cast(pl.Float32).alias(
                    f"v4_status_{status}_paym_ge3_mean"
                ),
            ]
        )

    risky_product = (
        (pl.col("pre_loans_total_overdue") > 0)
        | (pl.col("pre_loans90") != 2)
        | (pl.col("pre_over2limit") > 0)
        | (pl.col("v4_paym_ge3_count") > 0)
    )
    high_util_bad_paym = (pl.col("pre_util") >= 10) & (pl.col("v4_paym_ge3_count") > 0)
    overdue_high_util = (pl.col("pre_loans_total_overdue") > 0) & (pl.col("pre_util") >= 10)

    expressions.extend(
        [
            risky_product.cast(pl.Float32).mean().alias("v4_risky_product_share"),
            risky_product.cast(pl.UInt8).sum().alias("v4_risky_product_count"),
            high_util_bad_paym.cast(pl.Float32).mean().alias("v4_high_util_bad_paym_share"),
            overdue_high_util.cast(pl.Float32).mean().alias("v4_overdue_high_util_share"),
            (pl.col("v4_paym_ge3_count") / (pl.col("pre_loans_credit_limit") + 1))
            .mean()
            .cast(pl.Float32)
            .alias("v4_bad_paym_per_limit_bin_mean"),
            (pl.col("pre_loans_outstanding") / (pl.col("pre_loans_credit_limit") + 1))
            .mean()
            .cast(pl.Float32)
            .alias("v4_outstanding_per_limit_bin_mean"),
            (pl.col("pre_over2limit") - pl.col("pre_util"))
            .mean()
            .cast(pl.Float32)
            .alias("v4_overlimit_minus_util_mean"),
            (pl.col("v4_paym_latest") - pl.col("v4_paym_mean"))
            .mean()
            .cast(pl.Float32)
            .alias("v4_latest_paym_minus_history_mean"),
        ]
    )
    return expressions


def build_v4_additions(source: Path, output: Path, force: bool, max_rows: int | None) -> None:
    if output.exists() and not force:
        print(f"Use cached V4 additions: {output}")
        return

    paym = [f"enc_paym_{index}" for index in range(25)]
    selected = [
        "id",
        "rn",
        "enc_loans_credit_status",
        *BIN_COLUMNS.keys(),
        *paym,
    ]
    print(f"Build compact V4 additions with Polars: {source}")
    source_lazy = pl.scan_parquet(source).select(selected)
    if max_rows is not None:
        source_lazy = source_lazy.head(max_rows)
    additions = (
        source_lazy
        .with_columns(payment_expressions())
        .sort(["id", "rn"])
        .group_by("id", maintain_order=True)
        .agg(aggregation_expressions())
    )
    additions.sink_parquet(output, compression="zstd")
    print(f"Saved V4 additions: {output}")


def load_features(base_path: Path, additions_path: Path, smoke: bool) -> pd.DataFrame:
    print(f"Load base features: {base_path}")
    base = pd.read_parquet(base_path)
    print(f"Load V4 additions: {additions_path}")
    additions = pd.read_parquet(additions_path)
    result = base.merge(additions, on="id", how="inner" if smoke else "left", validate="one_to_one")
    del base, additions
    gc.collect()
    print(f"Combined feature shape: {result.shape}")
    return result


def time_splits(size: int) -> list[tuple[str, slice, slice]]:
    boundaries = {
        "fold_60_70": (0.60, 0.70),
        "fold_70_80": (0.70, 0.80),
        "fold_80_100": (0.80, 1.00),
    }
    return [
        (name, slice(0, int(size * start)), slice(int(size * start), int(size * end)))
        for name, (start, end) in boundaries.items()
    ]


def run_time_cv(features: pd.DataFrame, target: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    data = features.merge(target, on="id", how="inner").sort_values("id").reset_index(drop=True)
    feature_cols = [col for col in data.columns if col not in {"id", "flag"}]
    rows: list[dict[str, object]] = []

    for name, train_slice, valid_slice in time_splits(len(data)):
        print(f"\nTime fold: {name}")
        train_part = data.iloc[train_slice]
        valid_part = data.iloc[valid_slice]
        model = fit_model(
            train_part[feature_cols],
            train_part["flag"],
            valid_part[feature_cols],
            valid_part["flag"],
            args,
            True,
        )
        prediction = model.predict_proba(valid_part[feature_cols])[:, 1]
        rows.append(
            {
                "fold": name,
                "train_rows": len(train_part),
                "valid_rows": len(valid_part),
                "valid_id_min": int(valid_part["id"].min()),
                "valid_id_max": int(valid_part["id"].max()),
                "auc": roc_auc_score(valid_part["flag"], prediction),
                "best_iteration": model.get_best_iteration(),
            }
        )
        pd.DataFrame(rows).to_csv(args.cv_output, index=False)
        del model, prediction, train_part, valid_part
        gc.collect()

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    print(f"Mean time CV AUC: {results['auc'].mean():.6f}")
    print(f"Saved CV results: {args.cv_output}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4 bin histograms, status-conditioned features and time CV.")
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--train-target", type=Path, default=TRAIN_TARGET)
    parser.add_argument("--iterations", type=int, default=2200)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--bagging-temperature", type=float, default=0.2)
    parser.add_argument("--border-count", type=int, default=128)
    parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--cv-output", type=Path, default=ROOT / "catboost_v4_time_cv.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_suffix = f"_{args.max_rows}" if args.max_rows is not None else ""
    additions_path = FEATURE_DIR / f"train_features_v4_additions{cache_suffix}.parquet"
    build_v4_additions(args.train_data, additions_path, args.force_features, args.max_rows)
    features = load_features(FEATURE_DIR / "train_features_v2.parquet", additions_path, args.max_rows is not None)
    target = pd.read_csv(args.train_target)
    run_time_cv(features, target, args)


if __name__ == "__main__":
    main()

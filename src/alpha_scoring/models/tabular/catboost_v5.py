from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import polars as pl

from alpha_scoring.models.tabular.catboost_baseline import FEATURE_DIR, ROOT, TRAIN_DATA, TRAIN_TARGET
from alpha_scoring.models.tabular.catboost_v4 import load_features, run_time_cv


PAYMENT_MONTHS = range(25)
SHIFTED_PAYMENT_MONTHS = {11, 20, 24}
RISK_STATUSES = (4, 5)
RISK_CREDIT_TYPES = (5,)


def normalized_payment_expressions() -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    for month in PAYMENT_MONTHS:
        source = pl.col(f"enc_paym_{month}")
        normalized = source - 1 if month in SHIFTED_PAYMENT_MONTHS else source
        expressions.append(normalized.cast(pl.UInt8).alias(f"v5_paym_{month}"))
    return expressions


def product_payment_expressions() -> list[pl.Expr]:
    columns = [pl.col(f"v5_paym_{month}") for month in PAYMENT_MONTHS]
    expressions: list[pl.Expr] = []

    for code in range(4):
        indicators = [(column == code).cast(pl.UInt8) for column in columns]
        expressions.extend(
            [
                pl.sum_horizontal(indicators).alias(f"v5_paym_code_{code}_count"),
                (pl.sum_horizontal(indicators) / len(columns)).cast(pl.Float32).alias(
                    f"v5_paym_code_{code}_share"
                ),
                (columns[0] == code).cast(pl.UInt8).alias(f"v5_paym_latest_code_{code}"),
            ]
        )
        for window in (3, 6, 12):
            recent = [(column == code).cast(pl.UInt8) for column in columns[:window]]
            expressions.append(
                pl.sum_horizontal(recent).alias(f"v5_paym_recent_{window}_code_{code}_count")
            )

    chronological = list(reversed(columns))
    for source_code in range(4):
        for target_code in range(4):
            transitions = [
                ((chronological[index] == source_code) & (chronological[index + 1] == target_code)).cast(pl.UInt8)
                for index in range(len(chronological) - 1)
            ]
            expressions.append(
                pl.sum_horizontal(transitions).alias(f"v5_transition_{source_code}_to_{target_code}_count")
            )

    code2_recency = [
        pl.when(columns[month] == 2).then(pl.lit(month)).otherwise(pl.lit(25))
        for month in PAYMENT_MONTHS
    ]
    current_code2_prefix = [
        pl.all_horizontal([(columns[index] == 2) for index in range(length)]).cast(pl.UInt8)
        for length in range(1, 26)
    ]
    expressions.extend(
        [
            pl.min_horizontal(code2_recency).cast(pl.UInt8).alias("v5_code2_recency"),
            pl.sum_horizontal(current_code2_prefix).cast(pl.UInt8).alias("v5_code2_current_run"),
            pl.sum_horizontal(
                [
                    ((chronological[index] != 2) & (chronological[index + 1] == 2)).cast(pl.UInt8)
                    for index in range(len(chronological) - 1)
                ]
            ).alias("v5_code2_entry_count"),
            pl.sum_horizontal(
                [
                    ((chronological[index] == 2) & (chronological[index + 1] != 2)).cast(pl.UInt8)
                    for index in range(len(chronological) - 1)
                ]
            ).alias("v5_code2_exit_count"),
        ]
    )
    return expressions


def client_aggregation_expressions() -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    row_features = [
        *[f"v5_paym_code_{code}_count" for code in range(4)],
        *[f"v5_paym_code_{code}_share" for code in range(4)],
        *[
            f"v5_paym_recent_{window}_code_{code}_count"
            for window in (3, 6, 12)
            for code in range(4)
        ],
        *[
            f"v5_transition_{source}_to_{target}_count"
            for source in range(4)
            for target in range(4)
        ],
        "v5_code2_recency",
        "v5_code2_current_run",
        "v5_code2_entry_count",
        "v5_code2_exit_count",
    ]

    for col in row_features:
        expressions.extend(
            [
                pl.col(col).mean().cast(pl.Float32).alias(f"{col}_mean"),
                pl.col(col).max().alias(f"{col}_max"),
            ]
        )

    for code in range(4):
        expressions.extend(
            [
                (pl.col(f"v5_paym_code_{code}_count") > 0)
                .cast(pl.Float32)
                .mean()
                .alias(f"v5_product_share_with_code_{code}"),
                pl.col(f"v5_paym_latest_code_{code}").last().alias(f"v5_last_product_latest_code_{code}"),
            ]
        )

    risky_status = pl.col("enc_loans_credit_status").is_in(RISK_STATUSES)
    risky_type = pl.col("enc_loans_credit_type").is_in(RISK_CREDIT_TYPES)
    has_code2 = pl.col("v5_paym_code_2_count") > 0
    expressions.extend(
        [
            risky_status.cast(pl.Float32).mean().alias("v5_risky_status_share"),
            risky_status.cast(pl.UInt8).sum().alias("v5_risky_status_count"),
            risky_status.last().cast(pl.UInt8).alias("v5_last_product_risky_status"),
            risky_type.cast(pl.Float32).mean().alias("v5_risky_credit_type_share"),
            risky_type.cast(pl.UInt8).sum().alias("v5_risky_credit_type_count"),
            risky_type.last().cast(pl.UInt8).alias("v5_last_product_risky_credit_type"),
            (risky_status & risky_type).cast(pl.Float32).mean().alias("v5_risky_status_type_share"),
            (risky_status & has_code2).cast(pl.Float32).mean().alias("v5_risky_status_code2_share"),
            (risky_type & has_code2).cast(pl.Float32).mean().alias("v5_risky_type_code2_share"),
            (risky_status & (pl.col("pre_loans_total_overdue") > 0))
            .cast(pl.Float32)
            .mean()
            .alias("v5_risky_status_overdue_share"),
            (risky_status & (pl.col("pre_util") >= 10))
            .cast(pl.Float32)
            .mean()
            .alias("v5_risky_status_high_util_share"),
        ]
    )

    for status in RISK_STATUSES:
        mask = pl.col("enc_loans_credit_status") == status
        expressions.extend(
            [
                mask.cast(pl.Float32).mean().alias(f"v5_status_{status}_share"),
                mask.last().cast(pl.UInt8).alias(f"v5_last_status_{status}"),
                pl.when(mask).then(pl.col("v5_paym_code_2_count")).mean().cast(pl.Float32).alias(
                    f"v5_status_{status}_code2_count_mean"
                ),
            ]
        )

    type5 = pl.col("enc_loans_credit_type") == 5
    expressions.extend(
        [
            type5.cast(pl.Float32).mean().alias("v5_credit_type_5_share"),
            type5.last().cast(pl.UInt8).alias("v5_last_credit_type_5"),
            pl.when(type5).then(pl.col("v5_paym_code_2_count")).mean().cast(pl.Float32).alias(
                "v5_credit_type_5_code2_count_mean"
            ),
        ]
    )
    return expressions


def build_v5_additions(source: Path, output: Path, force: bool, max_rows: int | None) -> None:
    if output.exists() and not force:
        print(f"Use cached V5 additions: {output}")
        return

    payment_cols = [f"enc_paym_{month}" for month in PAYMENT_MONTHS]
    selected = [
        "id",
        "rn",
        "enc_loans_credit_status",
        "enc_loans_credit_type",
        "pre_loans_total_overdue",
        "pre_util",
        *payment_cols,
    ]
    source_lazy = pl.scan_parquet(source).select(selected)
    if max_rows is not None:
        source_lazy = source_lazy.head(max_rows)

    print(f"Build normalized encoded features with Polars: {source}")
    additions = (
        source_lazy
        .with_columns(normalized_payment_expressions())
        .with_columns(product_payment_expressions())
        .sort(["id", "rn"])
        .group_by("id", maintain_order=True)
        .agg(client_aggregation_expressions())
    )
    additions.sink_parquet(output, compression="zstd")
    print(f"Saved V5 additions: {output}")


def load_v5_features(v4_path: Path, v5_path: Path, smoke: bool) -> pd.DataFrame:
    features = load_features(FEATURE_DIR / "train_features_v2.parquet", v4_path, smoke)
    print(f"Load V5 additions: {v5_path}")
    additions = pd.read_parquet(v5_path)
    features = features.merge(additions, on="id", how="inner" if smoke else "left", validate="one_to_one")
    del additions
    gc.collect()
    print(f"V5 combined feature shape: {features.shape}")
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5 normalized encoded features and time CV.")
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--train-target", type=Path, default=TRAIN_TARGET)
    parser.add_argument("--iterations", type=int, default=2300)
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
    parser.add_argument("--cv-output", type=Path, default=ROOT / "catboost_v5_time_cv.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_suffix = f"_{args.max_rows}" if args.max_rows is not None else ""
    v4_path = FEATURE_DIR / f"train_features_v4_additions{cache_suffix}.parquet"
    if not v4_path.exists():
        raise FileNotFoundError(f"Build V4 additions first: {v4_path}")
    v5_path = FEATURE_DIR / f"train_features_v5_additions{cache_suffix}.parquet"
    build_v5_additions(args.train_data, v5_path, args.force_features, args.max_rows)
    features = load_v5_features(v4_path, v5_path, args.max_rows is not None)
    target = pd.read_csv(args.train_target)
    run_time_cv(features, target, args)


if __name__ == "__main__":
    main()

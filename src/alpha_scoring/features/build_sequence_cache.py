from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq


from alpha_scoring.paths import PROJECT_ROOT as ROOT
PAYMENT_COLUMNS = [f"enc_paym_{index}" for index in range(25)]
PRODUCT_COLUMNS = [
    "pre_since_opened",
    "pre_since_confirmed",
    "pre_pterm",
    "pre_fterm",
    "pre_till_pclose",
    "pre_till_fclose",
    "pre_loans_credit_limit",
    "pre_loans_next_pay_summ",
    "pre_loans_outstanding",
    "pre_loans_total_overdue",
    "pre_loans_max_overdue_sum",
    "pre_loans_credit_cost_rate",
    "pre_loans5",
    "pre_loans530",
    "pre_loans3060",
    "pre_loans6090",
    "pre_loans90",
    "is_zero_loans5",
    "is_zero_loans530",
    "is_zero_loans3060",
    "is_zero_loans6090",
    "is_zero_loans90",
    "pre_util",
    "pre_over2limit",
    "pre_maxover2limit",
    "is_zero_util",
    "is_zero_over2limit",
    "is_zero_maxover2limit",
    "enc_loans_account_holder_type",
    "enc_loans_credit_status",
    "enc_loans_credit_type",
    "enc_loans_account_cur",
    "pclose_flag",
    "fclose_flag",
]
SHIFTED_PAYMENT_COLUMNS = {"enc_paym_11", "enc_paym_20", "enc_paym_24"}


def sorted_lazy(source: Path) -> pl.LazyFrame:
    expressions: list[pl.Expr] = [pl.col("id").cast(pl.Int32), pl.col("rn").cast(pl.UInt8)]
    expressions.extend([(pl.col(col) + 1).cast(pl.UInt8).alias(col) for col in PRODUCT_COLUMNS])
    for col in PAYMENT_COLUMNS:
        normalized = pl.col(col) - 1 if col in SHIFTED_PAYMENT_COLUMNS else pl.col(col)
        expressions.append((normalized + 1).cast(pl.UInt8).alias(col))
    return pl.scan_parquet(source).select(expressions).sort(["id", "rn"])


def build_split(source: Path, output_dir: Path, split: str, keep_sorted: bool) -> dict[str, object]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    sorted_path = split_dir / "sorted_products.parquet"
    print(f"Sort {source} into isolated cache: {sorted_path}")
    sorted_lazy(source).sink_parquet(sorted_path, compression="zstd")

    parquet = pq.ParquetFile(sorted_path)
    row_count = parquet.metadata.num_rows
    feature_columns = [*PRODUCT_COLUMNS, *PAYMENT_COLUMNS]
    values = np.lib.format.open_memmap(
        split_dir / "values.npy", mode="w+", dtype=np.uint8, shape=(row_count, len(feature_columns))
    )

    ids: list[int] = []
    offsets: list[int] = []
    row_position = 0
    previous_id: int | None = None
    maxima = np.zeros(len(feature_columns), dtype=np.uint16)

    print(f"Stream {row_count} product rows into values.npy")
    for batch in parquet.iter_batches(batch_size=262_144, columns=["id", *feature_columns]):
        frame = batch.to_pandas()
        batch_ids = frame.pop("id").to_numpy(dtype=np.int32, copy=False)
        batch_values = frame.to_numpy(dtype=np.uint8, copy=False)
        end = row_position + len(frame)
        values[row_position:end] = batch_values
        maxima = np.maximum(maxima, batch_values.max(axis=0))

        starts = np.flatnonzero(np.r_[True, batch_ids[1:] != batch_ids[:-1]])
        for local_start in starts:
            client_id = int(batch_ids[local_start])
            if previous_id is None or client_id != previous_id:
                ids.append(client_id)
                offsets.append(row_position + int(local_start))
                previous_id = client_id
        row_position = end

    values.flush()
    offsets.append(row_count)
    client_ids = np.asarray(ids, dtype=np.int32)
    client_offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(client_offsets)
    if lengths.max(initial=0) > 255:
        raise RuntimeError("Sequence length does not fit uint8")
    if not np.all(client_ids[1:] > client_ids[:-1]):
        raise RuntimeError("Client ids are not strictly sorted")

    np.save(split_dir / "ids.npy", client_ids)
    np.save(split_dir / "offsets.npy", client_offsets)
    np.save(split_dir / "lengths.npy", lengths.astype(np.uint8))
    if not keep_sorted:
        sorted_path.unlink()

    metadata = {
        "split": split,
        "source": source.name,
        "rows": row_count,
        "clients": len(client_ids),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "feature_columns": feature_columns,
        "product_columns": PRODUCT_COLUMNS,
        "payment_columns": PAYMENT_COLUMNS,
        "cardinalities": {col: int(maxima[index]) + 1 for index, col in enumerate(feature_columns)},
        "padding_code": 0,
        "value_shift": 1,
    }
    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {split}: clients={len(client_ids)}, rows={row_count}, max_length={lengths.max()}")
    return metadata


def attach_target(output_dir: Path, target_path: Path) -> None:
    ids = np.load(output_dir / "train" / "ids.npy")
    target = pd.read_csv(target_path).set_index("id")["flag"]
    aligned = target.reindex(ids)
    if aligned.isna().any():
        raise RuntimeError(f"Missing target for {int(aligned.isna().sum())} clients")
    np.save(output_dir / "train" / "target.npy", aligned.to_numpy(dtype=np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable sequence cache from source parquet files.")
    parser.add_argument("--train-data", type=Path, default=ROOT / "train_data.parquet")
    parser.add_argument("--test-data", type=Path, default=ROOT / "test_data.parquet")
    parser.add_argument("--train-target", type=Path, default=ROOT / "train_target.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sequence_cache")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-sorted", action="store_true")
    args = parser.parse_args()

    source_paths = {args.train_data.resolve(), args.test_data.resolve(), args.train_target.resolve()}
    if args.output_dir.resolve() in source_paths:
        raise RuntimeError("Output directory must not overlap source files")
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}. Use --force to rebuild.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_meta = build_split(args.train_data, args.output_dir, "train", args.keep_sorted)
    test_meta = build_split(args.test_data, args.output_dir, "test", args.keep_sorted)
    attach_target(args.output_dir, args.train_target)
    manifest = {"version": 1, "train": train_meta, "test": test_meta}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Sequence cache ready: {args.output_dir}")


if __name__ == "__main__":
    main()

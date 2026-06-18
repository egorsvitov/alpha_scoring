from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq


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


def sorted_products(source: Path) -> pl.LazyFrame:
    expressions: list[pl.Expr] = [pl.col("id").cast(pl.Int32), pl.col("rn").cast(pl.UInt8)]
    expressions.extend([(pl.col(col) + 1).cast(pl.UInt8).alias(col) for col in PRODUCT_COLUMNS])
    for col in PAYMENT_COLUMNS:
        normalized = pl.col(col) - 1 if col in SHIFTED_PAYMENT_COLUMNS else pl.col(col)
        expressions.append((normalized + 1).cast(pl.UInt8).alias(col))
    return pl.scan_parquet(source).select(expressions).sort(["id", "rn"])


def build_split(source: Path, output_dir: Path, split: str) -> dict[str, object]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    sorted_path = split_dir / "sorted_products.parquet"
    sorted_products(source).sink_parquet(sorted_path, compression="zstd")

    parquet = pq.ParquetFile(sorted_path)
    row_count = parquet.metadata.num_rows
    feature_columns = [*PRODUCT_COLUMNS, *PAYMENT_COLUMNS]
    values = np.lib.format.open_memmap(
        split_dir / "values.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(row_count, len(feature_columns)),
    )

    ids: list[int] = []
    offsets: list[int] = []
    row_position = 0
    previous_id: int | None = None
    maxima = np.zeros(len(feature_columns), dtype=np.uint16)

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
    sorted_path.unlink()
    offsets.append(row_count)
    client_ids = np.asarray(ids, dtype=np.int32)
    client_offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(client_offsets)
    if not np.all(client_ids[1:] > client_ids[:-1]):
        raise RuntimeError(f"{split} ids are not strictly sorted")

    np.save(split_dir / "ids.npy", client_ids)
    np.save(split_dir / "offsets.npy", client_offsets)
    np.save(split_dir / "lengths.npy", lengths.astype(np.uint8))
    metadata = {
        "split": split,
        "rows": row_count,
        "clients": len(client_ids),
        "max_length": int(lengths.max()),
        "feature_columns": feature_columns,
        "product_columns": PRODUCT_COLUMNS,
        "payment_columns": PAYMENT_COLUMNS,
        "cardinalities": {col: int(maxima[index]) + 1 for index, col in enumerate(feature_columns)},
        "padding_code": 0,
        "value_shift": 1,
    }
    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{split}: clients={len(client_ids)} rows={row_count} max_length={lengths.max()}", flush=True)
    return metadata


def attach_target(output_dir: Path, target_path: Path) -> None:
    ids = np.load(output_dir / "train" / "ids.npy")
    target = pd.read_csv(target_path).set_index("id")["flag"]
    aligned = target.reindex(ids)
    if aligned.isna().any():
        raise RuntimeError(f"Missing target for {int(aligned.isna().sum())} clients")
    np.save(output_dir / "train" / "target.npy", aligned.to_numpy(dtype=np.uint8))


def build_cache(train_data: Path, test_data: Path, train_target: Path, output_dir: Path, force: bool) -> None:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Cache exists: {output_dir}. Use --force to rebuild.")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_meta = build_split(train_data, output_dir, "train")
    test_meta = build_split(test_data, output_dir, "test")
    attach_target(output_dir, train_target)
    manifest = {"version": 1, "train": train_meta, "test": test_meta}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default=Path("data/train_data.parquet"))
    parser.add_argument("--test-data", type=Path, default=Path("data/test_data.parquet"))
    parser.add_argument("--train-target", type=Path, default=Path("data/train_target.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("sequence_cache"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_cache(args.train_data, args.test_data, args.train_target, args.output_dir, args.force)


if __name__ == "__main__":
    main()

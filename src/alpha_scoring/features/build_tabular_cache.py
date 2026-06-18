from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from catboost import CatBoostClassifier


from alpha_scoring.paths import PROJECT_ROOT as ROOT
FEATURE_FILES = {
    "base": ("train_features_v2.parquet", "test_features_v2.parquet"),
    "v4": ("train_features_v4_additions.parquet", "test_features_v4_additions.parquet"),
    "v5": ("train_features_v5_additions.parquet", "test_features_v5_additions.parquet"),
}


def select_features(model_path: Path, feature_dir: Path, count: int) -> tuple[list[str], dict[str, str]]:
    model = CatBoostClassifier()
    model.load_model(model_path)
    importance = model.get_feature_importance()
    ranked = [model.feature_names_[index] for index in np.argsort(importance)[::-1]]

    ownership: dict[str, str] = {}
    for group, (train_name, _) in FEATURE_FILES.items():
        schema = pq.read_schema(feature_dir / train_name)
        for name in schema.names:
            if name != "id":
                if name in ownership:
                    raise RuntimeError(f"Duplicate feature across caches: {name}")
                ownership[name] = group
    missing = [name for name in ranked if name not in ownership]
    if missing:
        raise RuntimeError(f"Model features missing from parquet caches: {missing[:5]}")
    return ranked[:count], ownership


def combined_lazy(feature_dir: Path, split: str, selected: list[str], ownership: dict[str, str]) -> pl.LazyFrame:
    file_index = 0 if split == "train" else 1
    groups: dict[str, list[str]] = {group: [] for group in FEATURE_FILES}
    for column in selected:
        groups[ownership[column]].append(column)

    result: pl.LazyFrame | None = None
    for group, names in groups.items():
        if not names:
            continue
        path = feature_dir / FEATURE_FILES[group][file_index]
        frame = pl.scan_parquet(path).select([pl.col("id").cast(pl.Int32), *[pl.col(name).cast(pl.Float32) for name in names]])
        result = frame if result is None else result.join(frame, on="id", how="inner")
    if result is None:
        raise RuntimeError("No selected features")
    return result.select(["id", *selected]).sort("id")


def calculate_scaler(sorted_train: Path, selected: list[str]) -> tuple[np.ndarray, np.ndarray]:
    expressions = []
    for name in selected:
        clean = pl.col(name).replace([float("inf"), float("-inf")], None)
        expressions.extend(
            [
                clean.median().alias(f"{name}__median"),
                clean.quantile(0.25).alias(f"{name}__q25"),
                clean.quantile(0.75).alias(f"{name}__q75"),
            ]
        )
    stats = pl.scan_parquet(sorted_train).select(expressions).collect()
    median = np.asarray([stats[0, f"{name}__median"] for name in selected], dtype=np.float32)
    q25 = np.asarray([stats[0, f"{name}__q25"] for name in selected], dtype=np.float32)
    q75 = np.asarray([stats[0, f"{name}__q75"] for name in selected], dtype=np.float32)
    median = np.nan_to_num(median, nan=0.0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return median, scale


def write_cache(sorted_path: Path, output_dir: Path, ids_path: Path, selected: list[str], median: np.ndarray, scale: np.ndarray) -> None:
    expected_ids = np.load(ids_path, mmap_mode="r")
    parquet = pq.ParquetFile(sorted_path)
    if parquet.metadata.num_rows != len(expected_ids):
        raise RuntimeError(f"Row count mismatch: {parquet.metadata.num_rows} != {len(expected_ids)}")
    values = np.lib.format.open_memmap(
        output_dir / "values.npy", mode="w+", dtype=np.float16, shape=(len(expected_ids), len(selected))
    )
    position = 0
    for batch in parquet.iter_batches(batch_size=131_072, columns=["id", *selected]):
        frame = batch.to_pandas()
        batch_ids = frame.pop("id").to_numpy(dtype=np.int32, copy=False)
        end = position + len(frame)
        if not np.array_equal(batch_ids, expected_ids[position:end]):
            raise RuntimeError(f"ID alignment failed at row {position}")
        matrix = frame.to_numpy(dtype=np.float32, copy=False)
        matrix[~np.isfinite(matrix)] = np.nan
        missing = np.isnan(matrix)
        if missing.any():
            matrix[missing] = np.broadcast_to(median, matrix.shape)[missing]
        matrix = np.clip((matrix - median) / scale, -10.0, 10.0)
        values[position:end] = matrix.astype(np.float16)
        position = end
    values.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact aligned V5 aggregate cache for dual-branch models.")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "features")
    parser.add_argument("--sequence-cache-dir", type=Path, default=ROOT / "sequence_cache")
    parser.add_argument("--model", type=Path, default=ROOT / "catboost_v5.cbm")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tabular_cache")
    parser.add_argument("--feature-count", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}. Use --force to rebuild.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected, ownership = select_features(args.model, args.feature_dir, args.feature_count)
    temporary: dict[str, Path] = {}
    for split in ("train", "test"):
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        temporary[split] = split_dir / "sorted_features.parquet"
        print(f"Join and sort {split} aggregate features", flush=True)
        combined_lazy(args.feature_dir, split, selected, ownership).sink_parquet(temporary[split], compression="zstd")

    print("Calculate train robust scaler", flush=True)
    median, scale = calculate_scaler(temporary["train"], selected)
    np.save(args.output_dir / "median.npy", median)
    np.save(args.output_dir / "scale.npy", scale)
    for split in ("train", "test"):
        print(f"Write aligned float16 cache: {split}", flush=True)
        write_cache(
            temporary[split],
            args.output_dir / split,
            args.sequence_cache_dir / split / "ids.npy",
            selected,
            median,
            scale,
        )
        temporary[split].unlink()

    metadata = {"feature_count": len(selected), "features": selected, "clip": 10.0, "dtype": "float16"}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Tabular cache ready: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

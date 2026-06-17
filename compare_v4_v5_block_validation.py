from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from catboost_baseline import FEATURE_DIR, ROOT, TRAIN_TARGET
from catboost_v3 import fit_model


def load_variant(addition_paths: list[Path]) -> pd.DataFrame:
    print(f"Load base: {FEATURE_DIR / 'train_features_v2.parquet'}")
    result = pd.read_parquet(FEATURE_DIR / "train_features_v2.parquet")
    for path in addition_paths:
        print(f"Join additions: {path}")
        additions = pd.read_parquet(path)
        result = result.merge(additions, on="id", how="left", validate="one_to_one")
        del additions
        gc.collect()
    print(f"Variant shape: {result.shape}")
    return result


def assign_blocks(ids: pd.Series, block_count: int) -> np.ndarray:
    id_min = int(ids.min())
    id_max = int(ids.max())
    width = id_max - id_min + 1
    blocks = ((ids.to_numpy(dtype=np.int64) - id_min) * block_count // width).astype(np.int16)
    return np.minimum(blocks, block_count - 1)


def validation_mask(ids: pd.Series, fraction: float, seed: int) -> np.ndarray:
    denominator = 10_000
    threshold = int(fraction * denominator)
    values = ids.to_numpy(dtype=np.uint64)
    # Stable integer hash: deterministic across Python and pandas versions.
    hashed = values * np.uint64(11400714819323198485) + np.uint64(seed)
    return (hashed % np.uint64(denominator)) < np.uint64(threshold)


def evaluate_variant(
    name: str,
    features: pd.DataFrame,
    target: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    data = features.merge(target, on="id", how="inner").sort_values("id").reset_index(drop=True)
    blocks = assign_blocks(data["id"], args.blocks)
    valid_mask = validation_mask(data["id"], args.valid_fraction, args.split_seed)
    feature_cols = [col for col in data.columns if col not in {"id", "flag"}]

    train_part = data.loc[~valid_mask]
    valid_part = data.loc[valid_mask]
    print(f"\n{name}: train={len(train_part)}, valid={len(valid_part)}, features={len(feature_cols)}")

    model = fit_model(
        train_part[feature_cols],
        train_part["flag"],
        valid_part[feature_cols],
        valid_part["flag"],
        args,
        True,
    )
    prediction = model.predict_proba(valid_part[feature_cols])[:, 1]
    overall_auc = roc_auc_score(valid_part["flag"], prediction)

    valid_blocks = blocks[valid_mask]
    block_rows: list[dict[str, object]] = []
    for block in range(args.blocks):
        mask = valid_blocks == block
        block_target = valid_part["flag"].to_numpy()[mask]
        block_prediction = prediction[mask]
        block_rows.append(
            {
                "variant": name,
                "block": block,
                "valid_rows": int(mask.sum()),
                "id_min": int(valid_part["id"].to_numpy()[mask].min()),
                "id_max": int(valid_part["id"].to_numpy()[mask].max()),
                "default_rate": float(block_target.mean()),
                "auc": roc_auc_score(block_target, block_prediction),
            }
        )

    summary = {
        "variant": name,
        "train_rows": len(train_part),
        "valid_rows": len(valid_part),
        "feature_count": len(feature_cols),
        "auc": overall_auc,
        "block_auc_mean": float(np.mean([row["auc"] for row in block_rows])),
        "block_auc_std": float(np.std([row["auc"] for row in block_rows])),
        "best_iteration": model.get_best_iteration(),
    }
    oof = pd.DataFrame(
        {
            "id": valid_part["id"].to_numpy(),
            "flag": valid_part["flag"].to_numpy(),
            "block": valid_blocks,
            f"pred_{name}": prediction,
        }
    )
    print(pd.DataFrame([summary]).to_string(index=False))
    print(pd.DataFrame(block_rows).to_string(index=False))

    del model, prediction, train_part, valid_part, data
    gc.collect()
    return summary, pd.DataFrame(block_rows), oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare V4 and V5 on time-stratified block validation.")
    parser.add_argument("--train-target", type=Path, default=TRAIN_TARGET)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=20260607)
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
    parser.add_argument("--summary-output", type=Path, default=ROOT / "v4_v5_block_validation.csv")
    parser.add_argument("--blocks-output", type=Path, default=ROOT / "v4_v5_block_auc.csv")
    parser.add_argument("--oof-output", type=Path, default=ROOT / "v4_v5_block_oof.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = pd.read_csv(args.train_target)
    variants = {
        "v4": [FEATURE_DIR / "train_features_v4_additions.parquet"],
        "v5": [
            FEATURE_DIR / "train_features_v4_additions.parquet",
            FEATURE_DIR / "train_features_v5_additions.parquet",
        ],
    }

    summaries: list[dict[str, object]] = []
    block_results: list[pd.DataFrame] = []
    oof_result: pd.DataFrame | None = None
    for name, paths in variants.items():
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
        features = load_variant(paths)
        summary, blocks, oof = evaluate_variant(name, features, target, args)
        summaries.append(summary)
        block_results.append(blocks)
        if oof_result is None:
            oof_result = oof
        else:
            prediction_col = f"pred_{name}"
            oof_result = oof_result.merge(oof[["id", prediction_col]], on="id", how="inner", validate="one_to_one")
        pd.DataFrame(summaries).to_csv(args.summary_output, index=False)
        pd.concat(block_results, ignore_index=True).to_csv(args.blocks_output, index=False)
        oof_result.to_parquet(args.oof_output, index=False)
        del features
        gc.collect()

    summary_frame = pd.DataFrame(summaries).sort_values("auc", ascending=False)
    print("\nFinal comparison")
    print(summary_frame.to_string(index=False))
    print(f"Saved: {args.summary_output}")
    print(f"Saved: {args.blocks_output}")
    print(f"Saved: {args.oof_output}")


if __name__ == "__main__":
    main()

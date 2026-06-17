from __future__ import annotations

import argparse
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from alpha_scoring.models.tabular.catboost_baseline import FEATURE_DIR, ROOT, TRAIN_TARGET
from alpha_scoring.models.tabular.compare_v4_v5_block_validation import assign_blocks, validation_mask
from alpha_scoring.models.tabular.catboost_v5 import load_v5_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM on V5 features with block validation.")
    parser.add_argument("--train-target", type=Path, default=TRAIN_TARGET)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=20260607)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=1500)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument("--reg-lambda", type=float, default=10.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--max-bin", type=int, default=63)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--summary-output", type=Path, default=ROOT / "lightgbm_v5_block_validation.csv")
    parser.add_argument("--blocks-output", type=Path, default=ROOT / "lightgbm_v5_block_auc.csv")
    parser.add_argument("--oof-output", type=Path, default=ROOT / "lightgbm_v5_block_oof.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_v5_features(
        FEATURE_DIR / "train_features_v4_additions.parquet",
        FEATURE_DIR / "train_features_v5_additions.parquet",
        smoke=False,
    )
    target = pd.read_csv(args.train_target)
    data = features.merge(target, on="id", how="inner", validate="one_to_one").sort_values("id").reset_index(drop=True)
    del features, target
    gc.collect()

    blocks = assign_blocks(data["id"], args.blocks)
    valid_mask = validation_mask(data["id"], args.valid_fraction, args.split_seed)
    feature_cols = [col for col in data.columns if col not in {"id", "flag"}]
    train_part = data.loc[~valid_mask]
    valid_part = data.loc[valid_mask]
    print(f"LightGBM V5: train={len(train_part)}, valid={len(valid_part)}, features={len(feature_cols)}")

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=1,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        max_bin=args.max_bin,
        n_jobs=args.n_jobs,
        random_state=42,
        class_weight="balanced",
        force_col_wise=True,
        verbosity=1,
    )
    model.fit(
        train_part[feature_cols],
        train_part["flag"],
        eval_set=[(valid_part[feature_cols], valid_part["flag"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds), lgb.log_evaluation(100)],
    )
    prediction = model.predict_proba(valid_part[feature_cols])[:, 1]
    auc = roc_auc_score(valid_part["flag"], prediction)
    valid_blocks = blocks[valid_mask]

    block_rows = []
    for block in range(args.blocks):
        mask = valid_blocks == block
        block_target = valid_part["flag"].to_numpy()[mask]
        block_prediction = prediction[mask]
        block_rows.append(
            {
                "variant": "lightgbm_v5",
                "block": block,
                "valid_rows": int(mask.sum()),
                "default_rate": float(block_target.mean()),
                "auc": roc_auc_score(block_target, block_prediction),
            }
        )
    summary = pd.DataFrame(
        [
            {
                "variant": "lightgbm_v5",
                "train_rows": len(train_part),
                "valid_rows": len(valid_part),
                "feature_count": len(feature_cols),
                "auc": auc,
                "block_auc_mean": float(np.mean([row["auc"] for row in block_rows])),
                "block_auc_std": float(np.std([row["auc"] for row in block_rows])),
                "best_iteration": model.best_iteration_,
            }
        ]
    )
    blocks_frame = pd.DataFrame(block_rows)
    oof = pd.DataFrame(
        {
            "id": valid_part["id"].to_numpy(),
            "flag": valid_part["flag"].to_numpy(),
            "block": valid_blocks,
            "pred_lightgbm_v5": prediction,
        }
    )
    summary.to_csv(args.summary_output, index=False)
    blocks_frame.to_csv(args.blocks_output, index=False)
    oof.to_parquet(args.oof_output, index=False)
    print(summary.to_string(index=False))
    print(blocks_frame.to_string(index=False))
    print(f"Saved: {args.summary_output}")
    print(f"Saved: {args.blocks_output}")
    print(f"Saved: {args.oof_output}")


if __name__ == "__main__":
    main()

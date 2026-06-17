from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from catboost_baseline import FEATURE_DIR, ROOT, SAMPLE_SUBMISSION, TRAIN_TARGET
from catboost_v3 import fit_model
from catboost_v4 import load_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final V4 and create submission.")
    parser.add_argument("--iterations", type=int, default=2180)
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
    parser.add_argument("--output", type=Path, default=ROOT / "submission_catboost_v4.csv")
    parser.add_argument("--model-output", type=Path, default=ROOT / "catboost_v4.cbm")
    args = parser.parse_args()

    train = load_features(
        FEATURE_DIR / "train_features_v2.parquet",
        FEATURE_DIR / "train_features_v4_additions.parquet",
        smoke=False,
    )
    target = pd.read_csv(TRAIN_TARGET)
    train = train.merge(target, on="id", how="inner", validate="one_to_one").sort_values("id").reset_index(drop=True)
    feature_cols = [col for col in train.columns if col not in {"id", "flag"}]
    model = fit_model(train[feature_cols], train["flag"], None, None, args, False)
    model.save_model(args.model_output)
    del train, target
    gc.collect()

    test = load_features(
        FEATURE_DIR / "test_features_v2.parquet",
        FEATURE_DIR / "test_features_v4_additions.parquet",
        smoke=False,
    )
    prediction = model.predict_proba(test[feature_cols])[:, 1]
    pred = pd.DataFrame({"id": test["id"].to_numpy(), "flag": prediction})
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    submission = sample[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
    if submission["flag"].isna().any() or not submission["flag"].between(0, 1).all():
        raise RuntimeError("Invalid V4 predictions")
    submission.to_csv(args.output, index=False)
    print(f"Saved: {args.output}, shape={submission.shape}")


if __name__ == "__main__":
    main()

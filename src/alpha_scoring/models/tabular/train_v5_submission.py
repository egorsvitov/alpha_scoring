from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from alpha_scoring.models.tabular.catboost_baseline import FEATURE_DIR, ROOT, SAMPLE_SUBMISSION, TEST_DATA, TRAIN_TARGET
from alpha_scoring.models.tabular.catboost_v3 import fit_model
from alpha_scoring.models.tabular.catboost_v4 import build_v4_additions, load_features
from alpha_scoring.models.tabular.catboost_v5 import build_v5_additions


def load_matrix(base_path: Path, v4_path: Path, v5_path: Path) -> pd.DataFrame:
    matrix = load_features(base_path, v4_path, smoke=False)
    print(f"Join V5 additions: {v5_path}")
    v5 = pd.read_parquet(v5_path)
    matrix = matrix.merge(v5, on="id", how="left", validate="one_to_one")
    del v5
    gc.collect()
    print(f"V5 matrix shape: {matrix.shape}")
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final V5 and create submission.")
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--train-target", type=Path, default=TRAIN_TARGET)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION)
    parser.add_argument("--iterations", type=int, default=2216)
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
    parser.add_argument("--force-test-features", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "submission_catboost_v5.csv")
    parser.add_argument("--model-output", type=Path, default=ROOT / "catboost_v5.cbm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_v4 = FEATURE_DIR / "train_features_v4_additions.parquet"
    train_v5 = FEATURE_DIR / "train_features_v5_additions.parquet"
    test_v4 = FEATURE_DIR / "test_features_v4_additions.parquet"
    test_v5 = FEATURE_DIR / "test_features_v5_additions.parquet"

    print("Build missing test additions before loading the train matrix")
    build_v4_additions(args.test_data, test_v4, args.force_test_features, max_rows=None)
    build_v5_additions(args.test_data, test_v5, args.force_test_features, max_rows=None)

    train = load_matrix(FEATURE_DIR / "train_features_v2.parquet", train_v4, train_v5)
    target = pd.read_csv(args.train_target)
    train = train.merge(target, on="id", how="inner", validate="one_to_one").sort_values("id").reset_index(drop=True)
    feature_cols = [col for col in train.columns if col not in {"id", "flag"}]
    print(f"Fit final V5: rows={len(train)}, features={len(feature_cols)}, iterations={args.iterations}")
    model = fit_model(train[feature_cols], train["flag"], None, None, args, False)
    model.save_model(args.model_output)
    print(f"Saved model: {args.model_output}")

    del train, target
    gc.collect()

    test = load_matrix(FEATURE_DIR / "test_features_v2.parquet", test_v4, test_v5)
    missing = sorted(set(feature_cols) - set(test.columns))
    if missing:
        raise RuntimeError(f"Test misses {len(missing)} features: {missing[:5]}")
    prediction = model.predict_proba(test[feature_cols])[:, 1]
    test_prediction = pd.DataFrame({"id": test["id"].to_numpy(), "flag": prediction})
    sample = pd.read_csv(args.sample_submission)
    submission = sample[["id"]].merge(test_prediction, on="id", how="left", validate="one_to_one")
    if submission["flag"].isna().any():
        raise RuntimeError(f"Missing predictions: {int(submission['flag'].isna().sum())}")
    if not submission["flag"].between(0, 1).all():
        raise RuntimeError("Predictions are outside [0, 1]")
    submission.to_csv(args.output, index=False)
    print(f"Saved submission: {args.output}, shape={submission.shape}")
    print(submission["flag"].describe().to_string())


if __name__ == "__main__":
    main()

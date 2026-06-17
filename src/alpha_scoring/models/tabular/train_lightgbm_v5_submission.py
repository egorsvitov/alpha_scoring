from __future__ import annotations

import argparse
import gc
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from alpha_scoring.models.tabular.catboost_baseline import FEATURE_DIR, ROOT, SAMPLE_SUBMISSION, TRAIN_TARGET
from alpha_scoring.models.tabular.catboost_v4 import load_features


def load_matrix(base_path: Path, v4_path: Path, v5_path: Path) -> pd.DataFrame:
    matrix = load_features(base_path, v4_path, smoke=False)
    v5 = pd.read_parquet(v5_path)
    matrix = matrix.merge(v5, on="id", how="left", validate="one_to_one")
    del v5
    gc.collect()
    print(f"LightGBM matrix shape: {matrix.shape}")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final LightGBM V5 and create submission.")
    parser.add_argument("--n-estimators", type=int, default=1217)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-child-samples", type=int, default=1500)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument("--reg-lambda", type=float, default=10.0)
    parser.add_argument("--max-bin", type=int, default=63)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "submission_lightgbm_v5.csv")
    parser.add_argument("--model-output", type=Path, default=ROOT / "lightgbm_v5.txt")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    if args.predict_only:
        booster = lgb.Booster(model_file=str(args.model_output))
        feature_cols = booster.feature_name()
    else:
        train = load_matrix(
            FEATURE_DIR / "train_features_v2.parquet",
            FEATURE_DIR / "train_features_v4_additions.parquet",
            FEATURE_DIR / "train_features_v5_additions.parquet",
        )
        target = pd.read_csv(TRAIN_TARGET)
        train = train.merge(target, on="id", how="inner", validate="one_to_one").sort_values("id").reset_index(drop=True)
        feature_cols = [col for col in train.columns if col not in {"id", "flag"}]
        print(f"Fit LightGBM: rows={len(train)}, features={len(feature_cols)}, trees={args.n_estimators}")
        model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            subsample_freq=1,
            colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda,
            max_bin=args.max_bin,
            n_jobs=args.n_jobs,
            random_state=42,
            class_weight="balanced",
            force_col_wise=True,
            verbosity=1,
        )
        model.fit(train[feature_cols], train["flag"], callbacks=[lgb.log_evaluation(100)])
        model.booster_.save_model(args.model_output)
        booster = model.booster_
        del train, target, model
        gc.collect()

    test = load_matrix(
        FEATURE_DIR / "test_features_v2.parquet",
        FEATURE_DIR / "test_features_v4_additions.parquet",
        FEATURE_DIR / "test_features_v5_additions.parquet",
    )
    prediction = booster.predict(test[feature_cols])
    pred = pd.DataFrame({"id": test["id"].to_numpy(), "flag": prediction})
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    submission = sample[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
    if submission["flag"].isna().any() or not submission["flag"].between(0, 1).all():
        raise RuntimeError("Invalid LightGBM predictions")
    submission.to_csv(args.output, index=False)
    print(f"Saved: {args.output}, shape={submission.shape}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from alpha_scoring.models.tabular.catboost_v2 import build_features
from alpha_scoring.models.tabular.catboost_v4 import build_v4_additions
from alpha_scoring.models.tabular.catboost_v5 import build_v5_additions


from alpha_scoring.paths import PROJECT_ROOT as ROOT
FEATURE_DIR = ROOT / "features"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tabular aggregate features from source parquet files.")
    parser.add_argument("--train-data", type=Path, default=ROOT / "train_data.parquet")
    parser.add_argument("--test-data", type=Path, default=ROOT / "test_data.parquet")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if feature parquet files already exist.")
    args = parser.parse_args()

    feature_dir = args.feature_dir
    feature_dir.mkdir(exist_ok=True)
    build_features(args.train_data, feature_dir / "train_features_v2.parquet", force=args.force)
    build_features(args.test_data, feature_dir / "test_features_v2.parquet", force=args.force)
    build_v4_additions(args.train_data, feature_dir / "train_features_v4_additions.parquet", args.force, None)
    build_v4_additions(args.test_data, feature_dir / "test_features_v4_additions.parquet", args.force, None)
    build_v5_additions(args.train_data, feature_dir / "train_features_v5_additions.parquet", args.force, None)
    build_v5_additions(args.test_data, feature_dir / "test_features_v5_additions.parquet", args.force, None)


if __name__ == "__main__":
    main()

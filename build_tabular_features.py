from __future__ import annotations

from pathlib import Path

from catboost_v2 import build_features
from catboost_v4 import build_v4_additions
from catboost_v5 import build_v5_additions


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "features"


def main() -> None:
    FEATURE_DIR.mkdir(exist_ok=True)
    build_features(ROOT / "train_data.parquet", FEATURE_DIR / "train_features_v2.parquet", force=False)
    build_features(ROOT / "test_data.parquet", FEATURE_DIR / "test_features_v2.parquet", force=False)
    build_v4_additions(ROOT / "train_data.parquet", FEATURE_DIR / "train_features_v4_additions.parquet", False, None)
    build_v4_additions(ROOT / "test_data.parquet", FEATURE_DIR / "test_features_v4_additions.parquet", False, None)
    build_v5_additions(ROOT / "train_data.parquet", FEATURE_DIR / "train_features_v5_additions.parquet", False, None)
    build_v5_additions(ROOT / "test_data.parquet", FEATURE_DIR / "test_features_v5_additions.parquet", False, None)


if __name__ == "__main__":
    main()

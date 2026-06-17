from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rank01(values: pd.Series) -> np.ndarray:
    return values.rank(method="average").to_numpy(np.float64) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank-average aligned OOF or submission predictions.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = [pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path) for path in args.inputs]
    prediction_column = "prediction" if "prediction" in frames[0].columns else "flag"
    reference = frames[0]
    for path, frame in zip(args.inputs[1:], frames[1:]):
        if not reference["id"].equals(frame["id"]):
            raise RuntimeError(f"ID order mismatch: {path}")
    prediction = np.mean([rank01(frame[prediction_column]) for frame in frames], axis=0)
    result = reference.copy()
    result[prediction_column] = prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        result.to_parquet(args.output, index=False)
    else:
        result[["id", prediction_column]].to_csv(args.output, index=False)
    message = f"Saved {args.output}: inputs={len(frames)}, rows={len(result)}"
    if prediction_column == "prediction" and "flag" in result:
        message += f", auc={roc_auc_score(result['flag'], prediction):.9f}"
    print(message)


if __name__ == "__main__":
    main()

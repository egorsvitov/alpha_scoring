from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rank01(values: pd.Series) -> np.ndarray:
    return values.rank(method="average").to_numpy(np.float64) / len(values)


def parse_weighted_path(value: str) -> tuple[float, Path]:
    weight, path = value.split(":", maxsplit=1)
    return float(weight), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a weighted rank blend of aligned prediction files.")
    parser.add_argument("inputs", nargs="+", type=parse_weighted_path, metavar="WEIGHT:PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    total_weight = sum(weight for weight, _ in args.inputs)
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"Weights must sum to 1, got {total_weight}")
    frames = [pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path) for _, path in args.inputs]
    prediction_column = "prediction" if "prediction" in frames[0].columns else "flag"
    reference = frames[0]
    for (_, path), frame in zip(args.inputs[1:], frames[1:]):
        if not reference["id"].equals(frame["id"]):
            raise RuntimeError(f"ID order mismatch: {path}")
    prediction = sum(weight * rank01(frame[prediction_column]) for (weight, _), frame in zip(args.inputs, frames))
    result = reference.copy()
    result[prediction_column] = prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        result.to_parquet(args.output, index=False)
    else:
        result[["id", prediction_column]].to_csv(args.output, index=False)
    message = f"Saved {args.output}: rows={len(result)}"
    if prediction_column == "prediction" and "flag" in result:
        message += f", auc={roc_auc_score(result['flag'], prediction):.9f}"
    print(message)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def rank01(values: pd.Series) -> np.ndarray:
    return values.rank(method="average").to_numpy(np.float64) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/submission_full100_checkpoint_average_idprior.csv"))
    parser.add_argument("--alpha", type=Path, default=Path("runs/full100_alpha_gru_payment/submission_epoch_9.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--alpha-weight", type=float, default=0.161)
    parser.add_argument("--output", type=Path, default=Path("submissions/submission_alpha_gru161.csv"))
    args = parser.parse_args()

    base = pd.read_csv(args.base, usecols=["id", "flag"]).rename(columns={"flag": "base"})
    alpha = pd.read_csv(args.alpha, usecols=["id", "flag"]).rename(columns={"flag": "alpha"})
    sample = pd.read_csv(args.sample, usecols=["id"])
    data = base.merge(alpha, on="id", validate="one_to_one")
    if not data["id"].equals(sample["id"]):
        raise RuntimeError("Submission id order does not match sample_submission.csv")

    prediction = (1 - args.alpha_weight) * rank01(data["base"]) + args.alpha_weight * rank01(data["alpha"])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Prediction contains non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({"id": data["id"], "flag": prediction})
    result.to_csv(args.output, index=False)
    print(
        f"saved {args.output}: rows={len(result)} min={result['flag'].min():.8f} max={result['flag'].max():.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

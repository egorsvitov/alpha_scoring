from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_id_target_prior import make_prior


ROOT = Path(__file__).resolve().parent


def rank01(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(np.float64) / len(values)


def load(path: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["id", "flag"]).rename(columns={"flag": name})


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the checkpoint-averaged full-train ensemble.")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "sequence_runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path)
    parser.add_argument("--prior-weight", type=float, default=0.021)
    args = parser.parse_args()

    runs = args.runs_dir
    sources = [
        (runs / "full100_pooling_seed42/submission_epoch_average.csv", "pooling"),
        (runs / "full100_pooling_seed137/submission_epoch_average.csv", "pooling_seed137"),
        (runs / "full100_product_transformer/submission_epoch_average.csv", "product_transformer"),
        (runs / "full100_dual_pooling/submission_epoch_average.csv", "dual_pooling"),
        (runs / "full100_late_fusion/submission_epoch_average.csv", "late_fusion"),
        (runs / "full100_payment_transformer/submission_epoch_average.csv", "payment_transformer"),
        (runs / "full100_hierarchical_transformer/submission_epoch_average.csv", "hierarchical"),
        (runs / "blends/submission_temporal_full_two_seed.csv", "temporal_pooling"),
    ]
    data: pd.DataFrame | None = None
    for path, name in sources:
        frame = load(path, name)
        data = frame if data is None else data.merge(frame, on="id", validate="one_to_one")
    assert data is not None
    for path, name in [
        (ROOT / "submission_catboost_v4.csv", "v4"),
        (ROOT / "submission_catboost_v5.csv", "v5"),
        (ROOT / "submission_lightgbm_v5.csv", "lightgbm"),
    ]:
        data = data.merge(load(path, name), on="id", validate="one_to_one")

    tabular = 0.30 * rank01(data["v4"]) + 0.20 * rank01(data["v5"]) + 0.50 * rank01(data["lightgbm"])
    polished = (
        0.218196 * rank01(data["pooling"])
        + 0.250000 * rank01(data["pooling_seed137"])
        + 0.241164 * rank01(data["product_transformer"])
        + 0.114840 * rank01(tabular)
        + 0.085800 * rank01(data["dual_pooling"])
        + 0.090000 * rank01(data["late_fusion"])
    )
    payment = 0.75 * rank01(polished) + 0.25 * rank01(data["payment_transformer"])
    hierarchical = 0.79 * rank01(payment) + 0.21 * rank01(data["hierarchical"])
    prediction = 0.696 * rank01(hierarchical) + 0.304 * rank01(data["temporal_pooling"])

    sample = pd.read_csv(ROOT / "sample_submission.csv", usecols=["id"])
    if not data["id"].equals(sample["id"]):
        raise RuntimeError("Submission id order mismatch")
    result = pd.DataFrame({"id": data["id"], "flag": prediction})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {args.output}: rows={len(result)}")

    if args.prior_output is not None:
        target = pd.read_csv(ROOT / "train_target.csv", usecols=["id", "flag"])
        prior = make_prior(
            target["id"].to_numpy(np.int64),
            target["flag"].to_numpy(np.float64),
            data["id"].to_numpy(np.int64),
            bins=400,
            sigma=3.0,
            smoothing=0.0,
            id_min=int(min(target["id"].min(), data["id"].min())),
            id_max=int(max(target["id"].max(), data["id"].max())),
        )
        prior_prediction = (1 - args.prior_weight) * rank01(prediction) + args.prior_weight * rank01(prior)
        pd.DataFrame({"id": data["id"], "flag": prior_prediction}).to_csv(args.prior_output, index=False)
        print(f"Saved {args.prior_output}: prior_weight={args.prior_weight:.3f}")


if __name__ == "__main__":
    main()

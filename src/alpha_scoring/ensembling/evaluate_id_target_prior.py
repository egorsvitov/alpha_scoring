from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from alpha_scoring.paths import PROJECT_ROOT as ROOT


def rank01(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(np.float64) / len(values)


def current_ensemble(runs_dir: Path) -> pd.DataFrame:
    raise RuntimeError(
        "OOF audit assembly is not included in this trimmed final pipeline. "
        f"Cannot build current ensemble from {runs_dir}."
    )


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(4 * sigma)))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / sigma) ** 2)
    return kernel / kernel.sum()


def make_prior(
    fit_ids: np.ndarray,
    fit_target: np.ndarray,
    predict_ids: np.ndarray,
    bins: int,
    sigma: float,
    smoothing: float,
    id_min: int,
    id_max: int,
) -> np.ndarray:
    width = id_max - id_min + 1
    fit_bins = np.minimum((fit_ids - id_min) * bins // width, bins - 1)
    predict_bins = np.minimum(np.maximum((predict_ids - id_min) * bins // width, 0), bins - 1)
    counts = np.bincount(fit_bins, minlength=bins).astype(np.float64)
    positives = np.bincount(fit_bins, weights=fit_target, minlength=bins).astype(np.float64)
    if sigma > 0:
        kernel = gaussian_kernel(sigma)
        counts = np.convolve(counts, kernel, mode="same")
        positives = np.convolve(positives, kernel, mode="same")
    global_rate = fit_target.mean()
    rates = (positives + smoothing * global_rate) / (counts + smoothing)
    return rates[predict_bins]


def block_scores(target: np.ndarray, prediction: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    return np.asarray(
        [roc_auc_score(target[blocks == block], prediction[blocks == block]) for block in np.unique(blocks)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate leakage-safe local target-rate priors over application id.")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "sequence_runs")
    parser.add_argument("--output", type=Path, default=ROOT / "id_target_prior_search.csv")
    parser.add_argument("--weights", default="0,0.01,0.02,0.03,0.05,0.08,0.12")
    args = parser.parse_args()

    base = current_ensemble(args.runs_dir).sort_values("id").reset_index(drop=True)
    target_frame = pd.read_csv(ROOT / "train_target.csv", usecols=["id", "flag"])
    validation_ids = pd.Index(base["id"])
    fit = target_frame.loc[~target_frame["id"].isin(validation_ids)]
    if len(fit) + len(base) != len(target_frame):
        raise RuntimeError("OOF ids do not partition train_target.csv as expected")

    fit_ids = fit["id"].to_numpy(np.int64)
    fit_target = fit["flag"].to_numpy(np.float64)
    predict_ids = base["id"].to_numpy(np.int64)
    target = base["flag"].to_numpy(np.int8)
    blocks = base["block"].to_numpy()
    base_rank = rank01(base["current"])
    base_auc = roc_auc_score(target, base_rank)
    base_blocks = block_scores(target, base_rank, blocks)
    weights = np.asarray([float(value) for value in args.weights.split(",")])

    configs = []
    for bins in (50, 100, 200, 400, 800, 1200, 2000):
        for sigma in (0.0, 0.75, 1.5, 3.0, 6.0, 12.0):
            configs.append((bins, sigma, 0.0))

    rows = []
    for bins, sigma, smoothing in configs:
        prior = make_prior(
            fit_ids,
            fit_target,
            predict_ids,
            bins,
            sigma,
            smoothing,
            int(target_frame["id"].min()),
            int(target_frame["id"].max()),
        )
        prior_rank = rank01(prior)
        score_matrix = []
        overall = []
        for weight in weights:
            prediction = (1 - weight) * base_rank + weight * prior_rank
            overall.append(roc_auc_score(target, prediction))
            score_matrix.append(block_scores(target, prediction, blocks))
        score_matrix = np.asarray(score_matrix)
        overall = np.asarray(overall)

        selected_indices = []
        held_out_deltas = []
        for block_index in range(score_matrix.shape[1]):
            train_means = np.delete(score_matrix, block_index, axis=1).mean(axis=1)
            selected_index = int(np.argmax(train_means))
            selected_indices.append(selected_index)
            held_out_deltas.append(score_matrix[selected_index, block_index] - base_blocks[block_index])
        selected_weights = weights[selected_indices]
        robust_weight = float(selected_weights.mean())
        robust_prediction = (1 - robust_weight) * base_rank + robust_weight * prior_rank
        best_index = int(np.argmax(overall))
        rows.append(
            {
                "bins": bins,
                "sigma": sigma,
                "smoothing": smoothing,
                "prior_auc": roc_auc_score(target, prior_rank),
                "global_best_weight": weights[best_index],
                "global_best_delta": overall[best_index] - base_auc,
                "robust_weight": robust_weight,
                "robust_delta": roc_auc_score(target, robust_prediction) - base_auc,
                "held_out_delta_mean": np.mean(held_out_deltas),
                "held_out_delta_min": np.min(held_out_deltas),
                "held_out_positive_blocks": np.sum(np.asarray(held_out_deltas) > 0),
                "selected_weight_min": selected_weights.min(),
                "selected_weight_max": selected_weights.max(),
            }
        )

    report = pd.DataFrame(rows).sort_values(
        ["held_out_delta_mean", "robust_delta"], ascending=False
    )
    report.to_csv(args.output, index=False)
    print(f"base_auc={base_auc:.9f} fit_rows={len(fit)} validation_rows={len(base)}")
    print(report.head(30).to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

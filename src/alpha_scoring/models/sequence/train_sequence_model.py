from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from alpha_scoring.models.tabular.compare_v4_v5_block_validation import assign_blocks, validation_mask
from alpha_scoring.models.sequence.sequence_models import SequenceCacheDataset, SequenceClassifier, collate_sequences, load_metadata


from alpha_scoring.paths import PROJECT_ROOT as ROOT


def build_temporal_config(
    ids: np.ndarray,
    target: np.ndarray,
    train_indices: np.ndarray,
    bins: int,
    smoothing: float,
) -> dict[str, object]:
    train_ids = ids[train_indices].astype(np.int64)
    train_target = target[train_indices].astype(np.float64)
    id_min = int(ids.min())
    id_max = int(ids.max())
    width = max(id_max - id_min + 1, 1)
    bin_index = np.minimum((train_ids - id_min) * bins // width, bins - 1)
    counts = np.bincount(bin_index, minlength=bins).astype(np.float64)
    positives = np.bincount(bin_index, weights=train_target, minlength=bins)
    global_rate = float(train_target.mean())
    priors = (positives + smoothing * global_rate) / (counts + smoothing)
    return {
        "id_min": id_min,
        "id_max": id_max,
        "bins": bins,
        "smoothing": smoothing,
        "global_rate": global_rate,
        "bin_counts": counts.tolist(),
        "bin_positives": positives.tolist(),
        "bin_priors": priors.tolist(),
    }


def temporal_features(
    ids: np.ndarray,
    config: dict[str, object],
    target: np.ndarray | None = None,
) -> np.ndarray:
    id_min = int(config["id_min"])
    id_max = int(config["id_max"])
    bins = int(config["bins"])
    width = max(id_max - id_min + 1, 1)
    position = np.clip((ids.astype(np.float64) - id_min) / width, 0.0, 1.0)
    bin_index = np.minimum(((ids.astype(np.int64) - id_min) * bins // width).clip(min=0), bins - 1)
    if target is None:
        priors = np.asarray(config["bin_priors"], dtype=np.float64)[bin_index]
    else:
        counts = np.asarray(config["bin_counts"], dtype=np.float64)[bin_index] - 1
        positives = np.asarray(config["bin_positives"], dtype=np.float64)[bin_index] - target
        smoothing = float(config["smoothing"])
        priors = (positives + smoothing * float(config["global_rate"])) / (counts + smoothing)
    global_rate = float(config["global_rate"])
    prior_logit = np.log(np.clip(priors, 1e-6, 1 - 1e-6) / np.clip(1 - priors, 1e-6, 1))
    global_logit = np.log(global_rate / (1 - global_rate))
    features = [position, position**2, prior_logit - global_logit]
    for period in (10.0, 20.0, 50.0):
        angle = 2 * np.pi * position * period
        features.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack(features).astype(np.float32)


def id_sample_weights(ids: torch.Tensor, id_min: int, id_max: int, mode: str, strength: float) -> torch.Tensor:
    if mode == "none" or strength <= 0:
        return torch.ones_like(ids, dtype=torch.float32)
    position = (ids.float() - float(id_min)) / max(float(id_max - id_min), 1.0)
    position = position.clamp(0.0, 1.0)
    if mode == "linear":
        weights = 1.0 + strength * position
    elif mode == "exponential":
        weights = torch.exp(strength * (position - 0.5))
    else:
        raise ValueError(f"Unknown sample weight mode: {mode}")
    return weights / weights.mean().clamp_min(1e-6)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader | None, np.ndarray, np.ndarray, dict[str, object] | None]:
    ids = np.load(args.cache_dir / "train" / "ids.npy", mmap_mode="r")
    target = np.load(args.cache_dir / "train" / "target.npy", mmap_mode="r")
    if args.full_train:
        train_indices = np.arange(len(ids), dtype=np.int64)
        valid_indices = np.empty(0, dtype=np.int64)
    else:
        valid = validation_mask(pd.Series(ids), args.valid_fraction, args.split_seed)
        train_indices = np.flatnonzero(~valid)
        valid_indices = np.flatnonzero(valid)
    if args.max_train_samples is not None:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_valid_samples is not None:
        valid_indices = valid_indices[: args.max_valid_samples]
    temporal_config = None
    time_features = None
    if args.temporal_features:
        temporal_config = build_temporal_config(ids, target, train_indices, args.temporal_bins, args.temporal_smoothing)
        time_features = temporal_features(ids, temporal_config)
        train_time_features = time_features.copy()
        train_time_features[train_indices] = temporal_features(
            ids[train_indices], temporal_config, target[train_indices]
        )
    else:
        train_time_features = None
    train_dataset = SequenceCacheDataset(
        args.cache_dir, "train", train_indices, args.tabular_cache_dir, args.tabular_feature_count, train_time_features
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": collate_sequences,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common)
    valid_loader = None
    if len(valid_indices):
        valid_dataset = SequenceCacheDataset(
            args.cache_dir, "train", valid_indices, args.tabular_cache_dir, args.tabular_feature_count, time_features
        )
        valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, valid_loader, train_indices, valid_indices, temporal_config


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    ids: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        values = batch["values"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        tabular = batch.get("tabular")
        temporal = batch.get("temporal")
        if tabular is not None:
            tabular = tabular.to(device, non_blocking=True)
        if temporal is not None:
            temporal = temporal.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(values, mask, tabular, temporal)
        ids.append(batch["id"].numpy())
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        if "target" in batch:
            targets.append(batch["target"].numpy())
    target_array = np.concatenate(targets) if targets else None
    return np.concatenate(ids), np.concatenate(predictions), target_array


def block_metrics(ids: np.ndarray, target: np.ndarray, prediction: np.ndarray, blocks: int) -> pd.DataFrame:
    block_values = assign_blocks(pd.Series(ids), blocks)
    rows = []
    for block in range(blocks):
        mask = block_values == block
        rows.append(
            {
                "block": block,
                "rows": int(mask.sum()),
                "default_rate": float(target[mask].mean()),
                "auc": roc_auc_score(target[mask], prediction[mask]),
            }
        )
    return pd.DataFrame(rows)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for sequence training")
    amp = args.amp and device.type == "cuda"
    metadata = load_metadata(args.cache_dir)
    tabular_dim = 0
    if args.tabular_cache_dir is not None:
        tabular_metadata = json.loads((args.tabular_cache_dir / "metadata.json").read_text(encoding="utf-8"))
        available_tabular_dim = int(tabular_metadata["feature_count"])
        tabular_dim = args.tabular_feature_count or available_tabular_dim
        if tabular_dim > available_tabular_dim:
            raise ValueError(f"Requested {tabular_dim} tabular features, cache has {available_tabular_dim}")
    args.tabular_dim = tabular_dim
    ids = np.load(args.cache_dir / "train" / "ids.npy", mmap_mode="r")
    args.train_id_min = int(ids.min())
    args.train_id_max = int(ids.max())
    train_loader, valid_loader, _, valid_indices, temporal_config = make_loaders(args)
    args.temporal_config = temporal_config
    args.temporal_dim = 9 if temporal_config is not None else 0
    model = SequenceClassifier(
        metadata,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        tabular_dim=tabular_dim,
        fusion_mode=args.fusion_mode,
        fusion_init=args.fusion_init,
        payment_encoder=args.payment_encoder,
        payment_hidden_dim=args.payment_hidden_dim,
        payment_layers=args.payment_layers,
        payment_heads=args.payment_heads,
        payment_windows=args.payment_windows,
        recent_products=args.recent_products,
        payment_aggregates=args.payment_aggregates,
        product_recency=args.product_recency,
        temporal_dim=args.temporal_dim,
    ).to(device)
    if args.pretrained_sequence is not None:
        pretrained = torch.load(args.pretrained_sequence, map_location=device)
        missing, unexpected = model.load_state_dict(pretrained["model"], strict=False)
        allowed_missing = {name for name in missing if name.startswith(("tabular_encoder.", "tabular_classifier.", "fusion_logit"))}
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(f"Pretrained state mismatch: missing={missing}, unexpected={unexpected}")
        print(f"Loaded pretrained sequence: {args.pretrained_sequence}", flush=True)
    if args.freeze_sequence:
        model.set_sequence_trainable(False)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
        )
    elif args.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device), reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -np.inf
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_sequence:
            model.set_frozen_sequence_eval()
        running_loss = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            values = batch["values"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            tabular = batch.get("tabular")
            temporal = batch.get("temporal")
            if tabular is not None:
                tabular = tabular.to(device, non_blocking=True)
            if temporal is not None:
                temporal = temporal.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(values, mask, tabular, temporal)
                loss_values = loss_function(logits, target)
                weights = id_sample_weights(
                    batch["id"].to(device, non_blocking=True),
                    args.train_id_min,
                    args.train_id_max,
                    args.sample_weight_mode,
                    args.sample_weight_strength,
                )
                loss = (loss_values * weights).sum() / weights.sum().clamp_min(1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if args.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()
            running_loss += float(loss.detach()) * len(target)
            seen += len(target)
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss / seen:.6f}", flush=True)

        if args.full_train:
            epoch_row = {
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_row)
            pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
            print(json.dumps(epoch_row), flush=True)
            checkpoint = {
                "model": model.state_dict(),
                "metadata": metadata,
                "args": vars(args),
                "epoch": epoch,
                "auc": None,
            }
            if args.save_epoch_checkpoints:
                torch.save(checkpoint, args.output_dir / f"epoch_{epoch}.pt")
            torch.save(checkpoint, args.output_dir / "last.pt")
            if args.scheduler == "cosine" and scheduler is not None:
                scheduler.step()
            continue

        if valid_loader is None:
            raise RuntimeError("Validation loader is missing")
        valid_ids, valid_prediction, valid_target = predict(model, valid_loader, device, amp)
        if valid_target is None:
            raise RuntimeError("Validation target is missing")
        auc = roc_auc_score(valid_target, valid_prediction)
        block_frame = block_metrics(valid_ids, valid_target, valid_prediction, args.blocks)
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "valid_auc": auc,
            "block_auc_mean": float(block_frame["auc"].mean()),
            "block_auc_std": float(block_frame["auc"].std(ddof=0)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if model.fusion_logit is not None:
            epoch_row["fusion_weight"] = float(torch.sigmoid(model.fusion_logit).detach().cpu())
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        block_frame.to_csv(args.output_dir / f"blocks_epoch_{epoch}.csv", index=False)
        print(json.dumps(epoch_row), flush=True)

        if args.save_epoch_checkpoints:
            checkpoint = {
                "model": model.state_dict(),
                "metadata": metadata,
                "args": vars(args),
                "epoch": epoch,
                "auc": auc,
            }
            torch.save(checkpoint, args.output_dir / f"epoch_{epoch}.pt")
            pd.DataFrame(
                {"id": valid_ids, "flag": valid_target.astype(np.uint8), "block": assign_blocks(pd.Series(valid_ids), args.blocks), "prediction": valid_prediction}
            ).to_parquet(args.output_dir / f"oof_epoch_{epoch}.parquet", index=False)

        if auc > best_auc + args.min_delta:
            best_auc = auc
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "metadata": metadata,
                    "args": vars(args),
                    "epoch": epoch,
                    "auc": auc,
                },
                args.output_dir / "best.pt",
            )
            pd.DataFrame(
                {"id": valid_ids, "flag": valid_target.astype(np.uint8), "block": assign_blocks(pd.Series(valid_ids), args.blocks), "prediction": valid_prediction}
            ).to_parquet(args.output_dir / "oof.parquet", index=False)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_auc={best_auc:.6f}")
                break
        if args.scheduler == "cosine" and scheduler is not None:
            scheduler.step()


def predict_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint["args"]
    model = SequenceClassifier(
        checkpoint["metadata"],
        architecture=saved_args["architecture"],
        hidden_dim=saved_args["hidden_dim"],
        embedding_dim=saved_args["embedding_dim"],
        layers=saved_args["layers"],
        heads=saved_args["heads"],
        dropout=saved_args["dropout"],
        tabular_dim=saved_args.get("tabular_dim", 0),
        fusion_mode=saved_args.get("fusion_mode", "concat"),
        fusion_init=saved_args.get("fusion_init", 0.1),
        payment_encoder=saved_args.get("payment_encoder", "cnn"),
        payment_hidden_dim=saved_args.get("payment_hidden_dim", 32),
        payment_layers=saved_args.get("payment_layers", 2),
        payment_heads=saved_args.get("payment_heads", 4),
        payment_windows=tuple(saved_args.get("payment_windows", (6, 12, 25))),
        recent_products=saved_args.get("recent_products", 5),
        payment_aggregates=saved_args.get("payment_aggregates", False),
        product_recency=saved_args.get("product_recency", False),
        temporal_dim=saved_args.get("temporal_dim", 0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    tabular_cache_dir = args.tabular_cache_dir
    if tabular_cache_dir is None and saved_args.get("tabular_cache_dir"):
        saved_path = Path(saved_args["tabular_cache_dir"])
        tabular_cache_dir = saved_path if saved_path.exists() else ROOT / saved_path.name
    time_features = None
    if saved_args.get("temporal_config") is not None:
        test_ids = np.load(args.cache_dir / "test" / "ids.npy", mmap_mode="r")
        time_features = temporal_features(test_ids, saved_args["temporal_config"])
    dataset = SequenceCacheDataset(
        args.cache_dir,
        "test",
        tabular_cache_dir=tabular_cache_dir,
        tabular_feature_count=saved_args.get("tabular_feature_count"),
        temporal_features=time_features,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_sequences,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    ids, prediction, _ = predict(model, loader, device, args.amp)
    pd.DataFrame({"id": ids, "flag": prediction}).to_csv(args.test_output, index=False)
    print(f"Saved test predictions: {args.test_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pooling, GRU or Transformer on product sequences.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "sequence_cache")
    parser.add_argument("--tabular-cache-dir", type=Path, default=None)
    parser.add_argument("--tabular-feature-count", type=int, default=None)
    parser.add_argument(
        "--architecture", choices=["pooling", "gru", "alpha_gru", "transformer", "multiscale_transformer"], default="gru"
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fusion-mode", choices=["concat", "late"], default="concat")
    parser.add_argument("--fusion-init", type=float, default=0.1)
    parser.add_argument("--payment-encoder", choices=["cnn", "transformer", "multiscale_transformer"], default="cnn")
    parser.add_argument("--payment-hidden-dim", type=int, default=32)
    parser.add_argument("--payment-layers", type=int, default=2)
    parser.add_argument("--payment-heads", type=int, default=4)
    parser.add_argument("--payment-windows", type=lambda value: tuple(map(int, value.split(","))), default=(6, 12, 25))
    parser.add_argument("--recent-products", type=int, default=5)
    parser.add_argument("--payment-aggregates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--product-recency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-bins", type=int, default=50)
    parser.add_argument("--temporal-smoothing", type=float, default=2000.0)
    parser.add_argument("--pretrained-sequence", type=Path, default=None)
    parser.add_argument("--freeze-sequence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--save-epoch-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["cosine", "onecycle", "none"], default="cosine")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10000.0)
    parser.add_argument("--sample-weight-mode", choices=["none", "linear", "exponential"], default="none")
    parser.add_argument("--sample-weight-strength", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=20260607)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sequence_runs" / "gru")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--test-output", type=Path, default=ROOT / "sequence_test_predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predict_test:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required with --predict-test")
        predict_test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

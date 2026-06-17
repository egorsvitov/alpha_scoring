from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from alpha_scoring.model import AlphaGRUClassifier, SequenceDataset, collate_sequences, load_metadata


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def id_sample_weights(ids: torch.Tensor, id_min: int, id_max: int, strength: float) -> torch.Tensor:
    position = (ids.float() - float(id_min)) / max(float(id_max - id_min), 1.0)
    weights = 1.0 + strength * position.clamp(0.0, 1.0)
    return weights / weights.mean().clamp_min(1e-6)


def make_loader(cache_dir: Path, split: str, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SequenceDataset(cache_dir, split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_sequences,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required")

    metadata = load_metadata(args.cache_dir)
    ids = np.load(args.cache_dir / "train" / "ids.npy", mmap_mode="r")
    train_loader = make_loader(args.cache_dir, "train", args.batch_size, args.workers, shuffle=True)
    model = AlphaGRUClassifier(metadata, args.hidden_dim, args.embedding_dim, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device), reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            values = batch["values"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(values, mask)
                weights = id_sample_weights(
                    batch["id"].to(device, non_blocking=True),
                    int(ids.min()),
                    int(ids.max()),
                    args.sample_weight_strength,
                )
                loss = (loss_function(logits, target) * weights).sum() / weights.sum().clamp_min(1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.detach()) * len(target)
            seen += len(target)
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss / seen:.6f}", flush=True)

        row = {"epoch": epoch, "train_loss": running_loss / seen, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        checkpoint = {"model": model.state_dict(), "metadata": metadata, "args": vars(args), "epoch": epoch}
        torch.save(checkpoint, args.output_dir / f"epoch_{epoch}.pt")
        torch.save(checkpoint, args.output_dir / "last.pt")


@torch.no_grad()
def predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint["args"]
    model = AlphaGRUClassifier(
        checkpoint["metadata"],
        saved_args["hidden_dim"],
        saved_args["embedding_dim"],
        saved_args["layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loader = make_loader(args.cache_dir, "test", args.batch_size, args.workers, shuffle=False)
    ids: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch in loader:
        values = batch["values"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            logits = model(values, mask)
        ids.append(batch["id"].numpy())
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": np.concatenate(ids), "flag": np.concatenate(predictions)}).to_csv(args.output, index=False)
    print(f"saved {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--cache-dir", type=Path, default=Path("sequence_cache"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("runs/full100_alpha_gru_payment"))
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--embedding-dim", type=int, default=8)
    train_parser.add_argument("--layers", type=int, default=2)
    train_parser.add_argument("--batch-size", type=int, default=1024)
    train_parser.add_argument("--workers", type=int, default=6)
    train_parser.add_argument("--epochs", type=int, default=9)
    train_parser.add_argument("--learning-rate", type=float, default=0.0015)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--sample-weight-strength", type=float, default=1.0)
    train_parser.add_argument("--pos-weight", type=float, default=5.0)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--log-every", type=int, default=200)
    train_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--cache-dir", type=Path, default=Path("sequence_cache"))
    predict_parser.add_argument("--checkpoint", type=Path, default=Path("runs/full100_alpha_gru_payment/epoch_9.pt"))
    predict_parser.add_argument("--output", type=Path, default=Path("runs/full100_alpha_gru_payment/submission_epoch_9.csv"))
    predict_parser.add_argument("--batch-size", type=int, default=2048)
    predict_parser.add_argument("--workers", type=int, default=6)
    predict_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()

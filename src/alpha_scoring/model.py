from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str) -> None:
        split_dir = cache_dir / split
        self.values = np.load(split_dir / "values.npy", mmap_mode="r")
        self.ids = np.load(split_dir / "ids.npy", mmap_mode="r")
        self.offsets = np.load(split_dir / "offsets.npy", mmap_mode="r")
        target_path = split_dir / "target.npy"
        self.target = np.load(target_path, mmap_mode="r") if target_path.exists() else None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray, int | None]:
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        target = int(self.target[index]) if self.target is not None else None
        return int(self.ids[index]), np.asarray(self.values[start:end]), target


def collate_sequences(batch: list[tuple[int, np.ndarray, int | None]]) -> dict[str, torch.Tensor]:
    max_length = max(len(values) for _, values, _ in batch)
    feature_count = batch[0][1].shape[1]
    values = torch.zeros((len(batch), max_length, feature_count), dtype=torch.long)
    mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    ids = torch.empty(len(batch), dtype=torch.long)
    has_target = batch[0][2] is not None
    target = torch.empty(len(batch), dtype=torch.float32) if has_target else None
    for row, (client_id, sequence, label) in enumerate(batch):
        length = len(sequence)
        values[row, :length] = torch.from_numpy(sequence.astype(np.int64, copy=True))
        mask[row, :length] = True
        ids[row] = client_id
        if target is not None:
            target[row] = float(label)
    result = {"id": ids, "values": values, "mask": mask}
    if target is not None:
        result["target"] = target
    return result


class PaymentTransformer(nn.Module):
    def __init__(self, cardinality: int, payment_count: int, output_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.cardinality = cardinality
        self.payment_count = payment_count
        self.code_embedding = nn.Embedding(cardinality, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(payment_count + 1, hidden_dim)
        self.transition_embedding = nn.Embedding(cardinality * cardinality, hidden_dim, padding_idx=0)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim), nn.GELU())

    def forward(self, payments: torch.Tensor) -> torch.Tensor:
        batch = payments.shape[0]
        positions = torch.arange(1, self.payment_count + 1, device=payments.device)
        encoded = self.code_embedding(payments) + self.position_embedding(positions).unsqueeze(0)
        previous = torch.cat([torch.zeros_like(payments[:, :1]), payments[:, :-1]], dim=1)
        encoded = encoded + self.transition_embedding(previous * self.cardinality + payments)
        cls = self.cls.expand(batch, -1, -1) + self.position_embedding.weight[:1].unsqueeze(0)
        return self.output(self.transformer(torch.cat([cls, encoded], dim=1))[:, 0])


class ProductEncoder(nn.Module):
    def __init__(self, metadata: dict[str, object], hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        product_columns: list[str] = metadata["product_columns"]  # type: ignore[assignment]
        payment_columns: list[str] = metadata["payment_columns"]  # type: ignore[assignment]
        cardinalities: dict[str, int] = metadata["cardinalities"]  # type: ignore[assignment]
        self.product_count = len(product_columns)
        self.payment_count = len(payment_columns)
        self.product_embeddings = nn.ModuleList(
            [nn.Embedding(cardinalities[col], embedding_dim, padding_idx=0) for col in product_columns]
        )
        payment_cardinality = max(cardinalities[col] for col in payment_columns)
        self.payment_encoder = PaymentTransformer(payment_cardinality, self.payment_count, hidden_dim // 2)
        input_dim = self.product_count * embedding_dim + hidden_dim // 2
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = values.shape
        product = torch.cat(
            [embedding(values[:, :, index]) for index, embedding in enumerate(self.product_embeddings)],
            dim=-1,
        )
        payments = values[:, :, self.product_count : self.product_count + self.payment_count]
        flat_payments = payments.reshape(batch * length, self.payment_count)
        flat_mask = mask.reshape(-1)
        real_payment = self.payment_encoder(flat_payments[flat_mask])
        payment = real_payment.new_zeros((batch * length, real_payment.shape[-1]))
        payment[flat_mask] = real_payment
        payment = payment.reshape(batch, length, -1)
        return self.projection(torch.cat([product, payment], dim=-1))


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(values).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class AlphaGRUClassifier(nn.Module):
    def __init__(self, metadata: dict[str, object], hidden_dim: int = 128, embedding_dim: int = 8, layers: int = 2) -> None:
        super().__init__()
        self.encoder = ProductEncoder(metadata, hidden_dim, embedding_dim)
        self.position = nn.Embedding(64, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.output_attention = AttentionPool(hidden_dim)
        self.input_attention = AttentionPool(hidden_dim)
        self.final_state_projection = nn.Sequential(
            nn.Linear(hidden_dim * layers, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded_input = self.encoder(values, mask)
        positions = torch.arange(values.shape[1], device=values.device)
        encoded_input = encoded_input + self.position(positions).unsqueeze(0)
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(encoded_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, final_state = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=mask.shape[1])

        output_masked = output.masked_fill(~mask.unsqueeze(-1), 0)
        input_masked = encoded_input.masked_fill(~mask.unsqueeze(-1), 0)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        parts = [
            output_masked.sum(dim=1) / denominator,
            output.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values,
            self.output_attention(output, mask),
            input_masked.sum(dim=1) / denominator,
            encoded_input.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values,
            self.input_attention(encoded_input, mask),
            self.final_state_projection(final_state.transpose(0, 1).reshape(values.shape[0], -1)),
        ]
        return self.classifier(torch.cat(parts, dim=-1)).squeeze(-1)


def load_metadata(cache_dir: Path, split: str = "train") -> dict[str, object]:
    return json.loads((cache_dir / split / "metadata.json").read_text(encoding="utf-8"))

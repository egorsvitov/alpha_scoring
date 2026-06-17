from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class SequenceCacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        split: str,
        indices: np.ndarray | None = None,
        tabular_cache_dir: Path | None = None,
        tabular_feature_count: int | None = None,
        temporal_features: np.ndarray | None = None,
    ) -> None:
        self.split_dir = cache_dir / split
        self.values = np.load(self.split_dir / "values.npy", mmap_mode="r")
        self.ids = np.load(self.split_dir / "ids.npy", mmap_mode="r")
        self.offsets = np.load(self.split_dir / "offsets.npy", mmap_mode="r")
        target_path = self.split_dir / "target.npy"
        self.target = np.load(target_path, mmap_mode="r") if target_path.exists() else None
        self.tabular = np.load(tabular_cache_dir / split / "values.npy", mmap_mode="r") if tabular_cache_dir else None
        self.tabular_feature_count = tabular_feature_count
        self.temporal_features = temporal_features
        if self.tabular is not None and len(self.tabular) != len(self.ids):
            raise RuntimeError(f"Tabular cache row mismatch for {split}")
        if self.temporal_features is not None and len(self.temporal_features) != len(self.ids):
            raise RuntimeError(f"Temporal feature row mismatch for {split}")
        self.indices = np.arange(len(self.ids), dtype=np.int64) if indices is None else indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[int, np.ndarray, int | None, np.ndarray | None, np.ndarray | None]:
        index = int(self.indices[item])
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        target = int(self.target[index]) if self.target is not None else None
        tabular = np.asarray(self.tabular[index, : self.tabular_feature_count]) if self.tabular is not None else None
        temporal = np.asarray(self.temporal_features[index]) if self.temporal_features is not None else None
        return int(self.ids[index]), np.asarray(self.values[start:end]), target, tabular, temporal


def collate_sequences(
    batch: list[tuple[int, np.ndarray, int | None, np.ndarray | None, np.ndarray | None]],
) -> dict[str, torch.Tensor]:
    max_length = max(len(values) for _, values, _, _, _ in batch)
    feature_count = batch[0][1].shape[1]
    values = torch.zeros((len(batch), max_length, feature_count), dtype=torch.long)
    mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    ids = torch.empty(len(batch), dtype=torch.long)
    has_target = batch[0][2] is not None
    targets = torch.empty(len(batch), dtype=torch.float32) if has_target else None
    has_tabular = batch[0][3] is not None
    tabular = torch.empty((len(batch), len(batch[0][3])), dtype=torch.float32) if has_tabular else None
    has_temporal = batch[0][4] is not None
    temporal = torch.empty((len(batch), len(batch[0][4])), dtype=torch.float32) if has_temporal else None
    for row, (client_id, sequence, target, aggregate, time_features) in enumerate(batch):
        length = len(sequence)
        values[row, :length] = torch.from_numpy(sequence.astype(np.int64, copy=True))
        mask[row, :length] = True
        ids[row] = client_id
        if targets is not None:
            targets[row] = float(target)
        if tabular is not None and aggregate is not None:
            tabular[row] = torch.from_numpy(aggregate.astype(np.float32, copy=True))
        if temporal is not None and time_features is not None:
            temporal[row] = torch.from_numpy(time_features.astype(np.float32, copy=True))
    result = {"id": ids, "values": values, "mask": mask}
    if targets is not None:
        result["target"] = targets
    if tabular is not None:
        result["tabular"] = tabular
    if temporal is not None:
        result["temporal"] = temporal
    return result


class PaymentTransformer(nn.Module):
    def __init__(
        self,
        cardinality: int,
        payment_count: int,
        output_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        use_aggregates: bool = False,
    ) -> None:
        super().__init__()
        self.cardinality = cardinality
        self.payment_count = payment_count
        self.use_aggregates = use_aggregates
        self.code_embedding = nn.Embedding(cardinality, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(payment_count + 1, hidden_dim)
        self.transition_embedding = nn.Embedding(cardinality * cardinality, hidden_dim, padding_idx=0)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        if use_aggregates:
            aggregate_dim = 34
            self.aggregate_encoder = nn.Sequential(
                nn.LayerNorm(aggregate_dim),
                nn.Linear(aggregate_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            output_input_dim = hidden_dim * 2
        else:
            self.aggregate_encoder = None
            output_input_dim = hidden_dim
        self.output = nn.Sequential(nn.LayerNorm(output_input_dim), nn.Linear(output_input_dim, output_dim), nn.GELU())

    @staticmethod
    def payment_aggregates(payments: torch.Tensor) -> torch.Tensor:
        # Cache values 1..4 correspond to original payment codes 0..3; 0 is padding only.
        codes = payments - 1
        features = []
        for window in (6, 12, payments.shape[1]):
            recent = codes[:, : min(window, payments.shape[1])]
            features.extend([(recent == code).float().mean(dim=1, keepdim=True) for code in range(4)])

        chronological = codes.flip(dims=(1,))
        source = chronological[:, :-1]
        target = chronological[:, 1:]
        for source_code in range(4):
            for target_code in range(4):
                transition = ((source == source_code) & (target == target_code)).float()
                features.append(transition.mean(dim=1, keepdim=True))

        changes = (source != target).float().mean(dim=1, keepdim=True)
        time = torch.linspace(-1, 1, payments.shape[1], device=payments.device)
        trend = (chronological.float() * time).mean(dim=1, keepdim=True)
        risky = chronological == 2
        runs = torch.zeros_like(risky, dtype=torch.float32)
        running = torch.zeros(payments.shape[0], device=payments.device)
        for index in range(payments.shape[1]):
            running = (running + 1) * risky[:, index]
            runs[:, index] = running
        longest_run = runs.max(dim=1, keepdim=True).values / payments.shape[1]
        current_run = runs[:, -1:].float() / payments.shape[1]
        entries = ((~risky[:, :-1]) & risky[:, 1:]).float().mean(dim=1, keepdim=True)
        exits = (risky[:, :-1] & (~risky[:, 1:])).float().mean(dim=1, keepdim=True)
        features.extend([changes, trend, longest_run, current_run, entries, exits])
        return torch.cat(features, dim=1)

    def forward(self, payments: torch.Tensor) -> torch.Tensor:
        batch = payments.shape[0]
        positions = torch.arange(1, self.payment_count + 1, device=payments.device)
        encoded = self.code_embedding(payments) + self.position_embedding(positions).unsqueeze(0)
        previous = torch.cat([torch.zeros_like(payments[:, :1]), payments[:, :-1]], dim=1)
        transition_ids = previous * self.cardinality + payments
        encoded = encoded + self.transition_embedding(transition_ids)
        cls = self.cls.expand(batch, -1, -1) + self.position_embedding.weight[:1].unsqueeze(0)
        encoded = self.transformer(torch.cat([cls, encoded], dim=1))
        pooled = encoded[:, 0]
        if self.aggregate_encoder is not None:
            aggregates = self.aggregate_encoder(self.payment_aggregates(payments))
            pooled = torch.cat([pooled, aggregates], dim=-1)
        return self.output(pooled)


class MultiWindowPaymentTransformer(nn.Module):
    def __init__(
        self,
        cardinality: int,
        payment_count: int,
        output_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        windows: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.payment_count = payment_count
        self.windows = tuple(dict.fromkeys(min(window, payment_count) for window in windows if window > 0))
        if not self.windows:
            raise ValueError("At least one payment window is required")
        branch_dim = max(hidden_dim, output_dim // len(self.windows))
        self.encoders = nn.ModuleList(
            [
                PaymentTransformer(
                    cardinality,
                    window,
                    branch_dim,
                    hidden_dim,
                    layers,
                    heads,
                    dropout,
                )
                for window in self.windows
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(branch_dim * len(self.windows), output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, payments: torch.Tensor) -> torch.Tensor:
        # enc_paym_0 is the latest month, so recent windows are prefixes.
        parts = [encoder(payments[:, :window]) for window, encoder in zip(self.windows, self.encoders)]
        return self.output(torch.cat(parts, dim=-1))


class ProductEncoder(nn.Module):
    def __init__(
        self,
        metadata: dict[str, object],
        hidden_dim: int,
        embedding_dim: int,
        payment_encoder: str = "cnn",
        payment_hidden_dim: int = 32,
        payment_layers: int = 2,
        payment_heads: int = 4,
        payment_windows: tuple[int, ...] = (6, 12, 25),
        payment_aggregates: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        columns: list[str] = metadata["feature_columns"]  # type: ignore[assignment]
        product_columns: list[str] = metadata["product_columns"]  # type: ignore[assignment]
        cardinalities: dict[str, int] = metadata["cardinalities"]  # type: ignore[assignment]
        self.product_count = len(product_columns)
        self.payment_count = len(metadata["payment_columns"])  # type: ignore[arg-type]
        self.product_embeddings = nn.ModuleList(
            [nn.Embedding(cardinalities[col], embedding_dim, padding_idx=0) for col in product_columns]
        )
        payment_cardinality = max(cardinalities[col] for col in metadata["payment_columns"])  # type: ignore[index]
        self.payment_encoder_type = payment_encoder
        if payment_encoder == "cnn":
            self.payment_embedding = nn.Embedding(payment_cardinality, embedding_dim, padding_idx=0)
            self.payment_encoder = nn.Sequential(
                nn.Conv1d(embedding_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
            )
        elif payment_encoder in {"transformer", "multiscale_transformer"}:
            self.payment_embedding = None
            if payment_encoder == "transformer":
                self.payment_encoder = PaymentTransformer(
                    payment_cardinality,
                    self.payment_count,
                    hidden_dim // 2,
                    payment_hidden_dim,
                    payment_layers,
                    payment_heads,
                    dropout,
                    payment_aggregates,
                )
            else:
                self.payment_encoder = MultiWindowPaymentTransformer(
                    payment_cardinality,
                    self.payment_count,
                    hidden_dim // 2,
                    payment_hidden_dim,
                    payment_layers,
                    payment_heads,
                    dropout,
                    payment_windows,
                )
        else:
            raise ValueError(f"Unknown payment encoder: {payment_encoder}")
        input_dim = self.product_count * embedding_dim + hidden_dim // 2
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = values.shape
        product_parts = [embedding(values[:, :, index]) for index, embedding in enumerate(self.product_embeddings)]
        product = torch.cat(product_parts, dim=-1)
        payments = values[:, :, self.product_count : self.product_count + self.payment_count]
        flat_payments = payments.reshape(batch * length, self.payment_count)
        flat_mask = mask.reshape(-1) if mask is not None else None
        real_payments = flat_payments[flat_mask] if flat_mask is not None else flat_payments
        if self.payment_encoder_type == "cnn":
            payment_emb = self.payment_embedding(real_payments).transpose(1, 2)  # type: ignore[operator]
            real_payment = self.payment_encoder(payment_emb)
        else:
            real_payment = self.payment_encoder(real_payments)
        if flat_mask is not None:
            payment = real_payment.new_zeros((batch * length, real_payment.shape[-1]))
            payment[flat_mask] = real_payment
        else:
            payment = real_payment
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


class RecencyAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.recency_strength = nn.Parameter(torch.tensor(-2.0))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        normalized_distance = distance.float() / mask.sum(dim=1, keepdim=True).clamp_min(1)
        scores = self.score(values).squeeze(-1)
        scores = scores - torch.nn.functional.softplus(self.recency_strength) * normalized_distance
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        metadata: dict[str, object],
        architecture: str,
        hidden_dim: int = 128,
        embedding_dim: int = 8,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        tabular_dim: int = 0,
        fusion_mode: str = "concat",
        fusion_init: float = 0.1,
        payment_encoder: str = "cnn",
        payment_hidden_dim: int = 32,
        payment_layers: int = 2,
        payment_heads: int = 4,
        payment_windows: tuple[int, ...] = (6, 12, 25),
        recent_products: int = 5,
        payment_aggregates: bool = False,
        product_recency: bool = False,
        temporal_dim: int = 0,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.tabular_dim = tabular_dim
        self.fusion_mode = fusion_mode
        self.recent_products = recent_products
        self.product_recency = product_recency
        self.temporal_dim = temporal_dim
        self.encoder = ProductEncoder(
            metadata,
            hidden_dim,
            embedding_dim,
            payment_encoder,
            payment_hidden_dim,
            payment_layers,
            payment_heads,
            payment_windows,
            payment_aggregates,
            dropout,
        )
        self.position = nn.Embedding(64, hidden_dim)
        self.reverse_position = nn.Embedding(64, hidden_dim) if product_recency else None
        if architecture in {"gru", "alpha_gru"}:
            self.sequence = nn.GRU(
                hidden_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0
            )
            if architecture == "alpha_gru":
                self.embedding_attention_pool = AttentionPool(hidden_dim)
                self.final_state_projection = nn.Sequential(
                    nn.Linear(hidden_dim * layers, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            else:
                self.embedding_attention_pool = None
                self.final_state_projection = None
        elif architecture in {"transformer", "multiscale_transformer"}:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence = nn.TransformerEncoder(layer, num_layers=layers)
            if architecture == "multiscale_transformer":
                recent_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.recent_sequence = nn.TransformerEncoder(recent_layer, num_layers=layers)
                self.recent_position = nn.Embedding(recent_products, hidden_dim)
                self.recent_attention_pool = AttentionPool(hidden_dim)
                self.scale_gate = nn.Sequential(nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
            else:
                self.recent_sequence = None
                self.recent_position = None
                self.recent_attention_pool = None
                self.scale_gate = None
        elif architecture == "pooling":
            self.sequence = nn.Identity()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.attention_pool = RecencyAttentionPool(hidden_dim) if product_recency else AttentionPool(hidden_dim)
        if tabular_dim:
            self.tabular_encoder = nn.Sequential(
                nn.LayerNorm(tabular_dim),
                nn.Linear(tabular_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.tabular_encoder = None
        if temporal_dim:
            self.temporal_encoder = nn.Sequential(
                nn.LayerNorm(temporal_dim),
                nn.Linear(temporal_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.temporal_encoder = None
        classifier_input = hidden_dim * (7 if architecture == "alpha_gru" else 3)
        if tabular_dim and fusion_mode == "concat":
            classifier_input += hidden_dim
        if temporal_dim:
            classifier_input += hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        if tabular_dim and fusion_mode == "late":
            self.tabular_classifier = nn.Linear(hidden_dim, 1)
            fusion_init = min(max(fusion_init, 1e-4), 1 - 1e-4)
            self.fusion_logit = nn.Parameter(torch.tensor(float(np.log(fusion_init / (1 - fusion_init)))))
        else:
            self.tabular_classifier = None
            self.fusion_logit = None

    def sequence_parameters(self):
        modules = [
            self.encoder,
            self.position,
            self.reverse_position,
            self.sequence,
            self.attention_pool,
            self.classifier,
            getattr(self, "embedding_attention_pool", None),
            getattr(self, "final_state_projection", None),
        ]
        if self.architecture == "multiscale_transformer":
            modules.extend([self.recent_sequence, self.recent_position, self.recent_attention_pool, self.scale_gate])
        for module in modules:
            if module is None:
                continue
            yield from module.parameters()

    def set_sequence_trainable(self, trainable: bool) -> None:
        for parameter in self.sequence_parameters():
            parameter.requires_grad = trainable

    def set_frozen_sequence_eval(self) -> None:
        for module in [self.encoder, self.position, self.sequence, self.attention_pool, self.classifier]:
            module.eval()

    def _recent_sequence(self, encoded: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, hidden = encoded.shape
        lengths = mask.sum(dim=1)
        recent_length = min(self.recent_products, encoded.shape[1])
        offsets = torch.arange(recent_length, device=encoded.device).unsqueeze(0)
        starts = (lengths - recent_length).clamp_min(0).unsqueeze(1)
        indices = starts + offsets
        recent_mask = offsets < lengths.clamp_max(recent_length).unsqueeze(1)
        recent = encoded.gather(1, indices.unsqueeze(-1).expand(batch, recent_length, hidden))
        return recent, recent_mask

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        tabular: torch.Tensor | None = None,
        temporal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(values, mask)
        positions = torch.arange(values.shape[1], device=values.device)
        encoded = encoded + self.position(positions).unsqueeze(0)
        lengths = mask.sum(dim=1, keepdim=True)
        reverse_positions = (lengths - 1 - positions.unsqueeze(0)).clamp_min(0)
        if self.reverse_position is not None:
            encoded = encoded + self.reverse_position(reverse_positions)
        if self.architecture == "multiscale_transformer":
            recent, recent_mask = self._recent_sequence(encoded, mask)
        embedding_encoded = encoded
        final_state = None
        if self.architecture in {"gru", "alpha_gru"}:
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
            packed_output, final_state = self.sequence(packed)  # type: ignore[arg-type]
            encoded, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=mask.shape[1])
        elif self.architecture in {"transformer", "multiscale_transformer"}:
            encoded = self.sequence(encoded, src_key_padding_mask=~mask)  # type: ignore[call-arg]
        masked = encoded.masked_fill(~mask.unsqueeze(-1), 0)
        mean = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        if self.product_recency:
            attention = self.attention_pool(encoded, mask, reverse_positions)  # type: ignore[call-arg]
        else:
            attention = self.attention_pool(encoded, mask)
        sequence_parts = [mean, maximum, attention]
        if self.architecture == "alpha_gru":
            embedding_masked = embedding_encoded.masked_fill(~mask.unsqueeze(-1), 0)
            embedding_mean = embedding_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
            embedding_maximum = embedding_encoded.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
            embedding_attention = self.embedding_attention_pool(embedding_encoded, mask)  # type: ignore[operator]
            if final_state is None:
                raise RuntimeError("GRU final state is required for alpha_gru")
            flattened_state = final_state.transpose(0, 1).reshape(final_state.shape[1], -1)
            projected_state = self.final_state_projection(flattened_state)  # type: ignore[operator]
            sequence_parts.extend([embedding_mean, embedding_maximum, embedding_attention, projected_state])
        if self.architecture == "multiscale_transformer":
            recent_positions = torch.arange(recent.shape[1], device=values.device)
            recent = recent + self.recent_position(recent_positions).unsqueeze(0)  # type: ignore[operator]
            recent = self.recent_sequence(recent, src_key_padding_mask=~recent_mask)  # type: ignore[operator]
            recent_masked = recent.masked_fill(~recent_mask.unsqueeze(-1), 0)
            recent_mean = recent_masked.sum(dim=1) / recent_mask.sum(dim=1, keepdim=True).clamp_min(1)
            recent_maximum = recent.masked_fill(~recent_mask.unsqueeze(-1), -1e4).max(dim=1).values
            recent_attention = self.recent_attention_pool(recent, recent_mask)  # type: ignore[operator]
            full = torch.stack(sequence_parts, dim=1)
            recent_parts = torch.stack([recent_mean, recent_maximum, recent_attention], dim=1)
            gates = torch.sigmoid(self.scale_gate(torch.cat([*sequence_parts, recent_mean, recent_maximum, recent_attention], dim=-1)))  # type: ignore[operator]
            fused = gates.unsqueeze(-1) * recent_parts + (1 - gates.unsqueeze(-1)) * full
            sequence_parts = list(fused.unbind(dim=1))
        temporal_encoded = None
        if self.temporal_encoder is not None:
            if temporal is None:
                raise RuntimeError("Temporal input is required when temporal features are enabled")
            temporal_encoded = self.temporal_encoder(temporal)
            sequence_parts.append(temporal_encoded)
        if self.tabular_encoder is not None:
            if tabular is None:
                raise RuntimeError("Tabular input is required for dual-branch model")
            tabular_encoded = self.tabular_encoder(tabular)
            if self.fusion_mode == "concat":
                return self.classifier(torch.cat([*sequence_parts, tabular_encoded], dim=-1)).squeeze(-1)
            if self.fusion_mode == "late":
                sequence_logit = self.classifier(torch.cat(sequence_parts, dim=-1)).squeeze(-1)
                tabular_logit = self.tabular_classifier(tabular_encoded).squeeze(-1)  # type: ignore[operator]
                weight = torch.sigmoid(self.fusion_logit)  # type: ignore[arg-type]
                return (1 - weight) * sequence_logit + weight * tabular_logit
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        return self.classifier(torch.cat(sequence_parts, dim=-1)).squeeze(-1)


def load_metadata(cache_dir: Path, split: str = "train") -> dict[str, object]:
    return json.loads((cache_dir / split / "metadata.json").read_text(encoding="utf-8"))

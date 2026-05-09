from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cluster.data.loader import load_split_indices, load_track_index, resolve_music_data_dir
from cluster.utils import (
    ArrayCache,
    apply_scale,
    fit_scaler_state,
    get_global_cache,
    load_optional_array,
    load_required_array,
    normalize_split_key,
    safe_load_json,
    set_seed,
)
from cluster.features.va_geometry import VA_GEOMETRY_FEATURE_NAMES, build_va_geometry_features


def _resolve_split_indices(data_dir: str, split_protocol: str, split: str, n_samples: int) -> np.ndarray:
    split = normalize_split_key(split)
    if split == "all":
        return np.arange(n_samples, dtype=np.int64)
    split_indices = load_split_indices(data_dir, split_protocol)
    if split not in split_indices:
        raise ValueError(f"Unknown split '{split}'. Available: {sorted(split_indices.keys())}.")
    return split_indices[split]


def _canonical_metadata_frame(data_dir: str, indices: np.ndarray) -> pd.DataFrame:
    path = os.path.join(data_dir, "canonical_metadata.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.iloc[indices].reset_index(drop=True)


@dataclass
class MusicDiscoveryDatasetArtifacts:
    train_dataset: "MusicDiscoveryDataset"
    val_dataset: "MusicDiscoveryDataset"
    test_dataset: "MusicDiscoveryDataset"
    all_dataset: "MusicDiscoveryDataset"
    scaler_state: Dict[str, Dict[str, List[float]]]
    metadata_feature_names: List[str]


class MusicDiscoveryDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        split_protocol: str,
        scaler_state: Dict[str, Dict[str, List[float]]],
        *,
        cache: Optional[ArrayCache] = None,
    ) -> None:
        self.data_dir = resolve_music_data_dir(data_dir)
        self.split = normalize_split_key(split)
        self.split_protocol = split_protocol
        self.meta = safe_load_json(os.path.join(self.data_dir, "meta.json"))

        _cache = cache or get_global_cache()
        audio = load_required_array(self.data_dir, "audio", np.float32, cache=_cache)
        lyrics = load_required_array(self.data_dir, "lyrics", np.float32, cache=_cache)
        metadata = load_required_array(self.data_dir, "metadata", np.float32, cache=_cache)
        metadata_recon_target = load_optional_array(self.data_dir, "metadata_binary", np.float32, cache=_cache)
        view_mask = load_optional_array(self.data_dir, "view_mask", np.float32, cache=_cache)
        consistency = load_required_array(self.data_dir, "consistency", np.float32, cache=_cache)
        va_diff = load_required_array(self.data_dir, "va_diff", np.float32, cache=_cache)
        labels = load_required_array(self.data_dir, "labels_emotion", np.int64, cache=_cache)
        original_va = load_optional_array(self.data_dir, "original_va", np.float32, cache=_cache)
        diff_observed = load_optional_array(self.data_dir, "diff_observed", np.float32, cache=_cache)
        n_samples = int(audio.shape[0])

        if not (lyrics.shape[0] == metadata.shape[0] == consistency.shape[0] == va_diff.shape[0] == labels.shape[0] == n_samples):
            raise ValueError("Processed discovery arrays do not share the same number of samples.")
        if metadata_recon_target is None:
            metadata_recon_target = metadata
        if metadata_recon_target.shape[0] != n_samples:
            raise ValueError(
                f"metadata_binary.npy must have {n_samples} rows, got {metadata_recon_target.shape[0]}."
            )
        if view_mask is None:
            view_mask = np.ones((n_samples, 3), dtype=np.float32)
        if view_mask.shape != (n_samples, 3):
            raise ValueError(f"view_mask.npy must have shape [{n_samples}, 3], got {view_mask.shape}.")
        if original_va is None:
            original_va = 0.5 * (audio + lyrics)
        if original_va.shape != (n_samples, 2):
            raise ValueError(f"original_va.npy must have shape [{n_samples}, 2], got {original_va.shape}.")
        if diff_observed is None:
            diff_observed = np.zeros((n_samples, 1), dtype=np.float32)
        if diff_observed.ndim == 1:
            diff_observed = diff_observed.reshape(-1, 1)
        if diff_observed.shape[0] != n_samples:
            raise ValueError(f"diff_observed.npy must have {n_samples} rows, got {diff_observed.shape[0]}.")

        indices = _resolve_split_indices(
            data_dir=self.data_dir,
            split_protocol=self.split_protocol,
            split=self.split,
            n_samples=n_samples,
        )
        track_index = load_track_index(self.data_dir)
        self.identifiers = [track_index["identifier"][int(idx)] for idx in indices.tolist()]
        self.lyric_identifiers = [track_index["lyric_identifier"][int(idx)] for idx in indices.tolist()]
        self.canonical_metadata = _canonical_metadata_frame(self.data_dir, indices)

        self.raw_audio = audio[indices].astype(np.float32)
        self.raw_lyrics = lyrics[indices].astype(np.float32)
        self.raw_metadata = metadata[indices].astype(np.float32)
        self.raw_metadata_recon_target = metadata_recon_target[indices].astype(np.float32)
        self.raw_metadata_report = self.raw_metadata_recon_target
        self.view_mask = view_mask[indices].astype(np.float32)
        self.consistency = consistency[indices].astype(np.float32).reshape(-1)
        self.va_diff = va_diff[indices].astype(np.float32)
        self.labels = labels[indices].astype(np.int64)
        self.original_va = original_va[indices].astype(np.float32)

        # Diff geometry features (computed eagerly from raw VAs)
        from cluster.features.diff_geometry import build_diff_geometry_features
        self.diff_geometry, self.diff_observed = build_diff_geometry_features(
            self.raw_audio, self.raw_lyrics, self.view_mask,
        )

        self.audio = apply_scale(self.raw_audio, scaler_state, "audio")
        self.lyrics = apply_scale(self.raw_lyrics, scaler_state, "lyrics")
        self.metadata = apply_scale(self.raw_metadata, scaler_state, "metadata")
        self.audio[self.view_mask[:, 0] <= 0.0] = 0.0
        self.lyrics[self.view_mask[:, 1] <= 0.0] = 0.0
        self.metadata[self.view_mask[:, 2] <= 0.0] = 0.0

    def __len__(self) -> int:
        return int(self.audio.shape[0])

    def _mean_va(self, idx: int) -> np.ndarray:
        weights = self.view_mask[idx, :2].astype(np.float32)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            return np.asarray([0.5, 0.5], dtype=np.float32)
        mean_va = (self.raw_audio[idx] * weights[0] + self.raw_lyrics[idx] * weights[1]) / total_weight
        return mean_va.astype(np.float32)

    def _signed_va_diff(self, idx: int) -> np.ndarray:
        if self.view_mask[idx, 0] <= 0.0 or self.view_mask[idx, 1] <= 0.0:
            return np.zeros(2, dtype=np.float32)
        return (self.raw_audio[idx] - self.raw_lyrics[idx]).astype(np.float32)

    def _va_geometry(self, idx: int) -> np.ndarray:
        return build_va_geometry_features(
            self.raw_audio[idx : idx + 1],
            self.raw_lyrics[idx : idx + 1],
            self.view_mask[idx : idx + 1],
        )[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "audio": torch.from_numpy(self.audio[idx]),
            "lyrics": torch.from_numpy(self.lyrics[idx]),
            "metadata": torch.from_numpy(self.metadata[idx]),
            "metadata_recon_target": torch.from_numpy(self.raw_metadata_recon_target[idx]),
            "view_mask": torch.from_numpy(self.view_mask[idx]),
            "consistency": torch.tensor(self.consistency[idx], dtype=torch.float32),
            "va_diff": torch.from_numpy(self.va_diff[idx]),
            "mean_va": torch.from_numpy(self._mean_va(idx)),
            "signed_va_diff": torch.from_numpy(self._signed_va_diff(idx)),
            "va_geometry": torch.from_numpy(self._va_geometry(idx)),
            "diff_geometry": torch.from_numpy(self.diff_geometry[idx]),
            "diff_observed": torch.tensor(self.diff_observed[idx, 0], dtype=torch.float32),
            "original_va": torch.from_numpy(self.original_va[idx]),
            "label_ref": int(self.labels[idx]),
            "label_emotion": int(self.labels[idx]),
            "identifier": self.identifiers[idx],
            "lyric_identifier": self.lyric_identifiers[idx],
            "track_id": self.identifiers[idx],
            "split": self.split,
        }
        return item


def create_music_discovery_datasets(
    data_dir: str,
    split_protocol: str,
    scaler_state: Optional[Dict[str, Dict[str, List[float]]]] = None,
) -> MusicDiscoveryDatasetArtifacts:
    resolved_dir = resolve_music_data_dir(data_dir)
    if scaler_state is None:
        scaler_state = fit_scaler_state(
            resolved_dir,
            split_protocol,
            ["audio", "lyrics", "metadata"],
        )
    names_path = os.path.join(resolved_dir, "metadata_binary_feature_names.json")
    if not os.path.exists(names_path):
        names_path = os.path.join(resolved_dir, "metadata_feature_names.json")
    if os.path.exists(names_path):
        with open(names_path, "r", encoding="utf-8") as f:
            metadata_feature_names = [str(item) for item in json.load(f)]
    else:
        metadata = load_required_array(resolved_dir, "metadata", np.float32)
        metadata_feature_names = [f"metadata::{idx}" for idx in range(metadata.shape[1])]

    cache = ArrayCache()
    return MusicDiscoveryDatasetArtifacts(
        train_dataset=MusicDiscoveryDataset(resolved_dir, "train", split_protocol, scaler_state, cache=cache),
        val_dataset=MusicDiscoveryDataset(resolved_dir, "val", split_protocol, scaler_state, cache=cache),
        test_dataset=MusicDiscoveryDataset(resolved_dir, "test", split_protocol, scaler_state, cache=cache),
        all_dataset=MusicDiscoveryDataset(resolved_dir, "all", split_protocol, scaler_state, cache=cache),
        scaler_state=scaler_state,
        metadata_feature_names=metadata_feature_names,
    )


def create_music_discovery_loader(
    dataset: MusicDiscoveryDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=max(int(num_workers), 0),
        pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
    )


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class _ViewAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.0,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        decoder_output_dim = int(input_dim if output_dim is None else output_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, decoder_output_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class DiffEncoder(nn.Module):
    """Encodes diff geometry features (26-dim) into latent space.

    Output is gated by diff_observed so audio-only or lyrics-only samples
    produce zero vectors, preventing missingness leakage.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, int(latent_dim)),
        )

    def forward(self, diff_features: torch.Tensor, diff_observed: torch.Tensor) -> torch.Tensor:
        z = self.net(diff_features)
        gate = diff_observed.to(device=z.device, dtype=z.dtype).reshape(-1, 1)
        return z * gate


class VAHead(nn.Module):
    """Predicts consensus VA coordinates from a latent vector."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SignedDiffHead(nn.Module):
    """Predicts signed audio-minus-lyrics VA delta from a latent vector."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DECClusterHead(nn.Module):
    def __init__(self, n_clusters: int, latent_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        n_clusters = int(n_clusters)
        if n_clusters <= 1:
            raise ValueError("DEC cluster head requires n_clusters > 1.")
        temperature = float(temperature)
        if temperature <= 0.0:
            raise ValueError("DEC cluster temperature must be positive.")
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, int(latent_dim)))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squared_distances = torch.sum(
            (z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2,
            dim=-1,
        )
        return torch.softmax(-squared_distances / self.temperature, dim=1)


class MusicMetadataDiscoveryNet(nn.Module):
    def __init__(
        self,
        audio_dim: int,
        lyrics_dim: int,
        metadata_dim: int,
        metadata_recon_dim: Optional[int] = None,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        metadata_hidden_dim: int = 128,
        gate_hidden_dim: int = 128,
        metadata_aux_scale: float = 0.60,
        dropout: float = 0.0,
        metadata_logit_offset: float = 0.0,
        cluster_head_k: int = 0,
        cluster_temperature: float = 1.0,
        gate_context_dim: int = len(VA_GEOMETRY_FEATURE_NAMES) + 3,
        diff_input_dim: int = 26,
    ) -> None:
        super().__init__()
        cluster_head_k = int(cluster_head_k)
        self.gate_context_dim = int(gate_context_dim)
        self.metadata_recon_dim = int(metadata_dim if metadata_recon_dim is None else metadata_recon_dim)
        self.audio_view = _ViewAutoencoder(audio_dim, hidden_dim, latent_dim, dropout=dropout)
        self.lyrics_view = _ViewAutoencoder(lyrics_dim, hidden_dim, latent_dim, dropout=dropout)
        self.metadata_view = _ViewAutoencoder(
            metadata_dim,
            metadata_hidden_dim,
            latent_dim,
            dropout=dropout,
            output_dim=self.metadata_recon_dim,
        )
        self.fused_audio_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, audio_dim),
        )
        self.fused_lyrics_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lyrics_dim),
        )
        # Independent projectors per view (no shared projector)
        self.proj_audio = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.proj_lyrics = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.proj_metadata = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.proj_fused = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        gate_input_dim = latent_dim * 3 + self.gate_context_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim // 2, 3),
        )
        self.metadata_aux_scale = float(metadata_aux_scale)
        self.metadata_logit_offset = float(metadata_logit_offset)
        self.cluster_head_k = cluster_head_k
        self.cluster_temperature = float(cluster_temperature)
        self.cluster_head = (
            DECClusterHead(cluster_head_k, latent_dim, temperature=self.cluster_temperature)
            if cluster_head_k > 0
            else None
        )
        self.diff_encoder = DiffEncoder(
            input_dim=diff_input_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.cluster_fusion_mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.va_head = VAHead(latent_dim=latent_dim)
        self.signed_diff_head = SignedDiffHead(latent_dim=latent_dim)
        # Gate weight initialization: Xavier for learning, asymmetric bias
        for layer in self.gate_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        final_linear = self.gate_mlp[-1]
        with torch.no_grad():
            # Asymmetric bias: audio slightly favoured over lyrics,
            # metadata suppressed — breaks symmetry so gate can specialise.
            final_linear.bias.copy_(torch.tensor([0.3, 0.0, -0.5], dtype=final_linear.bias.dtype))

    def forward(
        self,
        audio: torch.Tensor,
        lyrics: torch.Tensor,
        metadata: torch.Tensor,
        consistency: torch.Tensor,
        va_diff: torch.Tensor,
        va_geometry: Optional[torch.Tensor] = None,
        view_mask: Optional[torch.Tensor] = None,
        diff_features: Optional[torch.Tensor] = None,
        diff_observed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if view_mask is None:
            view_mask = torch.ones((audio.shape[0], 3), dtype=audio.dtype, device=audio.device)
        view_mask = view_mask.to(device=audio.device, dtype=audio.dtype)
        if view_mask.ndim != 2 or view_mask.shape[1] != 3:
            raise ValueError(f"Expected view_mask shape [B, 3], got {tuple(view_mask.shape)}.")
        # Avoid all -inf softmax rows if an upstream row has no available view.
        no_view = view_mask.sum(dim=1, keepdim=True) <= 0.0
        if bool(no_view.any()):
            view_mask = torch.where(no_view, torch.ones_like(view_mask), view_mask)

        z_audio = self.audio_view.encode(audio)
        z_lyrics = self.lyrics_view.encode(lyrics)
        z_metadata = self.metadata_view.encode(metadata)

        audio_recon = self.audio_view.decode(z_audio)
        lyrics_recon = self.lyrics_view.decode(z_lyrics)
        metadata_recon = self.metadata_view.decode(z_metadata)

        if consistency.ndim == 1:
            consistency = consistency.unsqueeze(1)
        if va_diff.ndim != 2 or va_diff.shape[1] != 2:
            raise ValueError(f"Expected va_diff shape [B, 2], got {tuple(va_diff.shape)}.")
        if va_geometry is None:
            va_geometry = torch.zeros(
                (audio.shape[0], len(VA_GEOMETRY_FEATURE_NAMES)),
                dtype=audio.dtype,
                device=audio.device,
            )
        va_geometry = va_geometry.to(device=audio.device, dtype=audio.dtype)
        if va_geometry.ndim != 2 or va_geometry.shape[1] != len(VA_GEOMETRY_FEATURE_NAMES):
            raise ValueError(
                f"Expected va_geometry shape [B, {len(VA_GEOMETRY_FEATURE_NAMES)}], got {tuple(va_geometry.shape)}."
            )

        if self.gate_context_dim == 6:
            gate_context = torch.cat([view_mask, consistency, torch.abs(va_diff)], dim=1)
        elif self.gate_context_dim == len(VA_GEOMETRY_FEATURE_NAMES) + 3:
            gate_context = torch.cat([view_mask, va_geometry], dim=1)
        else:
            raise ValueError(f"Unsupported gate_context_dim={self.gate_context_dim}.")

        gate_input = torch.cat(
            [
                z_audio,
                z_lyrics,
                z_metadata,
                gate_context,
            ],
            dim=1,
        )
        gate_logits = self.gate_mlp(gate_input)
        # Logit offset for metadata suppression (cleaner than post-softmax multiply)
        if self.metadata_logit_offset != 0.0:
            logit_offset = gate_logits.new_tensor([0.0, 0.0, self.metadata_logit_offset])
            gate_logits = gate_logits + logit_offset
        gate_logits = gate_logits.masked_fill(view_mask <= 0.0, -1e9)
        gate_weights = torch.softmax(gate_logits, dim=1)
        if self.metadata_logit_offset == 0.0 and self.metadata_aux_scale != 1.0:
            gate_weights = gate_weights * gate_weights.new_tensor(
                [1.0, 1.0, self.metadata_aux_scale]
            ).view(1, 3)
            gate_weights = gate_weights * view_mask
            gate_weights = gate_weights / torch.clamp(gate_weights.sum(dim=1, keepdim=True), min=1e-8)

        fused = (
            gate_weights[:, 0:1] * z_audio
            + gate_weights[:, 1:2] * z_lyrics
            + gate_weights[:, 2:3] * z_metadata
        )
        fused_audio_recon = self.fused_audio_decoder(fused)
        fused_lyrics_recon = self.fused_lyrics_decoder(fused)

        # Diff encoder: encode pairwise geometry
        if diff_features is None:
            diff_features = torch.zeros(
                (audio.shape[0], self.diff_encoder.net[0].in_features),
                dtype=audio.dtype, device=audio.device,
            )
        if diff_observed is None:
            diff_observed = torch.zeros(audio.shape[0], dtype=audio.dtype, device=audio.device)
        z_diff = self.diff_encoder(diff_features, diff_observed)
        z_cluster = self.cluster_fusion_mlp(torch.cat([fused, z_diff, z_metadata], dim=1))
        z_affect = z_cluster if self.cluster_head is not None else fused

        # VA head: predict consensus VA from fused latent
        va_pred = self.va_head(fused)
        signed_diff_pred = self.signed_diff_head(z_diff)
        pair_signed_diff_pred = self.signed_diff_head(z_audio - z_lyrics)

        outputs = {
            "z_audio": z_audio,
            "z_lyrics": z_lyrics,
            "z_metadata": z_metadata,
            "z_fused": fused,
            "z_consensus": fused,
            "z_tension": z_diff,
            "z_cluster": z_cluster,
            "z_affect": z_affect,
            "audio_recon": audio_recon,
            "lyrics_recon": lyrics_recon,
            "metadata_recon": metadata_recon,
            "fused_audio_recon": fused_audio_recon,
            "fused_lyrics_recon": fused_lyrics_recon,
            "proj_audio": self.proj_audio(z_audio),
            "proj_lyrics": self.proj_lyrics(z_lyrics),
            "proj_metadata": self.proj_metadata(z_metadata),
            "proj_fused": self.proj_fused(fused),
            "gate_weights": gate_weights,
            "z_diff": z_diff,
            "va_pred": va_pred,
            "signed_diff_pred": signed_diff_pred,
            "pair_signed_diff_pred": pair_signed_diff_pred,
        }
        if self.cluster_head is not None:
            outputs.update(
                {
                    "q_audio": self.cluster_head(z_audio),
                    "q_lyrics": self.cluster_head(z_lyrics),
                    "q_metadata": self.cluster_head(z_metadata),
                    "q_fused": self.cluster_head(z_affect),
                }
            )
        return outputs


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def target_distribution(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q = torch.clamp(q, min=float(eps))
    frequency = torch.clamp(q.sum(dim=0, keepdim=True), min=float(eps))
    weight = (q ** 2) / frequency
    return weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=float(eps))


def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(x, y, dim=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype).reshape(-1)
    values = values.reshape(-1)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (values * mask).sum() / denom


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=pred.dtype).reshape(-1)
    per_sample = torch.mean((pred - target) ** 2, dim=1)
    return _masked_mean(per_sample, mask)


def _masked_bce_with_logits(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=pred.dtype).reshape(-1)
    target = target.to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        raise ValueError(
            f"metadata BCE target shape {tuple(target.shape)} does not match logits shape {tuple(pred.shape)}."
        )
    with torch.no_grad():
        pos = target.sum(dim=0)
        neg = target.shape[0] - pos
        pos_weight = torch.clamp((neg + 1.0) / (pos + 1.0), min=1.0, max=25.0)
    loss = F.binary_cross_entropy_with_logits(
        pred,
        target,
        pos_weight=pos_weight.to(device=pred.device, dtype=pred.dtype),
        reduction="none",
    ).mean(dim=1)
    return _masked_mean(loss, mask)


def _masked_gate_balance_loss(gate: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
    available = (view_mask.sum(dim=0) > 0).to(dtype=gate.dtype)
    available_count = torch.clamp(available.sum(), min=1.0)
    mean_gate = (gate * view_mask).sum(dim=0) / torch.clamp(view_mask.sum(dim=0), min=1.0)
    mean_gate = mean_gate * available
    mean_gate = mean_gate / torch.clamp(mean_gate.sum(), min=1e-8)
    uniform = available / available_count
    return torch.sum(mean_gate * (torch.log(mean_gate + 1e-8) - torch.log(uniform + 1e-8)))


def _kl_per_sample(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    target = torch.clamp(target, min=float(eps))
    pred = torch.clamp(pred, min=float(eps))
    return torch.sum(target * (torch.log(target) - torch.log(pred)), dim=1)


def _kl_batchmean(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return _kl_per_sample(target, pred).mean()


def _masked_kl_mean(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(_kl_per_sample(target, pred), mask)


def _assignment_balance_loss(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean_assignment = torch.clamp(q.mean(dim=0), min=float(eps))
    mean_assignment = mean_assignment / torch.clamp(mean_assignment.sum(), min=float(eps))
    uniform = torch.full_like(mean_assignment, 1.0 / max(int(q.shape[1]), 1))
    return torch.sum(mean_assignment * (torch.log(mean_assignment) - torch.log(uniform + eps)))


def _discovery_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    metadata_recon_weight: float,
    fused_recon_weight: float,
    align_weight: float,
    metadata_align_weight: float,
    gate_entropy_weight: float = 0.0,
    cluster_loss_weight: float = 0.0,
    cvcl_loss_weight: float = 0.0,
    assignment_balance_weight: float = 0.0,
    consensus_va_weight: float = 0.0,
    diff_preserve_weight: float = 0.0,
    metadata_recon_loss: str = "mse",
) -> Dict[str, torch.Tensor]:
    audio = batch["audio"]
    lyrics = batch["lyrics"]
    metadata = batch["metadata"]
    consistency = batch["consistency"].reshape(-1)
    view_mask = batch.get("view_mask")
    if view_mask is None:
        view_mask = torch.ones((audio.shape[0], 3), dtype=audio.dtype, device=audio.device)
    view_mask = view_mask.to(device=audio.device, dtype=audio.dtype)
    audio_mask = view_mask[:, 0]
    lyrics_mask = view_mask[:, 1]
    metadata_mask = view_mask[:, 2]

    recon_audio = _masked_mse(outputs["audio_recon"], audio, audio_mask)
    recon_lyrics = _masked_mse(outputs["lyrics_recon"], lyrics, lyrics_mask)
    metadata_loss_mode = str(metadata_recon_loss or "mse").strip().lower()
    metadata_target = batch.get("metadata_recon_target", metadata)
    if metadata_loss_mode == "bce":
        recon_metadata = _masked_bce_with_logits(outputs["metadata_recon"], metadata_target, metadata_mask)
    elif metadata_loss_mode == "mse":
        recon_metadata = _masked_mse(outputs["metadata_recon"], metadata_target, metadata_mask)
    else:
        raise ValueError("metadata_recon_loss must be 'mse' or 'bce'.")
    fused_recon = 0.5 * (
        _masked_mse(outputs["fused_audio_recon"], audio, audio_mask)
        + _masked_mse(outputs["fused_lyrics_recon"], lyrics, lyrics_mask)
    )

    audio_lyrics_mask = audio_mask * lyrics_mask
    audio_metadata_mask = audio_mask * metadata_mask
    lyrics_metadata_mask = lyrics_mask * metadata_mask
    signed_va_diff = batch.get("signed_va_diff")
    align_agreement = consistency
    if signed_va_diff is not None:
        signed_va_diff = signed_va_diff.to(device=audio.device, dtype=audio.dtype)
        if signed_va_diff.ndim == 2 and signed_va_diff.shape[1] == 2:
            diff_gap = torch.norm(signed_va_diff, dim=1)
            # Similar audio/lyrics pairs should align; large signed disagreements
            # should be allowed to remain separated so diff information survives.
            align_agreement = torch.exp(-3.0 * diff_gap)
    align_audio_lyrics = _masked_mean(
        align_agreement * _cosine_distance(outputs["proj_audio"], outputs["proj_lyrics"]),
        audio_lyrics_mask,
    )
    align_audio_metadata = _masked_mean(
        _cosine_distance(outputs["proj_audio"], outputs["proj_metadata"]),
        audio_metadata_mask,
    )
    align_lyrics_metadata = _masked_mean(
        _cosine_distance(outputs["proj_lyrics"], outputs["proj_metadata"]),
        lyrics_metadata_mask,
    )
    align_fused_audio = _masked_mean(_cosine_distance(outputs["proj_fused"], outputs["proj_audio"]), audio_mask)
    align_fused_lyrics = _masked_mean(_cosine_distance(outputs["proj_fused"], outputs["proj_lyrics"]), lyrics_mask)

    # Consensus VA preservation: keep z_fused grounded in VA space
    consensus_loss = audio.new_tensor(0.0)
    diff_preserve_loss = audio.new_tensor(0.0)
    if consensus_va_weight > 0:
        mean_va = batch.get("mean_va")
        if mean_va is not None:
            va_pred = outputs["va_pred"]
            any_view_mask = (audio_mask + lyrics_mask) > 0.0
            consensus_loss = _masked_mse(va_pred, mean_va, any_view_mask)

    # Disagreement preservation: keep latent distance proportional to VA distance
    if diff_preserve_weight > 0:
        diff_obs = batch.get("diff_observed")
        if signed_va_diff is not None and diff_obs is not None:
            signed_va_diff = signed_va_diff.to(device=audio.device, dtype=audio.dtype)
            diff_mask = (diff_obs > 0.0).to(device=audio.device, dtype=audio.dtype)
            if diff_mask.any():
                z_dist = torch.norm(outputs["z_audio"] - outputs["z_lyrics"], dim=1)
                z_diff_dist = torch.norm(outputs["z_diff"], dim=1)
                va_dist = torch.norm(signed_va_diff, dim=1)
                alpha = 0.5
                latent_pair_loss = F.smooth_l1_loss(z_dist, alpha * va_dist, reduction="none")
                diff_latent_loss = F.smooth_l1_loss(z_diff_dist, alpha * va_dist, reduction="none")
                magnitude_loss = 0.5 * (latent_pair_loss + diff_latent_loss)
                signed_terms = []
                nonzero_target = (va_dist > 1e-6).to(dtype=audio.dtype)
                for pred_key in ("signed_diff_pred", "pair_signed_diff_pred"):
                    pred = outputs.get(pred_key)
                    if pred is None:
                        continue
                    vector_loss = F.smooth_l1_loss(
                        pred,
                        signed_va_diff,
                        reduction="none",
                    ).mean(dim=1)
                    direction_loss = (
                        1.0
                        - F.cosine_similarity(pred, signed_va_diff, dim=1, eps=1e-8)
                    ) * nonzero_target
                    signed_terms.append(vector_loss + 0.25 * direction_loss)
                if signed_terms:
                    signed_loss = torch.stack(signed_terms, dim=0).mean(dim=0)
                    per_sample_diff_loss = 0.65 * signed_loss + 0.35 * magnitude_loss
                else:
                    per_sample_diff_loss = magnitude_loss
                diff_preserve_loss = _masked_mean(
                    per_sample_diff_loss,
                    diff_mask,
                )

    total = (
        recon_audio
        + recon_lyrics
        + metadata_recon_weight * recon_metadata
        + fused_recon_weight * fused_recon
        + align_weight * align_audio_lyrics
        + metadata_align_weight * (0.5 * (align_audio_metadata + align_lyrics_metadata))
        + 0.5 * align_weight * (align_fused_audio + align_fused_lyrics)
    )

    gate = outputs["gate_weights"]
    gate_entropy = -(gate * (gate + 1e-8).log()).sum(dim=-1).mean()
    if gate_entropy_weight > 0:
        total = total + gate_entropy_weight * _masked_gate_balance_loss(gate, view_mask)

    if consensus_va_weight > 0:
        total = total + consensus_va_weight * consensus_loss
    if diff_preserve_weight > 0:
        total = total + diff_preserve_weight * diff_preserve_loss

    zero = total.new_tensor(0.0)
    cluster_loss = zero
    cvcl_loss = zero
    assignment_balance = zero
    q_fused = outputs.get("q_fused")
    if q_fused is not None:
        cluster_target = target_distribution(q_fused.detach())
        cluster_loss = _kl_batchmean(cluster_target, q_fused)
        assignment_balance = _assignment_balance_loss(q_fused)

        cvcl_terms = []
        for key, mask in (
            ("q_audio", audio_mask),
            ("q_lyrics", lyrics_mask),
            ("q_metadata", metadata_mask),
        ):
            if key in outputs:
                cvcl_terms.append(_masked_kl_mean(q_fused.detach(), outputs[key], mask))
        if cvcl_terms:
            cvcl_loss = torch.stack(cvcl_terms).mean()

        total = (
            total
            + float(cluster_loss_weight) * cluster_loss
            + float(cvcl_loss_weight) * cvcl_loss
            + float(assignment_balance_weight) * assignment_balance
        )

    return {
        "loss": total,
        "recon_audio": recon_audio.detach(),
        "recon_lyrics": recon_lyrics.detach(),
        "recon_metadata": recon_metadata.detach(),
        "fused_recon": fused_recon.detach(),
        "align_audio_lyrics": align_audio_lyrics.detach(),
        "align_audio_metadata": align_audio_metadata.detach(),
        "align_lyrics_metadata": align_lyrics_metadata.detach(),
        "gate_entropy": gate_entropy.detach(),
        "cluster_loss": cluster_loss.detach(),
        "cvcl_loss": cvcl_loss.detach(),
        "assignment_balance": assignment_balance.detach(),
        "consensus_va": consensus_loss.detach(),
        "diff_preserve": diff_preserve_loss.detach(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, dtype=torch.float32 if value.dtype != torch.int64 else value.dtype)
        else:
            moved[key] = value
    return moved


class EarlyStopping:
    """Early stopping tracker based on validation loss."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _run_epoch(
    model: MusicMetadataDiscoveryNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    metadata_recon_weight: float,
    fused_recon_weight: float,
    align_weight: float,
    metadata_align_weight: float,
    gate_entropy_weight: float = 0.0,
    cluster_loss_weight: float = 0.0,
    cvcl_loss_weight: float = 0.0,
    assignment_balance_weight: float = 0.0,
    consensus_va_weight: float = 0.0,
    diff_preserve_weight: float = 0.0,
    metadata_recon_loss: str = "mse",
    *,
    grad_clip_norm: float = 0.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    totals: Dict[str, float] = {}
    total_samples = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
            outputs = model(
                audio=batch["audio"],
                lyrics=batch["lyrics"],
                metadata=batch["metadata"],
                consistency=batch["consistency"],
                va_diff=batch["va_diff"],
                va_geometry=batch.get("va_geometry"),
                view_mask=batch.get("view_mask"),
                diff_features=batch.get("diff_geometry"),
                diff_observed=batch.get("diff_observed"),
            )
            losses = _discovery_loss(
                outputs=outputs,
                batch=batch,
                metadata_recon_weight=metadata_recon_weight,
                fused_recon_weight=fused_recon_weight,
                align_weight=align_weight,
                metadata_align_weight=metadata_align_weight,
                gate_entropy_weight=gate_entropy_weight,
                cluster_loss_weight=cluster_loss_weight,
                cvcl_loss_weight=cvcl_loss_weight,
                assignment_balance_weight=assignment_balance_weight,
                consensus_va_weight=consensus_va_weight,
                diff_preserve_weight=diff_preserve_weight,
                metadata_recon_loss=metadata_recon_loss,
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(losses["loss"]).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["loss"].backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

        batch_size = int(batch["audio"].shape[0])
        total_samples += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size

    return {key: value / max(total_samples, 1) for key, value in totals.items()}


def train_music_discovery_model(
    model: MusicMetadataDiscoveryNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    metadata_recon_weight: float,
    fused_recon_weight: float,
    align_weight: float,
    metadata_align_weight: float,
    gate_entropy_weight: float = 0.0,
    cluster_loss_weight: float = 0.0,
    cvcl_loss_weight: float = 0.0,
    assignment_balance_weight: float = 0.0,
    consensus_va_weight: float = 0.0,
    diff_preserve_weight: float = 0.0,
    metadata_recon_loss: str = "mse",
    grad_clip_norm: float = 0.0,
    use_amp: bool = False,
    early_stopping_patience: int = 0,
    scheduler_T0: int = 0,
    scheduler_Tmult: int = 2,
    scheduler_eta_min: float = 1e-6,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]], Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    # LR scheduler
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if scheduler_T0 > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(scheduler_T0),
            T_mult=int(scheduler_Tmult),
            eta_min=float(scheduler_eta_min),
        )

    # AMP scaler
    amp_scaler: Optional[torch.amp.GradScaler] = None
    if use_amp and device.type == "cuda":
        amp_scaler = torch.amp.GradScaler("cuda")

    # Early stopping
    stopper: Optional[EarlyStopping] = None
    if early_stopping_patience > 0:
        stopper = EarlyStopping(patience=int(early_stopping_patience))

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(int(epochs)):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            metadata_recon_weight=metadata_recon_weight,
            fused_recon_weight=fused_recon_weight,
            align_weight=align_weight,
            metadata_align_weight=metadata_align_weight,
            gate_entropy_weight=gate_entropy_weight,
            cluster_loss_weight=cluster_loss_weight,
            cvcl_loss_weight=cvcl_loss_weight,
            assignment_balance_weight=assignment_balance_weight,
            consensus_va_weight=consensus_va_weight,
            diff_preserve_weight=diff_preserve_weight,
            metadata_recon_loss=metadata_recon_loss,
            grad_clip_norm=grad_clip_norm,
            scaler=amp_scaler,
            use_amp=use_amp,
        )
        with torch.inference_mode():
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                metadata_recon_weight=metadata_recon_weight,
                fused_recon_weight=fused_recon_weight,
                align_weight=align_weight,
                metadata_align_weight=metadata_align_weight,
                gate_entropy_weight=gate_entropy_weight,
                cluster_loss_weight=cluster_loss_weight,
                cvcl_loss_weight=cvcl_loss_weight,
                assignment_balance_weight=assignment_balance_weight,
                consensus_va_weight=consensus_va_weight,
                diff_preserve_weight=diff_preserve_weight,
                metadata_recon_loss=metadata_recon_loss,
                use_amp=use_amp,
            )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": float(epoch + 1), "lr": current_lr}
        row.update({f"train_{key}": float(value) for key, value in train_metrics.items()})
        row.update({f"val_{key}": float(value) for key, value in val_metrics.items()})
        history.append(row)
        print(
            f"[Discovery][Train] epoch={epoch + 1}/{epochs} "
            f"lr={current_lr:.2e} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"train_align={train_metrics['align_audio_lyrics']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if stopper is not None and stopper.step(val_metrics["loss"]):
            print(f"[Discovery] Early stopping at epoch {epoch + 1} (patience={stopper.patience})")
            break

    if best_state is None:
        raise RuntimeError("Discovery training did not produce a valid checkpoint.")
    return best_state, history, {"best_val_loss": float(best_val)}


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_split_embeddings(
    model: MusicMetadataDiscoveryNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    blocks: Dict[str, List[np.ndarray]] = {
        "z_audio": [],
        "z_lyrics": [],
        "z_metadata": [],
        "z_fused": [],
        "gate_weights": [],
        "q_audio": [],
        "q_lyrics": [],
        "q_metadata": [],
        "q_fused": [],
        "consistency": [],
        "va_diff": [],
        "raw_label_id": [],
        "view_mask": [],
        "mean_va": [],
        "signed_va_diff": [],
        "va_geometry": [],
        "diff_geometry": [],
        "diff_observed": [],
        "original_va": [],
        "z_diff": [],
        "z_tension": [],
        "z_consensus": [],
        "z_cluster": [],
        "z_affect": [],
    }
    identifiers: List[str] = []
    lyric_identifiers: List[str] = []
    with torch.inference_mode():
        for batch in loader:
            identifiers.extend([str(item) for item in batch["identifier"]])
            lyric_identifiers.extend([str(item) for item in batch["lyric_identifier"]])
            batch = _move_batch_to_device(batch, device)
            outputs = model(
                audio=batch["audio"],
                lyrics=batch["lyrics"],
                metadata=batch["metadata"],
                consistency=batch["consistency"],
                va_diff=batch["va_diff"],
                va_geometry=batch.get("va_geometry"),
                view_mask=batch.get("view_mask"),
                diff_features=batch.get("diff_geometry"),
                diff_observed=batch.get("diff_observed"),
            )
            blocks["z_audio"].append(outputs["z_audio"].cpu().numpy().astype(np.float32))
            blocks["z_lyrics"].append(outputs["z_lyrics"].cpu().numpy().astype(np.float32))
            blocks["z_metadata"].append(outputs["z_metadata"].cpu().numpy().astype(np.float32))
            blocks["z_fused"].append(outputs["z_fused"].cpu().numpy().astype(np.float32))
            blocks["gate_weights"].append(outputs["gate_weights"].cpu().numpy().astype(np.float32))
            for key in ("q_audio", "q_lyrics", "q_metadata", "q_fused"):
                if key in outputs:
                    blocks[key].append(outputs[key].cpu().numpy().astype(np.float32))
            blocks["consistency"].append(batch["consistency"].cpu().numpy().astype(np.float32).reshape(-1, 1))
            blocks["va_diff"].append(batch["va_diff"].cpu().numpy().astype(np.float32))
            blocks["raw_label_id"].append(batch["label_ref"].cpu().numpy().astype(np.int64).reshape(-1, 1))
            blocks["view_mask"].append(batch["view_mask"].cpu().numpy().astype(np.float32))
            blocks["mean_va"].append(batch["mean_va"].cpu().numpy().astype(np.float32))
            blocks["signed_va_diff"].append(batch["signed_va_diff"].cpu().numpy().astype(np.float32))
            blocks["va_geometry"].append(batch["va_geometry"].cpu().numpy().astype(np.float32))
            blocks["diff_geometry"].append(batch["diff_geometry"].cpu().numpy().astype(np.float32))
            blocks["diff_observed"].append(batch["diff_observed"].cpu().numpy().astype(np.float32).reshape(-1, 1))
            blocks["original_va"].append(batch["original_va"].cpu().numpy().astype(np.float32))
            blocks["z_diff"].append(outputs["z_diff"].cpu().numpy().astype(np.float32))
            blocks["z_tension"].append(outputs["z_tension"].cpu().numpy().astype(np.float32))
            blocks["z_consensus"].append(outputs["z_consensus"].cpu().numpy().astype(np.float32))
            blocks["z_cluster"].append(outputs["z_cluster"].cpu().numpy().astype(np.float32))
            blocks["z_affect"].append(outputs["z_affect"].cpu().numpy().astype(np.float32))

    out: Dict[str, Any] = {
        key: np.concatenate(value, axis=0) if value else np.zeros((0, 0), dtype=np.float32)
        for key, value in blocks.items()
    }
    out["identifier"] = identifiers
    out["lyric_identifier"] = lyric_identifiers
    return out


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_discovery_checkpoint(
    model: MusicMetadataDiscoveryNet,
    checkpoint_path: str,
    *,
    scaler_state: Dict[str, Dict[str, List[float]]],
    config: Dict[str, Any],
    best_metrics: Dict[str, float],
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    amp_scaler_state: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    dataset_version: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    schema_hash: Optional[str] = None,
    metadata_schema: Optional[Dict[str, Any]] = None,
    rng_state: Optional[Dict[str, Any]] = None,
) -> None:
    directory = os.path.dirname(checkpoint_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if rng_state is None:
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
    }
    payload = {
        "format_version": 2,
        "model_state": model.state_dict(),
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "amp_scaler_state": amp_scaler_state,
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": best_metrics,
        "rng_state": rng_state,
        "config": config,
        "scaler_state": scaler_state,
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "schema_hash": schema_hash,
        "metadata_schema": metadata_schema,
        "runtime": runtime,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, checkpoint_path)
    sidecar_path = os.path.splitext(checkpoint_path)[0] + ".meta.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scaler_state": scaler_state,
                "config": config,
                "best_metrics": best_metrics,
                "epoch": epoch,
                "global_step": global_step,
                "dataset_version": dataset_version,
                "dataset_hash": dataset_hash,
                "schema_hash": schema_hash,
                "metadata_schema": metadata_schema,
                "runtime": runtime,
                "checkpoint_format_version": 2,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def initialize_discovery_runtime(seed: int, gpu: str) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    set_seed(int(seed))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_music_discovery_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[MusicMetadataDiscoveryNet, Dict[str, Any]]:
    sidecar_path = os.path.splitext(checkpoint_path)[0] + ".meta.json"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint '{checkpoint_path}'.")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Missing discovery checkpoint sidecar '{sidecar_path}'.")
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )

    scaler_state = sidecar.get("scaler_state", {})
    config = sidecar.get("config", {})
    audio_dim = len(scaler_state.get("audio", {}).get("mean", []))
    lyrics_dim = len(scaler_state.get("lyrics", {}).get("mean", []))
    metadata_dim = len(scaler_state.get("metadata", {}).get("mean", []))
    if min(audio_dim, lyrics_dim, metadata_dim) <= 0:
        raise ValueError("Checkpoint sidecar does not contain valid scaler_state dimensions.")
    latent_dim = int(config.get("latent_dim", 16))
    gate_weight = state_dict.get("gate_mlp.0.weight")
    inferred_gate_context_dim = None
    if torch.is_tensor(gate_weight):
        inferred_gate_context_dim = int(gate_weight.shape[1]) - latent_dim * 3

    model = MusicMetadataDiscoveryNet(
        audio_dim=int(audio_dim),
        lyrics_dim=int(lyrics_dim),
        metadata_dim=int(metadata_dim),
        metadata_recon_dim=int(config.get("metadata_recon_dim", metadata_dim)),
        latent_dim=latent_dim,
        hidden_dim=int(config.get("hidden_dim", 32)),
        metadata_hidden_dim=int(config.get("metadata_hidden_dim", 128)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 32)),
        metadata_aux_scale=float(config.get("metadata_aux_scale", 0.60)),
        dropout=float(config.get("dropout", 0.0)),
        metadata_logit_offset=float(config.get("metadata_logit_offset", 0.0)),
        cluster_head_k=int(config.get("cluster_head_k", 0)),
        cluster_temperature=float(config.get("cluster_temperature", 1.0)),
        gate_context_dim=int(config.get("gate_context_dim", inferred_gate_context_dim or len(VA_GEOMETRY_FEATURE_NAMES) + 3)),
        diff_input_dim=int(config.get("diff_input_dim", 26)),
    ).to(device)

    model_state = model.state_dict()
    incompatible_shape_keys = [
        key for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    ]
    if incompatible_shape_keys:
        state_dict = {
            key: value for key, value in state_dict.items()
            if key not in set(incompatible_shape_keys)
        }
    load_result = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = ("diff_encoder.", "va_head.", "signed_diff_head.", "cluster_fusion_mlp.")
    disallowed_missing = [
        key for key in load_result.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if disallowed_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint model_state is incompatible with MusicMetadataDiscoveryNet. "
            f"Missing keys: {disallowed_missing}; unexpected keys: {list(load_result.unexpected_keys)}."
        )
    initialized_modules = sorted(
        {
            key.split(".", 1)[0]
            for key in load_result.missing_keys
            if key.startswith(allowed_missing_prefixes)
        }
    )
    initialized_modules.extend(
        sorted({key.split(".", 1)[0] for key in incompatible_shape_keys})
    )
    initialized_modules = sorted(set(initialized_modules))
    if initialized_modules:
        sidecar["checkpoint_compatibility"] = {
            "initialized_missing_modules": initialized_modules,
            "message": (
                "Checkpoint predates diff/VA auxiliary modules; missing modules were "
                "left at fresh initialization. Do not use this checkpoint with "
                "cluster_feature_strategy='masked_diffaware'."
            ),
        }
    model.eval()
    return model, sidecar

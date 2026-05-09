import json

import numpy as np
import torch

from cluster.models.discovery_net import (
    MusicDiscoveryDataset,
    MusicMetadataDiscoveryNet,
    _discovery_loss,
)
from cluster.pipeline.rerun import _raw_feature_embeddings, _strategy_requires_checkpoint
from cluster.utils import fit_scaler_state


def _write_processed_dataset(root):
    np.save(root / "audio.npy", np.asarray([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]], dtype=np.float32))
    np.save(root / "lyrics.npy", np.asarray([[3.0, 4.0], [7.0, 8.0], [0.0, 0.0]], dtype=np.float32))
    np.save(root / "metadata.npy", np.asarray([[1.0], [0.0], [2.0]], dtype=np.float32))
    np.save(root / "view_mask.npy", np.asarray([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=np.float32))
    np.save(root / "consistency.npy", np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.save(root / "va_diff.npy", np.asarray([[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    np.save(root / "labels_emotion.npy", np.asarray([0, 1, -1], dtype=np.int64))
    np.save(root / "original_va.npy", np.asarray([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]], dtype=np.float32))
    (root / "metadata_feature_names.json").write_text(json.dumps(["metadata::x"]), encoding="utf-8")
    (root / "meta.json").write_text(json.dumps({"num_clusters": 4}), encoding="utf-8")
    (root / "split_70_15_15.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"indices": [0, 1], "track_ids": ["a", "b"]},
                    "val": {"indices": [2], "track_ids": ["c"]},
                    "test": {"indices": [], "track_ids": []},
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "track_index.tsv").write_text(
        "index\tidentifier\tlyric_identifier\tquadrant\n"
        "0\ta\ta\tQ1\n"
        "1\tb\tb\tQ2\n"
        "2\tc\tc\t\n",
        encoding="utf-8",
    )


def test_scaler_and_dataset_ignore_missing_view_values(tmp_path):
    _write_processed_dataset(tmp_path)

    scaler = fit_scaler_state(str(tmp_path), "70_15_15", ["audio", "lyrics", "metadata"])

    assert scaler["audio"]["mean"] == [1.0, 2.0]
    assert scaler["lyrics"]["mean"] == [5.0, 6.0]
    assert scaler["metadata"]["mean"] == [1.0]

    dataset = MusicDiscoveryDataset(
        data_dir=str(tmp_path),
        split="train",
        split_protocol="70_15_15",
        scaler_state=scaler,
    )

    missing_audio_item = dataset[1]
    assert torch.equal(missing_audio_item["view_mask"], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(missing_audio_item["audio"], torch.tensor([0.0, 0.0]))
    assert torch.equal(missing_audio_item["metadata"], torch.tensor([0.0]))
    assert torch.equal(dataset[0]["mean_va"], torch.tensor([2.0, 3.0]))
    assert torch.equal(missing_audio_item["mean_va"], torch.tensor([7.0, 8.0]))
    assert dataset[0]["va_geometry"].shape == (17,)
    assert missing_audio_item["va_geometry"][14].item() == 0.0
    assert missing_audio_item["va_geometry"][15].item() == 0.0
    assert missing_audio_item["va_geometry"][16].item() == 1.0
    assert missing_audio_item["track_id"] == "b"
    assert missing_audio_item["split"] == "train"

    raw_embeddings = _raw_feature_embeddings(dataset)
    assert set(["mean_va", "va_geometry", "z_fused", "gate_weights"]).issubset(raw_embeddings)
    np.testing.assert_allclose(raw_embeddings["mean_va"][0], np.asarray([2.0, 3.0], dtype=np.float32))
    assert raw_embeddings["va_geometry"].shape == (2, 17)
    assert raw_embeddings["z_fused"].shape == (2, 2)


def test_rerun_checkpoint_requirement_matches_feature_strategy():
    assert _strategy_requires_checkpoint("full")
    assert _strategy_requires_checkpoint("fused_only")
    assert _strategy_requires_checkpoint("fused_va_geometry")
    assert not _strategy_requires_checkpoint("mean_va")
    assert not _strategy_requires_checkpoint("va_geometry")
    assert not _strategy_requires_checkpoint("mean_va_diff")
    assert not _strategy_requires_checkpoint("original_va")


def test_model_gate_and_loss_are_mask_aware():
    model = MusicMetadataDiscoveryNet(
        audio_dim=2,
        lyrics_dim=2,
        metadata_dim=1,
        metadata_recon_dim=2,
        latent_dim=4,
        hidden_dim=8,
        metadata_hidden_dim=8,
        gate_hidden_dim=12,
    )
    model.train()
    batch = {
        "audio": torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float32),
        "lyrics": torch.tensor([[1.5, 2.5], [3.0, 4.0]], dtype=torch.float32),
        "metadata": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        "metadata_recon_target": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "view_mask": torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "consistency": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "va_diff": torch.tensor([[0.5, 0.5], [0.0, 0.0]], dtype=torch.float32),
        "va_geometry": torch.tensor(
            [
                [1.25, 2.25, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.1, 1.8, 2.4, -0.6, 0.6, 1.0, 1.0, 1.0],
                [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    }
    assert model.gate_mlp[0].in_features == 4 * 3 + 20

    outputs = model(
        audio=batch["audio"],
        lyrics=batch["lyrics"],
        metadata=batch["metadata"],
        consistency=batch["consistency"],
        va_diff=batch["va_diff"],
        va_geometry=batch["va_geometry"],
        view_mask=batch["view_mask"],
    )

    assert outputs["metadata_recon"].shape == (2, 2)
    assert outputs["z_consensus"].shape == outputs["z_cluster"].shape == (2, 4)
    assert outputs["z_tension"].shape == (2, 4)
    assert outputs["gate_weights"][1, 0].item() == 0.0
    assert outputs["gate_weights"][1, 2].item() == 0.0
    assert outputs["gate_weights"][1, 1].item() == 1.0

    losses = _discovery_loss(
        outputs=outputs,
        batch=batch,
        metadata_recon_weight=0.35,
        fused_recon_weight=0.5,
        align_weight=0.2,
        metadata_align_weight=0.1,
        gate_entropy_weight=0.01,
        metadata_recon_loss="bce",
    )
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["recon_metadata"])

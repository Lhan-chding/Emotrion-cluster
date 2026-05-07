import torch

from cluster.models.discovery_net import (
    MusicMetadataDiscoveryNet,
    _discovery_loss,
    target_distribution,
)


def test_target_distribution_normalizes_and_sharpens_assignments():
    q = torch.tensor(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )

    p = target_distribution(q)

    assert p.shape == q.shape
    torch.testing.assert_close(p.sum(dim=1), torch.ones(3))
    assert p[0, 0] > q[0, 0]
    assert p[2, 1] > q[2, 1]


def test_model_cluster_head_emits_per_view_assignments():
    model = MusicMetadataDiscoveryNet(
        audio_dim=2,
        lyrics_dim=2,
        metadata_dim=1,
        latent_dim=4,
        hidden_dim=8,
        metadata_hidden_dim=8,
        gate_hidden_dim=12,
        cluster_head_k=3,
        cluster_temperature=0.7,
    )
    batch = {
        "audio": torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float32),
        "lyrics": torch.tensor([[1.5, 2.5], [3.0, 4.0]], dtype=torch.float32),
        "metadata": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        "view_mask": torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        "consistency": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "va_diff": torch.tensor([[0.5, 0.5], [0.0, 0.0]], dtype=torch.float32),
    }

    outputs = model(
        audio=batch["audio"],
        lyrics=batch["lyrics"],
        metadata=batch["metadata"],
        consistency=batch["consistency"],
        va_diff=batch["va_diff"],
        view_mask=batch["view_mask"],
    )

    for key in ("q_audio", "q_lyrics", "q_metadata", "q_fused"):
        assert outputs[key].shape == (2, 3)
        torch.testing.assert_close(outputs[key].sum(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)


def test_discovery_loss_includes_mask_aware_dec_and_cvcl_terms():
    model = MusicMetadataDiscoveryNet(
        audio_dim=2,
        lyrics_dim=2,
        metadata_dim=1,
        latent_dim=4,
        hidden_dim=8,
        metadata_hidden_dim=8,
        gate_hidden_dim=12,
        cluster_head_k=3,
    )
    batch = {
        "audio": torch.tensor([[1.0, 2.0], [0.0, 0.0], [4.0, 5.0]], dtype=torch.float32),
        "lyrics": torch.tensor([[1.5, 2.5], [3.0, 4.0], [0.0, 0.0]], dtype=torch.float32),
        "metadata": torch.tensor([[1.0], [0.0], [2.0]], dtype=torch.float32),
        "view_mask": torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        "consistency": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        "va_diff": torch.tensor([[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
    }
    outputs = model(
        audio=batch["audio"],
        lyrics=batch["lyrics"],
        metadata=batch["metadata"],
        consistency=batch["consistency"],
        va_diff=batch["va_diff"],
        view_mask=batch["view_mask"],
    )

    losses = _discovery_loss(
        outputs=outputs,
        batch=batch,
        metadata_recon_weight=0.35,
        fused_recon_weight=0.5,
        align_weight=0.2,
        metadata_align_weight=0.1,
        gate_entropy_weight=0.01,
        cluster_loss_weight=0.3,
        cvcl_loss_weight=0.2,
        assignment_balance_weight=0.1,
    )

    assert torch.isfinite(losses["loss"])
    assert losses["cluster_loss"].item() >= 0.0
    assert losses["cvcl_loss"].item() >= 0.0
    assert losses["assignment_balance"].item() >= 0.0


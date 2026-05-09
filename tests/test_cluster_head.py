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
        "va_geometry": torch.zeros((2, 17), dtype=torch.float32),
    }

    outputs = model(
        audio=batch["audio"],
        lyrics=batch["lyrics"],
        metadata=batch["metadata"],
        consistency=batch["consistency"],
        va_diff=batch["va_diff"],
        va_geometry=batch["va_geometry"],
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
        "va_geometry": torch.zeros((3, 17), dtype=torch.float32),
    }
    outputs = model(
        audio=batch["audio"],
        lyrics=batch["lyrics"],
        metadata=batch["metadata"],
        consistency=batch["consistency"],
        va_diff=batch["va_diff"],
        va_geometry=batch["va_geometry"],
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


def test_diff_preserve_loss_trains_diff_encoder():
    model = MusicMetadataDiscoveryNet(
        audio_dim=2,
        lyrics_dim=2,
        metadata_dim=1,
        latent_dim=4,
        hidden_dim=8,
        metadata_hidden_dim=8,
        gate_hidden_dim=12,
    )
    batch = {
        "audio": torch.tensor([[1.0, 2.0], [2.0, 1.0], [4.0, 5.0]], dtype=torch.float32),
        "lyrics": torch.tensor([[1.5, 2.5], [2.5, 1.5], [4.5, 5.5]], dtype=torch.float32),
        "metadata": torch.tensor([[1.0], [0.5], [2.0]], dtype=torch.float32),
        "view_mask": torch.ones((3, 3), dtype=torch.float32),
        "consistency": torch.ones(3, dtype=torch.float32),
        "va_diff": torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        "signed_va_diff": torch.tensor([[-0.5, -0.5], [-0.5, -0.5], [-0.5, -0.5]], dtype=torch.float32),
        "va_geometry": torch.zeros((3, 17), dtype=torch.float32),
        "diff_geometry": torch.randn((3, 26), dtype=torch.float32),
        "diff_observed": torch.ones(3, dtype=torch.float32),
        "mean_va": torch.tensor([[1.25, 2.25], [2.25, 1.25], [4.25, 5.25]], dtype=torch.float32),
    }

    outputs = model(
        audio=batch["audio"],
        lyrics=batch["lyrics"],
        metadata=batch["metadata"],
        consistency=batch["consistency"],
        va_diff=batch["va_diff"],
        va_geometry=batch["va_geometry"],
        view_mask=batch["view_mask"],
        diff_features=batch["diff_geometry"],
        diff_observed=batch["diff_observed"],
    )
    losses = _discovery_loss(
        outputs=outputs,
        batch=batch,
        metadata_recon_weight=0.35,
        fused_recon_weight=0.5,
        align_weight=0.2,
        metadata_align_weight=0.1,
        diff_preserve_weight=0.05,
    )
    losses["loss"].backward()

    grad_total = sum(
        0.0 if param.grad is None else float(param.grad.abs().sum())
        for param in model.diff_encoder.parameters()
    )
    assert grad_total > 0.0


def _minimal_loss_tensors():
    audio = torch.zeros((2, 2), dtype=torch.float32)
    lyrics = torch.zeros((2, 2), dtype=torch.float32)
    metadata = torch.zeros((2, 1), dtype=torch.float32)
    latent = torch.zeros((2, 4), dtype=torch.float32)
    gate = torch.full((2, 3), 1.0 / 3.0, dtype=torch.float32)
    outputs = {
        "z_audio": latent,
        "z_lyrics": latent,
        "z_metadata": latent,
        "z_fused": latent,
        "z_diff": latent,
        "audio_recon": audio,
        "lyrics_recon": lyrics,
        "metadata_recon": metadata,
        "fused_audio_recon": audio,
        "fused_lyrics_recon": lyrics,
        "proj_audio": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "proj_lyrics": torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        "proj_metadata": latent,
        "proj_fused": latent,
        "gate_weights": gate,
        "va_pred": audio,
    }
    batch = {
        "audio": audio,
        "lyrics": lyrics,
        "metadata": metadata,
        "view_mask": torch.ones((2, 3), dtype=torch.float32),
        "consistency": torch.ones(2, dtype=torch.float32),
        "va_diff": torch.zeros((2, 2), dtype=torch.float32),
        "signed_va_diff": torch.tensor([[0.4, -0.2], [-0.3, 0.5]], dtype=torch.float32),
        "diff_observed": torch.ones(2, dtype=torch.float32),
        "mean_va": torch.zeros((2, 2), dtype=torch.float32),
    }
    return outputs, batch


def test_diff_preserve_loss_is_signed_vector_aware():
    outputs, batch = _minimal_loss_tensors()
    matching = dict(outputs)
    matching["signed_diff_pred"] = batch["signed_va_diff"].clone()
    matching["pair_signed_diff_pred"] = batch["signed_va_diff"].clone()
    flipped = dict(outputs)
    flipped["signed_diff_pred"] = -batch["signed_va_diff"]
    flipped["pair_signed_diff_pred"] = -batch["signed_va_diff"]

    matching_losses = _discovery_loss(
        outputs=matching,
        batch=batch,
        metadata_recon_weight=0.0,
        fused_recon_weight=0.0,
        align_weight=0.0,
        metadata_align_weight=0.0,
        diff_preserve_weight=1.0,
    )
    flipped_losses = _discovery_loss(
        outputs=flipped,
        batch=batch,
        metadata_recon_weight=0.0,
        fused_recon_weight=0.0,
        align_weight=0.0,
        metadata_align_weight=0.0,
        diff_preserve_weight=1.0,
    )

    assert matching_losses["diff_preserve"].item() < flipped_losses["diff_preserve"].item()


def test_audio_lyrics_alignment_downweights_large_signed_disagreement():
    outputs, batch = _minimal_loss_tensors()
    small_gap = dict(batch)
    small_gap["signed_va_diff"] = torch.tensor([[0.02, 0.01], [0.01, -0.02]], dtype=torch.float32)
    large_gap = dict(batch)
    large_gap["signed_va_diff"] = torch.tensor([[0.8, -0.7], [-0.75, 0.6]], dtype=torch.float32)

    small_losses = _discovery_loss(
        outputs=outputs,
        batch=small_gap,
        metadata_recon_weight=0.0,
        fused_recon_weight=0.0,
        align_weight=1.0,
        metadata_align_weight=0.0,
    )
    large_losses = _discovery_loss(
        outputs=outputs,
        batch=large_gap,
        metadata_recon_weight=0.0,
        fused_recon_weight=0.0,
        align_weight=1.0,
        metadata_align_weight=0.0,
    )

    assert large_losses["align_audio_lyrics"].item() < small_losses["align_audio_lyrics"].item()

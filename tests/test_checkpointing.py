import torch

from cluster.models.discovery_net import (
    MusicMetadataDiscoveryNet,
    load_music_discovery_checkpoint,
    save_discovery_checkpoint,
)


def test_discovery_checkpoint_contains_resume_and_schema_metadata(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    model = MusicMetadataDiscoveryNet(
        audio_dim=2,
        lyrics_dim=2,
        metadata_dim=1,
        latent_dim=4,
        hidden_dim=8,
        metadata_hidden_dim=8,
        gate_hidden_dim=12,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    save_discovery_checkpoint(
        model=model,
        checkpoint_path=str(checkpoint_path),
        scaler_state={
            "audio": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "lyrics": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "metadata": {"mean": [0.0], "std": [1.0]},
        },
        config={"latent_dim": 4, "hidden_dim": 8, "metadata_hidden_dim": 8, "gate_hidden_dim": 12},
        best_metrics={"best_val_loss": 1.25},
        optimizer_state=optimizer.state_dict(),
        epoch=3,
        global_step=7,
        dataset_version="v-test",
        dataset_hash="dataset-hash",
        schema_hash="schema-hash",
        metadata_schema={"feature_names": ["metadata::x"]},
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert "model_state" in payload
    assert "state_dict" in payload
    assert payload["optimizer_state"] is not None
    assert payload["epoch"] == 3
    assert payload["global_step"] == 7
    assert payload["dataset_version"] == "v-test"
    assert payload["dataset_hash"] == "dataset-hash"
    assert payload["schema_hash"] == "schema-hash"
    assert payload["runtime"]["torch"]

    loaded_model, sidecar = load_music_discovery_checkpoint(str(checkpoint_path), torch.device("cpu"))
    assert isinstance(loaded_model, MusicMetadataDiscoveryNet)
    assert sidecar["schema_hash"] == "schema-hash"


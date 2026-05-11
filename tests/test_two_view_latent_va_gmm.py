import numpy as np

from cluster.models.two_view_latent_va_gmm import TwoViewLatentVAGMM


def _two_view_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    centers = np.asarray([[0.25, 0.30], [0.78, 0.72]], dtype=np.float32)
    audio_rows = []
    lyrics_rows = []
    mask_rows = []
    for center in centers:
        for _ in range(24):
            latent = center + rng.normal(0.0, 0.025, size=2)
            audio_rows.append(latent + np.asarray([0.06, -0.02]) + rng.normal(0.0, 0.025, size=2))
            lyrics_rows.append(latent + np.asarray([-0.04, 0.03]) + rng.normal(0.0, 0.04, size=2))
            mask_rows.append([1.0, 1.0, 1.0])
    return (
        np.asarray(audio_rows, dtype=np.float32),
        np.asarray(lyrics_rows, dtype=np.float32),
        np.asarray(mask_rows, dtype=np.float32),
    )


def test_two_view_latent_va_gmm_fits_predicts_and_exposes_reliability():
    audio, lyrics, view_mask = _two_view_fixture()

    model = TwoViewLatentVAGMM(
        n_components=2,
        covariance_type="diag",
        learn_bias=True,
        n_init=3,
        max_iter=60,
        random_state=31,
    ).fit(audio, lyrics, view_mask)

    labels = model.predict(audio, lyrics, view_mask)
    posterior = model.predict_proba(audio, lyrics, view_mask)
    consensus = model.posterior_consensus(audio, lyrics, view_mask)
    tension = model.posterior_tension(audio, lyrics, view_mask)
    reliability = model.view_reliability()

    assert labels.shape == (audio.shape[0],)
    assert posterior.shape == (audio.shape[0], 2)
    assert consensus.shape == (audio.shape[0], 2)
    assert tension.shape == (audio.shape[0], 2)
    assert reliability["alpha_audio"].shape == (2, 2)
    assert np.allclose(posterior.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(model.bic(audio, lyrics, view_mask))
    assert np.isfinite(model.icl(audio, lyrics, view_mask))
    assert np.unique(labels).size == 2


def test_two_view_latent_va_gmm_supports_single_missing_view_likelihood():
    audio, lyrics, view_mask = _two_view_fixture()
    partial_mask = view_mask.copy()
    partial_mask[::3, 1] = 0.0
    partial_mask[1::3, 0] = 0.0

    model = TwoViewLatentVAGMM(
        n_components=2,
        covariance_type="diag",
        learn_bias=True,
        n_init=2,
        max_iter=40,
        random_state=37,
    ).fit(audio, lyrics, partial_mask)

    labels = model.predict(audio, lyrics, partial_mask)
    consensus = model.posterior_consensus(audio, lyrics, partial_mask)

    assert labels.shape == (audio.shape[0],)
    assert consensus.shape == (audio.shape[0], 2)
    assert np.isfinite(consensus).all()

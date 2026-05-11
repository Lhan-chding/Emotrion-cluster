from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture


def _logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_values)
    return (max_values + np.log(np.sum(stable, axis=axis, keepdims=True))).squeeze(axis)


def _normalize_rows(log_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    log_norm = _logsumexp(log_probs, axis=1)
    return np.exp(log_probs - log_norm.reshape(-1, 1)), log_norm


@dataclass
class _FitState:
    lower_bound: float
    n_iter: int
    converged: bool
    weights: np.ndarray
    means: np.ndarray
    latent_vars: np.ndarray
    audio_biases: np.ndarray
    lyrics_biases: np.ndarray
    audio_noise_vars: np.ndarray
    lyrics_noise_vars: np.ndarray
    responsibilities: np.ndarray


class TwoViewLatentVAGMM:
    """Two-view latent VA mixture with missing-view likelihood.

    Each component models an unobserved consensus VA vector ``z``. Audio and
    lyrics VA are treated as noisy, biased observations of that consensus:
    ``audio = z + b_audio + eps_audio`` and
    ``lyrics = z + b_lyrics + eps_lyrics``.
    """

    def __init__(
        self,
        n_components: int,
        *,
        covariance_type: str = "diag",
        learn_bias: bool = True,
        share_view_noise: bool = False,
        alpha_prior_strength: float = 0.0,
        reg_covar: float = 1e-5,
        n_init: int = 5,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        if str(covariance_type).lower() != "diag":
            raise ValueError("TwoViewLatentVAGMM currently supports covariance_type='diag' only.")
        self.n_components = int(n_components)
        self.covariance_type = "diag"
        self.learn_bias = bool(learn_bias)
        self.share_view_noise = bool(share_view_noise)
        self.alpha_prior_strength = float(alpha_prior_strength)
        self.reg_covar = float(reg_covar)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def fit(
        self,
        audio_va: np.ndarray,
        lyrics_va: Optional[np.ndarray] = None,
        view_mask: Optional[np.ndarray] = None,
    ) -> "TwoViewLatentVAGMM":
        audio, lyrics, mask = self._validate_inputs(audio_va, lyrics_va, view_mask)
        consensus = self._observed_consensus(audio, lyrics, mask)
        rng = np.random.default_rng(self.random_state)
        best_state: Optional[_FitState] = None

        for init_idx in range(max(1, self.n_init)):
            seed = int(rng.integers(0, 2**31 - 1)) if self.random_state is not None else None
            state = self._fit_single_initialization(audio, lyrics, mask, consensus, seed, init_idx)
            if best_state is None or state.lower_bound > best_state.lower_bound:
                best_state = state

        if best_state is None:
            raise RuntimeError("TwoViewLatentVAGMM failed to initialize.")

        self.weights_ = best_state.weights.astype(np.float64)
        self.means_ = best_state.means.astype(np.float64)
        self.latent_vars_ = best_state.latent_vars.astype(np.float64)
        self.audio_biases_ = best_state.audio_biases.astype(np.float64)
        self.lyrics_biases_ = best_state.lyrics_biases.astype(np.float64)
        self.audio_noise_vars_ = best_state.audio_noise_vars.astype(np.float64)
        self.lyrics_noise_vars_ = best_state.lyrics_noise_vars.astype(np.float64)
        self.responsibilities_ = best_state.responsibilities.astype(np.float64)
        self.labels_ = np.argmax(self.responsibilities_, axis=1).astype(np.int64)
        self.lower_bound_ = float(best_state.lower_bound / max(audio.shape[0], 1))
        self.total_log_likelihood_ = float(best_state.lower_bound)
        self.n_iter_ = int(best_state.n_iter)
        self.converged_ = bool(best_state.converged)
        self.n_features_in_ = 4
        return self

    def predict(
        self,
        audio_va: np.ndarray,
        lyrics_va: Optional[np.ndarray] = None,
        view_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.argmax(self.predict_proba(audio_va, lyrics_va, view_mask), axis=1).astype(np.int64)

    def predict_proba(
        self,
        audio_va: np.ndarray,
        lyrics_va: Optional[np.ndarray] = None,
        view_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        audio, lyrics, mask = self._validate_inputs(audio_va, lyrics_va, view_mask)
        log_probs = self._estimate_log_prob(audio, lyrics, mask)
        responsibilities, _ = _normalize_rows(log_probs)
        return responsibilities.astype(np.float64)

    def score_samples(
        self,
        audio_va: np.ndarray,
        lyrics_va: Optional[np.ndarray] = None,
        view_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        audio, lyrics, mask = self._validate_inputs(audio_va, lyrics_va, view_mask)
        return _logsumexp(self._estimate_log_prob(audio, lyrics, mask), axis=1)

    def posterior_consensus(
        self,
        audio_va: np.ndarray,
        lyrics_va: Optional[np.ndarray] = None,
        view_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        audio, lyrics, mask = self._validate_inputs(audio_va, lyrics_va, view_mask)
        responsibilities = self.predict_proba(audio, lyrics, mask)
        component_means = self._component_posterior_means(audio, lyrics, mask)
        consensus = np.einsum("nk,nkd->nd", responsibilities, component_means)
        return consensus.astype(np.float32)

    def posterior_tension(
        self,
        audio_va: np.ndarray,
        lyrics_va: Optional[np.ndarray] = None,
        view_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        audio, lyrics, mask = self._validate_inputs(audio_va, lyrics_va, view_mask)
        responsibilities = self.predict_proba(audio, lyrics, mask)
        raw_delta = lyrics[:, None, :] - audio[:, None, :]
        expected_bias_delta = self.lyrics_biases_[None, :, :] - self.audio_biases_[None, :, :]
        tension_by_component = raw_delta - expected_bias_delta
        tension = np.einsum("nk,nkd->nd", responsibilities, tension_by_component)
        both = (mask[:, 0] > 0.0) & (mask[:, 1] > 0.0)
        return np.where(both.reshape(-1, 1), tension, 0.0).astype(np.float32)

    def view_reliability(self) -> Dict[str, np.ndarray]:
        audio_precision = 1.0 / np.maximum(self.audio_noise_vars_, self.reg_covar)
        lyrics_precision = 1.0 / np.maximum(self.lyrics_noise_vars_, self.reg_covar)
        total = np.maximum(audio_precision + lyrics_precision, self.reg_covar)
        return {
            "alpha_audio": (audio_precision / total).astype(np.float32),
            "alpha_lyrics": (lyrics_precision / total).astype(np.float32),
            "audio_noise_var": self.audio_noise_vars_.astype(np.float32),
            "lyrics_noise_var": self.lyrics_noise_vars_.astype(np.float32),
        }

    def bic(
        self,
        audio_va: np.ndarray,
        lyrics_va: Optional[np.ndarray] = None,
        view_mask: Optional[np.ndarray] = None,
    ) -> float:
        audio, lyrics, mask = self._validate_inputs(audio_va, lyrics_va, view_mask)
        log_likelihood = float(np.sum(self.score_samples(audio, lyrics, mask)))
        n_params = self._parameter_count()
        return float(-2.0 * log_likelihood + n_params * np.log(max(audio.shape[0], 2)))

    def icl(
        self,
        audio_va: np.ndarray,
        lyrics_va: Optional[np.ndarray] = None,
        view_mask: Optional[np.ndarray] = None,
    ) -> float:
        responsibilities = self.predict_proba(audio_va, lyrics_va, view_mask)
        entropy = -float(np.sum(responsibilities * np.log(np.maximum(responsibilities, 1e-12))))
        return float(self.bic(audio_va, lyrics_va, view_mask) + 2.0 * entropy)

    def _fit_single_initialization(
        self,
        audio: np.ndarray,
        lyrics: np.ndarray,
        mask: np.ndarray,
        consensus: np.ndarray,
        seed: Optional[int],
        init_idx: int,
    ) -> _FitState:
        jitter = 1e-4 * (init_idx + 1)
        init_matrix = consensus + np.random.default_rng(seed).normal(0.0, jitter, size=consensus.shape)
        init_gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            reg_covar=self.reg_covar,
            n_init=1,
            max_iter=100,
            random_state=seed,
        ).fit(init_matrix)
        responsibilities = init_gmm.predict_proba(init_matrix)
        previous = -float("inf")
        state: Optional[_FitState] = None
        converged = False

        for iteration in range(1, max(1, self.max_iter) + 1):
            params = self._m_step(audio, lyrics, mask, consensus, responsibilities)
            log_probs = self._estimate_log_prob_with_params(audio, lyrics, mask, params)
            responsibilities, log_norm = _normalize_rows(log_probs)
            lower_bound = float(np.sum(log_norm))
            state = _FitState(
                lower_bound=lower_bound,
                n_iter=iteration,
                converged=False,
                responsibilities=responsibilities,
                **params,
            )
            if iteration > 1 and abs(lower_bound - previous) <= self.tol * max(1.0, abs(previous)):
                converged = True
                state.converged = True
                break
            previous = lower_bound

        if state is None:
            raise RuntimeError("TwoViewLatentVAGMM did not run any EM iterations.")
        state.converged = bool(converged)
        return state

    def _m_step(
        self,
        audio: np.ndarray,
        lyrics: np.ndarray,
        mask: np.ndarray,
        consensus: np.ndarray,
        responsibilities: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        n, dim = consensus.shape
        effective_n = responsibilities.sum(axis=0) + 10.0 * self.reg_covar
        weights = effective_n / max(float(n), self.reg_covar)
        weights = weights / np.maximum(weights.sum(), self.reg_covar)
        means = (responsibilities.T @ consensus) / effective_n.reshape(-1, 1)

        audio_biases = np.zeros((self.n_components, dim), dtype=np.float64)
        lyrics_biases = np.zeros((self.n_components, dim), dtype=np.float64)
        obs_audio = (mask[:, 0] > 0.0).astype(np.float64)
        obs_lyrics = (mask[:, 1] > 0.0).astype(np.float64)
        if self.learn_bias:
            for k in range(self.n_components):
                weights_k = responsibilities[:, k]
                audio_den = float(np.sum(weights_k * obs_audio))
                lyrics_den = float(np.sum(weights_k * obs_lyrics))
                if audio_den > self.reg_covar:
                    audio_biases[k] = np.sum((weights_k * obs_audio).reshape(-1, 1) * (audio - means[k]), axis=0) / audio_den
                if lyrics_den > self.reg_covar:
                    lyrics_biases[k] = np.sum((weights_k * obs_lyrics).reshape(-1, 1) * (lyrics - means[k]), axis=0) / lyrics_den
                center = 0.5 * (audio_biases[k] + lyrics_biases[k])
                audio_biases[k] -= center
                lyrics_biases[k] -= center

        centered_consensus = consensus[:, None, :] - means[None, :, :]
        latent_vars = np.sum(responsibilities[:, :, None] * centered_consensus**2, axis=0) / effective_n.reshape(-1, 1)
        latent_vars = np.maximum(latent_vars, self.reg_covar)

        audio_noise_vars = np.zeros((self.n_components, dim), dtype=np.float64)
        lyrics_noise_vars = np.zeros((self.n_components, dim), dtype=np.float64)
        for k in range(self.n_components):
            weights_k = responsibilities[:, k]
            audio_den = float(np.sum(weights_k * obs_audio))
            lyrics_den = float(np.sum(weights_k * obs_lyrics))
            if audio_den > self.reg_covar:
                audio_resid = audio - consensus - audio_biases[k]
                audio_noise_vars[k] = np.sum((weights_k * obs_audio).reshape(-1, 1) * audio_resid**2, axis=0) / audio_den
            else:
                audio_noise_vars[k] = np.var(audio - consensus, axis=0)
            if lyrics_den > self.reg_covar:
                lyrics_resid = lyrics - consensus - lyrics_biases[k]
                lyrics_noise_vars[k] = np.sum((weights_k * obs_lyrics).reshape(-1, 1) * lyrics_resid**2, axis=0) / lyrics_den
            else:
                lyrics_noise_vars[k] = np.var(lyrics - consensus, axis=0)

        if self.share_view_noise:
            pooled = 0.5 * (audio_noise_vars + lyrics_noise_vars)
            audio_noise_vars = pooled.copy()
            lyrics_noise_vars = pooled.copy()

        if self.alpha_prior_strength > 0.0:
            pooled = 0.5 * (audio_noise_vars + lyrics_noise_vars)
            strength = float(self.alpha_prior_strength)
            audio_noise_vars = (audio_noise_vars + strength * pooled) / (1.0 + strength)
            lyrics_noise_vars = (lyrics_noise_vars + strength * pooled) / (1.0 + strength)

        return {
            "weights": weights.astype(np.float64),
            "means": means.astype(np.float64),
            "latent_vars": np.maximum(latent_vars, self.reg_covar).astype(np.float64),
            "audio_biases": audio_biases.astype(np.float64),
            "lyrics_biases": lyrics_biases.astype(np.float64),
            "audio_noise_vars": np.maximum(audio_noise_vars, self.reg_covar).astype(np.float64),
            "lyrics_noise_vars": np.maximum(lyrics_noise_vars, self.reg_covar).astype(np.float64),
        }

    def _estimate_log_prob(self, audio: np.ndarray, lyrics: np.ndarray, mask: np.ndarray) -> np.ndarray:
        params = {
            "weights": self.weights_,
            "means": self.means_,
            "latent_vars": self.latent_vars_,
            "audio_biases": self.audio_biases_,
            "lyrics_biases": self.lyrics_biases_,
            "audio_noise_vars": self.audio_noise_vars_,
            "lyrics_noise_vars": self.lyrics_noise_vars_,
        }
        return self._estimate_log_prob_with_params(audio, lyrics, mask, params)

    def _estimate_log_prob_with_params(
        self,
        audio: np.ndarray,
        lyrics: np.ndarray,
        mask: np.ndarray,
        params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        weights = np.maximum(params["weights"], 1e-12)
        means = params["means"]
        latent_vars = np.maximum(params["latent_vars"], self.reg_covar)
        audio_biases = params["audio_biases"]
        lyrics_biases = params["lyrics_biases"]
        audio_noise = np.maximum(params["audio_noise_vars"], self.reg_covar)
        lyrics_noise = np.maximum(params["lyrics_noise_vars"], self.reg_covar)
        log_probs = np.full((audio.shape[0], self.n_components), -np.inf, dtype=np.float64)
        obs_audio = mask[:, 0] > 0.0
        obs_lyrics = mask[:, 1] > 0.0

        for k in range(self.n_components):
            value = np.full(audio.shape[0], np.log(weights[k]), dtype=np.float64)
            mean_audio = means[k] + audio_biases[k]
            mean_lyrics = means[k] + lyrics_biases[k]
            var_audio = latent_vars[k] + audio_noise[k]
            var_lyrics = latent_vars[k] + lyrics_noise[k]

            audio_only = obs_audio & ~obs_lyrics
            if audio_only.any():
                value[audio_only] += self._diag_normal_logpdf(audio[audio_only], mean_audio, var_audio)

            lyrics_only = obs_lyrics & ~obs_audio
            if lyrics_only.any():
                value[lyrics_only] += self._diag_normal_logpdf(lyrics[lyrics_only], mean_lyrics, var_lyrics)

            both = obs_audio & obs_lyrics
            if both.any():
                value[both] += self._two_view_logpdf(
                    audio[both],
                    lyrics[both],
                    mean_audio,
                    mean_lyrics,
                    latent_vars[k],
                    audio_noise[k],
                    lyrics_noise[k],
                )
            log_probs[:, k] = value
        return log_probs

    def _component_posterior_means(self, audio: np.ndarray, lyrics: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n = audio.shape[0]
        out = np.zeros((n, self.n_components, 2), dtype=np.float64)
        obs_audio = (mask[:, 0] > 0.0).astype(np.float64).reshape(-1, 1)
        obs_lyrics = (mask[:, 1] > 0.0).astype(np.float64).reshape(-1, 1)
        for k in range(self.n_components):
            prior_precision = 1.0 / np.maximum(self.latent_vars_[k], self.reg_covar)
            audio_precision = obs_audio / np.maximum(self.audio_noise_vars_[k], self.reg_covar)
            lyrics_precision = obs_lyrics / np.maximum(self.lyrics_noise_vars_[k], self.reg_covar)
            denominator = prior_precision + audio_precision + lyrics_precision
            numerator = (
                self.means_[k] * prior_precision
                + audio_precision * (audio - self.audio_biases_[k])
                + lyrics_precision * (lyrics - self.lyrics_biases_[k])
            )
            out[:, k, :] = numerator / np.maximum(denominator, self.reg_covar)
        return out

    @staticmethod
    def _diag_normal_logpdf(values: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
        variance = np.maximum(variance, 1e-12)
        diff = values - mean.reshape(1, -1)
        return -0.5 * np.sum(np.log(2.0 * np.pi * variance) + diff**2 / variance, axis=1)

    @staticmethod
    def _two_view_logpdf(
        audio: np.ndarray,
        lyrics: np.ndarray,
        mean_audio: np.ndarray,
        mean_lyrics: np.ndarray,
        latent_var: np.ndarray,
        audio_noise: np.ndarray,
        lyrics_noise: np.ndarray,
    ) -> np.ndarray:
        va = np.maximum(latent_var + audio_noise, 1e-12)
        vl = np.maximum(latent_var + lyrics_noise, 1e-12)
        cov = np.maximum(latent_var, 1e-12)
        det = np.maximum(va * vl - cov**2, 1e-12)
        da = audio - mean_audio.reshape(1, -1)
        dl = lyrics - mean_lyrics.reshape(1, -1)
        quad = (vl * da**2 + va * dl**2 - 2.0 * cov * da * dl) / det
        return -0.5 * np.sum(np.log((2.0 * np.pi) ** 2 * det) + quad, axis=1)

    def _parameter_count(self) -> int:
        k = int(self.n_components)
        dim = 2
        weights = k - 1
        means = k * dim
        latent_vars = k * dim
        biases = k * dim if self.learn_bias else 0
        noises = dim if self.share_view_noise else 2 * k * dim
        return int(weights + means + latent_vars + biases + noises)

    @staticmethod
    def _observed_consensus(audio: np.ndarray, lyrics: np.ndarray, mask: np.ndarray) -> np.ndarray:
        obs_audio = (mask[:, 0:1] > 0.0).astype(np.float64)
        obs_lyrics = (mask[:, 1:2] > 0.0).astype(np.float64)
        denom = np.maximum(obs_audio + obs_lyrics, 1.0)
        return (audio * obs_audio + lyrics * obs_lyrics) / denom

    @staticmethod
    def _validate_inputs(
        audio_va: np.ndarray,
        lyrics_va: Optional[np.ndarray],
        view_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if lyrics_va is None:
            matrix = np.asarray(audio_va, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] < 4:
                raise ValueError("When lyrics_va is omitted, audio_va must have shape [N, >=4].")
            audio = matrix[:, :2]
            lyrics = matrix[:, 2:4]
        else:
            audio = np.asarray(audio_va, dtype=np.float64)
            lyrics = np.asarray(lyrics_va, dtype=np.float64)
            if audio.ndim != 2 or lyrics.ndim != 2 or audio.shape != lyrics.shape or audio.shape[1] != 2:
                raise ValueError(f"audio_va and lyrics_va must both have shape [N, 2], got {audio.shape} and {lyrics.shape}.")
        if view_mask is None:
            mask = np.ones((audio.shape[0], 2), dtype=np.float64)
        else:
            mask = np.asarray(view_mask, dtype=np.float64)
            if mask.ndim != 2 or mask.shape[0] != audio.shape[0] or mask.shape[1] < 2:
                raise ValueError(f"view_mask must have shape [N, >=2], got {mask.shape}.")
            mask = mask[:, :2]
        observed = (mask[:, 0] > 0.0) | (mask[:, 1] > 0.0)
        if not observed.all():
            raise ValueError("Every row must have at least one observed VA view.")
        return audio.astype(np.float64), lyrics.astype(np.float64), mask.astype(np.float64)

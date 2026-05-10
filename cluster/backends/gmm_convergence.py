from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture


def _attempt_schedule(
    *,
    n_init: int,
    max_iter: int,
    tol: float,
    reg_covar: float,
) -> List[Dict[str, Any]]:
    base_iter = max(20, int(max_iter))
    base_n_init = max(1, int(n_init))
    base_tol = float(tol)
    base_reg = float(reg_covar)
    return [
        {
            "n_init": base_n_init,
            "max_iter": base_iter,
            "tol": base_tol,
            "reg_covar": base_reg,
        },
        {
            "n_init": max(base_n_init, 5),
            "max_iter": max(base_iter * 3, 300),
            "tol": max(base_tol, 1e-3),
            "reg_covar": base_reg,
        },
        {
            "n_init": max(base_n_init, 10),
            "max_iter": max(base_iter * 6, 600),
            "tol": max(base_tol, 1e-3),
            "reg_covar": max(base_reg * 10.0, 1e-4),
        },
        {
            "n_init": max(base_n_init, 20),
            "max_iter": max(base_iter * 10, 1000),
            "tol": max(base_tol, 2e-3),
            "reg_covar": max(base_reg * 100.0, 1e-3),
        },
    ]


def fit_gaussian_mixture_robust(
    features: np.ndarray,
    *,
    n_components: int,
    covariance_type: str = "diag",
    reg_covar: float = 1e-5,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-3,
    random_state: int = 42,
    require_converged: bool = False,
    context: str = "GaussianMixture",
) -> GaussianMixture:
    """Fit sklearn GaussianMixture with deterministic convergence retries.

    sklearn emits ``ConvergenceWarning`` even when the fitted object is still
    usable. For search pipelines that fit many candidate GMMs, raw warnings
    make logs look failed and give no actionable context. This helper catches
    those warnings, retries with stricter numerical settings, and annotates the
    returned model with retry diagnostics.
    """

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {matrix.shape}.")

    best_model: Optional[GaussianMixture] = None
    best_lower_bound = -np.inf
    attempts = _attempt_schedule(
        n_init=int(n_init),
        max_iter=int(max_iter),
        tol=float(tol),
        reg_covar=float(reg_covar),
    )
    diagnostics: List[Dict[str, Any]] = []
    for attempt_index, params in enumerate(attempts):
        model = GaussianMixture(
            n_components=int(n_components),
            covariance_type=str(covariance_type),
            reg_covar=float(params["reg_covar"]),
            n_init=int(params["n_init"]),
            max_iter=int(params["max_iter"]),
            tol=float(params["tol"]),
            random_state=int(random_state) + attempt_index,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(matrix)
        warning_count = sum(1 for item in caught if issubclass(item.category, ConvergenceWarning))
        converged = bool(getattr(model, "converged_", warning_count == 0))
        lower_bound = float(getattr(model, "lower_bound_", -np.inf))
        diagnostics.append(
            {
                "attempt": attempt_index + 1,
                "converged": converged,
                "warning_count": int(warning_count),
                "max_iter": int(params["max_iter"]),
                "n_init": int(params["n_init"]),
                "tol": float(params["tol"]),
                "reg_covar": float(params["reg_covar"]),
                "lower_bound": lower_bound,
            }
        )
        if best_model is None or lower_bound > best_lower_bound:
            best_model = model
            best_lower_bound = lower_bound
        if converged and warning_count == 0:
            setattr(model, "convergence_retry_count_", attempt_index)
            setattr(model, "convergence_attempts_", diagnostics)
            return model

    assert best_model is not None
    setattr(best_model, "convergence_retry_count_", len(attempts) - 1)
    setattr(best_model, "convergence_attempts_", diagnostics)
    if require_converged:
        raise RuntimeError(
            f"{context} did not converge after {len(attempts)} attempts. "
            f"Last attempt: {diagnostics[-1]}"
        )
    return best_model

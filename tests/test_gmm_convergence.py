import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from cluster.backends import gmm_convergence


def test_robust_gmm_retries_without_leaking_convergence_warning(monkeypatch):
    attempts = []

    class FakeGaussianMixture:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, _features):
            attempts.append(self.kwargs)
            self.lower_bound_ = float(len(attempts))
            if len(attempts) == 1:
                self.converged_ = False
                warnings.warn("first attempt did not converge", ConvergenceWarning)
            else:
                self.converged_ = True
            return self

    monkeypatch.setattr(gmm_convergence, "GaussianMixture", FakeGaussianMixture)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = gmm_convergence.fit_gaussian_mixture_robust(
            np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
            n_components=2,
            max_iter=20,
            n_init=1,
        )

    assert model.converged_ is True
    assert model.convergence_retry_count_ == 1
    assert attempts[1]["max_iter"] >= 300
    assert not any(issubclass(item.category, ConvergenceWarning) for item in caught)


def test_robust_gmm_can_fail_closed_when_convergence_is_required(monkeypatch):
    class AlwaysWarningGaussianMixture:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, _features):
            self.lower_bound_ = 0.0
            self.converged_ = False
            warnings.warn("still not converged", ConvergenceWarning)
            return self

    monkeypatch.setattr(gmm_convergence, "GaussianMixture", AlwaysWarningGaussianMixture)

    with pytest.raises(RuntimeError, match="did not converge"):
        gmm_convergence.fit_gaussian_mixture_robust(
            np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
            n_components=2,
            max_iter=20,
            n_init=1,
            require_converged=True,
            context="test-gmm",
        )

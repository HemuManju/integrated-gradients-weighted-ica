import numpy as np

from sklearn.decomposition import FastICA
from sklearn.covariance import oas

from pyriemann.utils.ajd import ajd_pham


def fit_fast_ica(signal):
    ica = FastICA(n_components=3, whiten=True)
    S_ = ica.fit_transform(signal)  # Reconstruct signals
    A_ = ica.mixing_

    # assert np.allclose(signal,
    #                    np.dot(S_, A_.T) + ica.mean_), 'Sources are not close'
    return S_, A_


def fit_weighted_ica(signal, n):
    covariance_set = np.empty((n, signal.shape[0], signal.shape[0]))
    signal_cov, _ = oas(signal.T, assume_centered=True)

    # Sample and find the covariance matrix
    for i in range(n):
        index = np.random.choice(signal.shape[1], 1, replace=True)

        # Weights
        sample = signal[:, index]
        weight = np.random.multivariate_normal(mean=sample.ravel(),
                                               cov=signal_cov,
                                               size=signal.shape[1]).T
        weight = (weight - weight.min()) / (weight - weight.min()).sum()
        m_signal = np.average(signal, weights=weight, axis=-1)  # weighted mean

        # Center the signal
        centered_signal = (signal - m_signal[:, np.newaxis])

        # Take the weighted covariance
        cov = np.dot(centered_signal * weight, centered_signal.T)
        covariance_set[i, :, :] = 0.5 * (cov + cov.T)

    # Get the best estimate of the diagonalization
    V, D = ajd_pham(covariance_set)

    # Recovered signal
    recovered = np.dot(V, signal)

    return V, recovered

import numpy as np
from scipy.stats import multivariate_normal

from sklearn.decomposition import FastICA
from sklearn.covariance import oas

from pyriemann.utils.ajd import ajd_pham


def fit_fast_ica(signal):
    ica = FastICA(n_components=3, whiten=True)
    S_ = ica.fit_transform(signal)  # Reconstruct signals
    A_ = ica.mixing_
    return S_, A_


def compute_weight(signal, power):
    dim = signal.shape[0]
    n_samples = dim

    idx = np.random.randint(low=0, high=signal.shape[1], size=1)
    sampled_points = signal[:, idx].reshape(n_samples, -1)

    # Mean and covariance
    sampled_points_mean = np.mean(sampled_points, axis=-1)
    cov_mat, _ = oas(signal.T, assume_centered=False)

    # Multivariate distribution
    mvn = multivariate_normal(mean=sampled_points_mean, cov=cov_mat)
    weight = np.exp(mvn.logpdf(signal.T))
    return weight


def fit_weighted_ica(signal, n):
    covariance_set = np.empty((n, signal.shape[0], signal.shape[0]))
    # Sample and find the covariance matrix
    for i in range(n):
        # Weights
        weight = compute_weight(signal, power=2)
        m_signal = np.average(signal, weights=weight, axis=-1)  # weighted mean

        # Center the signal
        centered_signal = (signal - m_signal[:, np.newaxis])

        # Take the weighted covariance
        covariance_set[i, :, :] = np.dot(centered_signal * weight,
                                         centered_signal.T) / np.sum(weight,
                                                                     axis=-1)
        print(covariance_set[i, :, :])

    # Get the best estimate of the diagonalization
    V, D = ajd_pham(covariance_set)

    # Recovered signal
    recovered = np.dot(V, signal)  # - signal.mean(axis=-1)[:, np.newaxis])

    return V, recovered

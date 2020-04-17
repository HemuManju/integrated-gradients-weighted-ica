import numpy as np
from sklearn.decomposition import FastICA

from pyriemann.estimation import Covariances
from pyriemann.utils.ajd import ajd_pham


def fit_fast_ica(signal):
    ica = FastICA(n_components=2, whiten=True)
    S_ = ica.fit_transform(signal)  # Reconstruct signals
    A_ = ica.mixing_

    assert np.allclose(signal,
                       np.dot(S_, A_.T) + ica.mean_), 'Sources are not close'
    return S_, A_


def fit_weighted_ica(signal, n):
    covariance_set = np.empty((n, signal.shape[0], signal.shape[0]))
    cov = Covariances(estimator='oas')

    # Whiten the signal
    for i in range(n):
        index = np.random.choice(signal.shape[1],
                                 signal.shape[1],
                                 replace=False)
        # Weights
        weight = np.random.rand(signal.shape[1])
        m_signal = np.average(signal[:, index], weights=weight, axis=-1)

        # Center the signal
        temp = (signal[:, index] - m_signal[:, np.newaxis]) * np.sqrt(weight)
        epoch = np.expand_dims(temp, axis=0)

        # Take the weighted covariance
        covariance_set[i, :, :] = cov.fit_transform(epoch) / np.sum(weight)

    # Get the best estimate of the diagonalization
    W, D = ajd_pham(covariance_set)

    # Recovered signal
    recovered = np.matmul(W.T, signal)

    # Mixed signal
    mixed = np.matmul(np.linalg.pinv(W.T), recovered)
    assert np.allclose(signal, mixed), 'Sources are not close'

    return W, recovered, mixed

import numpy as np
from scipy import signal


def get_mixed_signals(size):
    ns = np.linspace(0, 100, size[1])
    # Sources with (1) sine wave, (2) saw tooth and (3) random noise
    S = np.array([
        np.sin(ns * 1) + 1,
        signal.sawtooth(ns * 1) + 1,
        np.random.random(len(ns))
    ])

    # Quadratic mixing matrix
    A = np.array([[0.5, 1, 0.2], [1, 0.5, 0.4], [0.5, 0.8, 1]])

    # Mixed signal matrix
    X = A.dot(S)
    return S, X

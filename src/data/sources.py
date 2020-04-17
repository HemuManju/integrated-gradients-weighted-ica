import numpy as np


def get_mixed_signals(size):
    time = np.linspace(0, 5, num=size[1])
    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal

    S = np.c_[s1, s2]
    S += 0.05 * np.random.normal(size=S.shape)  # Add noise
    S /= S.std(axis=0)
    # Unmixed signal
    unmixed = S.T

    # Mixed signal and mixing matrix
    mixing_mat = np.random.rand(size[0], size[0])
    mixed = np.matmul(mixing_mat, unmixed)
    return unmixed, mixed, mixing_mat

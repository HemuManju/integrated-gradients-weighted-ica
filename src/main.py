import yaml
from pathlib import Path

import matplotlib.pyplot as plt

from data.sources import get_mixed_signals
from models.seperate import fit_fast_ica, fit_weighted_ica

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'Run Fast ICA') as check, check():
    size = [2, 500]
    unmixed, mixed = get_mixed_signals(size)

    # Unmix them with fast ICA
    recovered, mixing_fastica = fit_fast_ica(mixed.T)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    for i in range(3):
        ax[i].plot(mixed[i, :], label='Recovered')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    for i in range(3):
        ax[i].plot(recovered[:, i], label='Recovered')
    # plt.show()

with skip_run('run', 'Run Weighted ICA') as check, check():
    size = [3, 500]
    unmixed, mixed = get_mixed_signals(size)

    W, recovered = fit_weighted_ica(mixed, n=100)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    for i in range(3):
        ax[i].plot(mixed[i, :], label='True sources')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    for i in range(3):
        ax[i].plot(recovered[i, :], label='Recovered')
    plt.show()

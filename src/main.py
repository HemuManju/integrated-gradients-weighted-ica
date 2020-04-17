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
    unmixed, mixed, mixing_mat = get_mixed_signals(size)

    # Unmix them with fast ICA
    unmixed_fastica, mixing_fastica = fit_fast_ica(mixed.T)
    plt.plot(unmixed_fastica, label='Recovered')
    plt.plot(unmixed.T, '--', label='True sources')
    plt.legend()
    plt.show()

with skip_run('run', 'Run Weighted ICA') as check, check():
    size = [2, 500]
    unmixed, mixed, mixing_mat = get_mixed_signals(size)

    W, recovered, _ = fit_weighted_ica(mixed, n=10)
    plt.plot(recovered.T, label='Recovered')
    plt.plot(unmixed.T, '--', label='True sources')
    plt.legend()
    plt.show()
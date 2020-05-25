import yaml
from pathlib import Path

import matplotlib.pyplot as plt

from data.sources import get_mixed_signals
from data.create_dataset import create_erp_dataset
from data.utils import save_dataset

from datasets.torch_dataset import subject_independent_data

from models.seperate import fit_fast_ica, fit_weighted_ica
from models.networks import ShallowERPNet
from models.train import train_torch_model
from models.interpret import compute_attribution
from models.utils import (save_trained_pytorch_model,
                          load_trained_pytorch_model)

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
        ax[i].plot(unmixed[i, :], label='Recovered')

    fig, ax = plt.subplots(nrows=3, ncols=1)
    for i in range(3):
        ax[i].plot(recovered[:, i], label='Recovered')
    # plt.show()

with skip_run('skip', 'Run Weighted ICA') as check, check():
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

with skip_run('skip', 'Create ERP dataset') as check, check():
    erp_dataset = create_erp_dataset(config)
    save_dataset(config['erp_dataset_path'], erp_dataset, save=True)

with skip_run('skip', 'Torch model for ERP classification') as check, check():
    data_iterator = subject_independent_data(config)
    network = ShallowERPNet(config['OUTPUT'], config)
    trained_model, model_info = train_torch_model(network,
                                                  config,
                                                  data_iterator,
                                                  new_weights=True)
    save_path = config['trained_model_path']
    save_trained_pytorch_model(trained_model,
                               model_info,
                               save_path,
                               save_model=True)

with skip_run('run', 'Explainability') as check, check():
    data_iterator = subject_independent_data(config)
    trained_model = load_trained_pytorch_model('experiment_0', 1)
    compute_attribution(trained_model, data_iterator)

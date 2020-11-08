import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from torch import nn, optim

from data.create_data import (create_eeg_epochs, create_robot_epochs)
from data.clean_data import clean_eeg_epochs
from data.stats import best_avg_metrics
from data.utils import save_dataset, save_dataframe

from features.psd_features import (eeg_psd_dataset, eeg_psd_df)
from features.eeg_epochs import (eeg_epochs_dataset, eeg_band_dataset,
                                 eeg_ir_matlab_dataset,
                                 weighted_eeg_band_dataset)

from datasets.torch_dataset import (get_psd_dataset, get_eeg_epochs_dataset,
                                    get_weighted_eeg_band_dataset,
                                    get_eeg_ir_matlab_dataset,
                                    get_eeg_ir_subject_wise_split_dataset)

from models.torch_networks import (ShiftScaleEEGClassify, ShallowNet,
                                   ShiftScaleEEGRegress)
from models.torch_train import TorchTrainer
from models.interpret import compute_ablation_scores, compute_ig_attributions
from models.ablation_analysis import selected_psd_features
from models.feature_models import (lda_classify_workload,
                                   lda_classify_workload_pooled_subjects)
from models.utils import CheckpointManager, load_saved_checkpoint, count_parameters

from visualization.visualize import plot_ir_index_hist, plot_selected_sensors

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yaml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'create_eeg_epoch_data') as check, check():
    eeg_epoch_dataset = create_eeg_epochs(config)
    save_path = config['interim_data']['eeg_epoch_path']
    save_dataset(save_path, eeg_epoch_dataset, save=True)

with skip_run('skip', 'clean_eeg_epoch_data') as check, check():
    clean_dataset = clean_eeg_epochs(config)
    save_path = config['processed_data']['clean_eeg_path']
    save_dataset(save_path, clean_dataset, save=True)

with skip_run('skip', 'create_robot_epoch_data') as check, check():
    eeg_epoch_dataset = create_robot_epochs(config)
    save_path = config['interim_data']['robot_epoch_path']
    save_dataset(save_path, eeg_epoch_dataset, save=True)

with skip_run('skip', 'eeg_psd_dataset') as check, check():
    psd_features = eeg_psd_dataset(config, z_score=True)
    save_path = config['processed_data']['psd_features_path']
    save_dataset(save_path, psd_features, save=False)

with skip_run('skip', 'eeg_psd_dataframe') as check, check():
    df_psd = eeg_psd_df(config, z_score=True)
    save_path = config['processed_data']['z_score_psd_power_path']
    save_dataframe(save_path, df_psd, save=True)

with skip_run('skip', 'eeg_epochs_dataset') as check, check():
    psd_features = eeg_epochs_dataset(config)
    save_path = config['processed_data']['eeg_epochs_path']
    save_dataset(save_path, psd_features, save=True)

with skip_run('skip', 'eeg_band_epochs_dataset') as check, check():
    band_epochs = eeg_band_dataset(config)
    save_path = config['processed_data']['eeg_band_epochs_path']
    save_dataset(save_path, band_epochs, save=True)

with skip_run('skip', 'eeg_epochs_with_ir_matlab') as check, check():
    eeg_epochs_ir = eeg_ir_matlab_dataset(config)
    save_path = config['processed_data']['eeg_epochs_ir_matlab_path']
    save_dataset(save_path, eeg_epochs_ir, save=True)

with skip_run('skip', 'shiftscale_psd_training') as check, check():
    data_iterator = get_psd_dataset(config)
    model = ShallowNet(config['OUTPUT'], config)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    # Train
    checkpoint_path = ['psd_models', 'iter_1']
    ckpt_manager = CheckpointManager(config,
                                     checkpoint_path,
                                     model_log_state=False,
                                     visual_log_state=True)
    trainer = TorchTrainer(config, data_iterator, model, criterion, optimizer,
                           ckpt_manager)
    trainer.train()

with skip_run('skip', 'shiftscale_eeg_classification') as check, check():
    for i in range(5):
        data_iterator = get_eeg_epochs_dataset(config)
        model = ShiftScaleEEGClassify(config['OUTPUT'], config)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
        # Train
        iteration = 'iter_' + str(i + 1)
        checkpoint_path = ['eeg_one_sec_models', iteration]
        ckpt_manager = CheckpointManager(config,
                                         checkpoint_path,
                                         model_log_state=True,
                                         visual_log_state=True)
        trainer = TorchTrainer(config, data_iterator, model, criterion,
                               optimizer, ckpt_manager)
        trainer.train()

with skip_run('skip', 'shiftscale_eeg_regression_matlab') as check, check():
    for i in range(5):
        data_iterator = get_eeg_ir_matlab_dataset(config)
        model = ShiftScaleEEGRegress(config['REG_OUTPUT'], config)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=config['LEARNING_RATE'],
                               weight_decay=1e-4)

        # Train
        iteration = 'iter_' + str(i + 1)
        checkpoint_path = ['eeg_ir_models', iteration]
        ckpt_manager = CheckpointManager(config,
                                         checkpoint_path,
                                         model_log_state=True,
                                         visual_log_state=True,
                                         mode='min')
        trainer = TorchTrainer(config,
                               data_iterator,
                               model,
                               criterion,
                               optimizer,
                               ckpt_manager,
                               task_type='regression')
        trainer.train()

with skip_run('skip', 'shiftscale_eeg_regression') as check, check():
    for i in range(2):
        data_iterator = get_eeg_ir_subject_wise_split_dataset(config)
        model = ShiftScaleEEGRegress(config['REG_OUTPUT'], config)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=config['LEARNING_RATE'],
                               weight_decay=1e-4)

        # Train
        iteration = 'iter_' + str(i + 4)
        checkpoint_path = ['eeg_ir_models_subject_wise', iteration]
        ckpt_manager = CheckpointManager(config,
                                         checkpoint_path,
                                         model_log_state=True,
                                         visual_log_state=True,
                                         mode='min')
        trainer = TorchTrainer(config,
                               data_iterator,
                               model,
                               criterion,
                               optimizer,
                               ckpt_manager,
                               task_type='regression')
        trainer.train()

with skip_run('skip', 'compute_ig_attibute_classify') as check, check():
    # Load the data
    data_iterator = get_eeg_epochs_dataset(config)

    # Load a trained model
    checkpoint_dirpath = ['eeg_one_sec_models', 'iter_1']
    trained_model = load_saved_checkpoint(config,
                                          0,
                                          checkpoint_dirpath,
                                          best=True,
                                          apply_weights=True)

    # A wrapper to get the probability
    def forward_func(*inputs):
        return trained_model(*inputs)[1]  # 0 for log_prob, 1 for prob

    # Compute the attributes
    attributions = compute_ig_attributions(config, forward_func, data_iterator)

    # Save them
    save_path = config['processed_data']['ig_attr_classify_path']
    save_dataset(save_path, attributions, save=True)

with skip_run('skip', 'weighted_eeg_band_epochs_dataset') as check, check():
    band_epochs = weighted_eeg_band_dataset(config, use_ig_weights=True)
    save_path = config['processed_data']['weighted_eeg_band_epochs_path']
    save_dataset(save_path, band_epochs, save=True)

with skip_run('skip', 'compute_weighted_ablation_classify') as check, check():
    # Load the data
    data_iterator = get_weighted_eeg_band_dataset(config)

    # Load a trained model
    checkpoint_dirpath = ['eeg_one_sec_models', 'iter_1']
    trained_model = load_saved_checkpoint(config,
                                          0,
                                          checkpoint_dirpath,
                                          best=True,
                                          apply_weights=True)

    # A wrapper to get the probability
    def forward_func(*inputs):
        return trained_model(*inputs)[1]  # 0 for log_prob, 1 for prob

    attributions = compute_ablation_scores(config, forward_func, data_iterator)
    save_path = config['processed_data']['albation_attr_classify_path']
    save_dataset(save_path, attributions, save=True)

with skip_run('skip', 'lda_classification') as check, check():
    # Get the data and split to train and test
    selected_features, pooled_features = selected_psd_features(
        config, exp_type='classification')
    features = pooled_features[0:17]
    print(features)
    lda_classify_workload(config, features)
    lda_classify_workload_pooled_subjects(config, features)

with skip_run('skip', 'compute_ig_attibute_regress') as check, check():
    # Load the data
    data_iterator = get_eeg_ir_subject_wise_split_dataset(config)

    # Load a trained model
    checkpoint_dirpath = ['eeg_ir_models_subject_wise', 'iter_1']
    trained_model = load_saved_checkpoint(config,
                                          0,
                                          checkpoint_dirpath,
                                          best=True,
                                          apply_weights=True)

    # A wrapper to get the probability
    def forward_func(*inputs):
        return trained_model(*inputs)[0]

    # Compute the attributes
    attributions = compute_ig_attributions(config,
                                           forward_func,
                                           data_iterator,
                                           exp_type='regression')

    # Save them
    save_path = config['processed_data']['ig_attr_regress_path']
    save_dataset(save_path, attributions, save=True)

with skip_run('skip', 'weighted_eeg_band_epochs_rg_dataset') as check, check():
    band_epochs = weighted_eeg_band_dataset(config,
                                            exp_type='regression',
                                            use_ig_weights=True)
    save_path = config['processed_data']['weighted_eeg_band_epochs_reg_path']
    save_dataset(save_path, band_epochs, save=True)

with skip_run('skip', 'compute_weighted_ablation_regress') as check, check():
    # Load the data
    data_iterator = get_weighted_eeg_band_dataset(config,
                                                  exp_type='regression')

    # Load a trained model
    checkpoint_dirpath = ['eeg_ir_models_subject_wise', 'iter_1']
    trained_model = load_saved_checkpoint(config,
                                          0,
                                          checkpoint_dirpath,
                                          best=True,
                                          apply_weights=True)

    # A wrapper to get the probability
    def forward_func(*inputs):
        loss = nn.MSELoss(reduce=False)
        output = trained_model(inputs[0])[0]
        squared_error = loss(output, inputs[1])
        return squared_error

    attributions = compute_ablation_scores(config,
                                           forward_func,
                                           data_iterator,
                                           exp_type='regression')
    save_path = config['processed_data']['albation_attr_regress_path']
    save_dataset(save_path, attributions, save=True)

with skip_run('skip', 'lda_regression') as check, check():
    # Get the data and split to train and test
    selected_features, pooled_features = selected_psd_features(
        config, exp_type='regression')
    features = pooled_features[0:10]
    print(features)
    # lda_classify_workload(config, features)
    # lda_classify_workload_pooled_subjects(config, features)

with skip_run('skip', 'lda_classify_n_features') as check, check():
    plt.style.use('clean')
    fig, ax = plt.subplots()
    selected_features, pooled_features = selected_psd_features(
        config, exp_type='classification')
    for i in range(1, 5):
        result = lda_classify_workload(config, features=pooled_features[0:i])
        plt.scatter(i, result, c='#4E79A7')
    plt.grid()
    plt.show()

with skip_run('skip', 'best_avg_metrics') as check, check():
    checkpoint_dirpath = 'eeg_one_sec_models'
    avg = best_avg_metrics(config,
                           checkpoint_dirpath,
                           metric_type='classification')

    checkpoint_dirpath = 'eeg_ir_models_subject_wise'
    avg = best_avg_metrics(config,
                           checkpoint_dirpath,
                           metric_type='regression')
    print(avg)

with skip_run('skip', 'plot_ir_index_hist') as check, check():
    plot_ir_index_hist(config)

with skip_run('skip', 'plot_selected_sensors') as check, check():
    selected_features, pooled_features = selected_psd_features(
        config, exp_type='classification')
    features = pooled_features[0:18]
    plot_selected_sensors(features)

with skip_run('skip', 'number_of_parameters') as check, check():
    checkpoint_dirpath = ['eeg_ir_models', 'iter_1']
    trained_model = load_saved_checkpoint(config,
                                          0,
                                          checkpoint_dirpath,
                                          best=True,
                                          apply_weights=True)
    count_parameters(trained_model)

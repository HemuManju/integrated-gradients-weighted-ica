import deepdish as dd
import numpy as np

from features.instability import instability_index
from data.utils import nested_dict

import matplotlib.pyplot as plt  # noqa

from .utils import convert_to_array, binary_mask_from_ig_weights


def extract_eeg_array_with_targets(subject, trial, config):
    # Data path and Data
    read_path = config['processed_data']['clean_eeg_path']
    epochs = dd.io.load(read_path, group='/' + subject + '/eeg/' + trial)
    data = epochs.get_data()[:, :, 0:config['epoch_length'] * 250] * 1e6
    # Convert them to array
    x_array, y_array = convert_to_array(data, trial, config)

    return x_array, y_array


def eeg_epochs_dataset(config):
    """Create pytorch dataset for all subjects.

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    trial : string
        e.g. HighFine, HighGross, LowFine, LowGross, AdoptComb, HighComb etc.

    Returns
    -------
    tensors
        All the data from subjects with targets.

    """
    # Parameters
    eeg_epochs = {}
    x = []
    y = []

    for subject in config['subjects']:
        for trial in config['trials']:
            if subject not in config['test_subjects']:
                x_array, y_array = extract_eeg_array_with_targets(
                    subject, trial, config)
                x.append(x_array)
                y.append(y_array)

    # Convert to array
    eeg_epochs['features'] = np.concatenate(x, axis=0)
    eeg_epochs['targets'] = np.concatenate(y, axis=0)

    return eeg_epochs


def extract_eeg_band_array_with_targets(subject, trial, config):
    # Data path and Data
    read_path = config['processed_data']['clean_eeg_path']
    epochs = dd.io.load(read_path, group='/' + subject + '/eeg/' + trial)

    # Band pass frequency bands
    x_array, y_array = [], []
    for freq in config['freq_bands']:
        temp_epochs = epochs.copy()
        filtered_epoch = temp_epochs.filter(l_freq=freq[0],
                                            h_freq=freq[1],
                                            fir_design='firwin')
        data = filtered_epoch.get_data()[:, 0:config['n_electrodes'],
                                         0:250 * config['epoch_length']] * 1e6
        # Convert them to array
        x_temp, y_temp = convert_to_array(data, trial, config)
        x_array.append(x_temp[:, None, :, :])

    # Add the epoch data also
    x_array.append(epochs.get_data()[:, None, 0:config['n_electrodes'],
                                     0:250 * config['epoch_length']] * 1e6)

    x_array = np.concatenate(x_array, axis=1)
    y_array = y_temp

    return x_array, y_array


def eeg_band_dataset(config):
    """Create pytorch dataset for all subjects.

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    trial : string
        e.g. HighFine, HighGross, LowFine, LowGross, AdoptComb, HighComb etc.

    Returns
    -------
    tensors
        All the data from subjects with targets.

    """
    # Parameters
    eeg_band_epochs = {}
    x = []
    y = []

    for subject in config['subjects']:
        for trial in config['trials']:
            if subject not in config['test_subjects']:
                x_array, y_array = extract_eeg_band_array_with_targets(
                    subject, trial, config)
                x.append(x_array)
                y.append(y_array)

    # Convert to array
    eeg_band_epochs['features'] = np.concatenate(x, axis=0)
    eeg_band_epochs['targets'] = np.concatenate(y, axis=0)

    return eeg_band_epochs


def extract_eeg_array_with_ir(subject, config):
    # Variables
    x, y = [], []

    for trial in config['trials']:
        if trial not in ['HighGross', 'LowGross']:
            read_path = config['processed_data']['clean_eeg_path']
            epochs = dd.io.load(read_path,
                                group='/' + subject + '/eeg/' + trial)
            data = epochs.get_data()[:, :,
                                     0:config['epoch_length'] * 250] * 1e6
            # Convert them to array
            x_array, y_array = convert_to_array(data, trial, config)

            # Replace y_array with ir index
            ir_index = instability_index(subject, trial, config)

            # For verification
            if y_array.shape[0] != ir_index.shape[0]:
                raise Exception('Two epochs are not of same length!')
            else:
                y_array = ir_index
            x.append(x_array)
            y.append(y_array)

    x_array = np.concatenate(x, axis=0)
    y_array = np.concatenate(y, axis=0)

    # Normalise
    y_array = y_array - np.mean(y_array)

    return x_array, y_array


def eeg_ir_dataset(config):
    # Parameters
    eeg_epochs = nested_dict()

    for subject in config['subjects']:
        x_array, y_array = extract_eeg_array_with_ir(subject, config)
        # Convert to array
        eeg_epochs[subject]['features'] = x_array
        eeg_epochs[subject]['targets'] = y_array

    return eeg_epochs


def eeg_array_with_ir_matlab(subject, config):
    # Variables
    x, y = [], []

    for trial in config['trials']:
        if trial not in ['HighGross', 'LowGross']:

            # Create an epoch object
            # epochs = eeg_epochs_from_csv(config, subject, trial)
            read_path = config['processed_data']['clean_eeg_path']
            epochs = dd.io.load(read_path,
                                group='/' + subject + '/eeg/' + trial)
            data = epochs.get_data()[:, :,
                                     0:config['epoch_length'] * 250] * 1e6

            # Convert them to array
            x_array, y_array = convert_to_array(data, trial, config)

            # Replace y_array with ir index
            read_path = config['external_data'] + subject + '/' + trial
            ir_index = np.genfromtxt(read_path + '_IR.csv', delimiter=',')
            ir_index = np.nan_to_num(ir_index)

            # Delete the epoch which is deleted in EEG
            drop_id = [id for id, val in enumerate(epochs.drop_log) if val]
            ir_index_temp = np.delete(ir_index, drop_id)

            # For verification
            if y_array.shape[0] != ir_index_temp.shape[0]:
                print('Two epochs are not of same length!')

            y_array = ir_index_temp[0:y_array.shape[0], None]
            x.append(x_array)
            y.append(y_array)

    x_array = np.concatenate(x, axis=0)
    y_array = np.concatenate(y, axis=0)

    # Mean shift to zero
    y_array = y_array - np.mean(y_array)
    return x_array, y_array


def eeg_ir_matlab_dataset(config):
    # Parameters
    eeg_epochs = nested_dict()

    for subject in config['subjects']:
        x_array, y_array = eeg_array_with_ir_matlab(subject, config)
        # Convert to array
        eeg_epochs[subject]['features'] = x_array
        eeg_epochs[subject]['targets'] = y_array

    return eeg_epochs


def weighted_eeg_band_dataset(config,
                              exp_type='classification',
                              use_ig_weights=False):

    if exp_type == 'classification':
        read_path = config['processed_data']['ig_attr_classify_path']
    else:
        read_path = config['processed_data']['ig_attr_regress_path']

    # Load the data
    saved_data = dd.io.load(read_path)
    epochs = saved_data['epochs']

    # Quantize the weights
    weights = binary_mask_from_ig_weights(saved_data['ig_attributions'])

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # test = saved_data['ig_attributions'][0, :, :]
    # ax[0].imshow(test, aspect='auto')
    # ax[1].imshow(weights[0, :, :], aspect='auto')
    # plt.show()

    # Band pass frequency bands
    x_array = []
    for freq in config['freq_bands']:
        temp_epochs = epochs.copy()
        if use_ig_weights:
            temp_epochs._data = temp_epochs._data * weights
        filtered_epoch = temp_epochs.filter(l_freq=freq[0],
                                            h_freq=freq[1],
                                            fir_design='firwin',
                                            verbose=False)
        data = filtered_epoch.get_data()[:, 0:config['n_electrodes'],
                                         0:250 * config['epoch_length']] * 1e6
        # Convert them to array
        x_array.append(data[:, None, :, :])

    # Add the epoch data also
    x_array.append(epochs.get_data()[:, None, 0:config['n_electrodes'],
                                     0:250 * config['epoch_length']] * 1e6)

    x_array = np.concatenate(x_array, axis=1)
    weighted_eeg = {}
    weighted_eeg['features'] = x_array
    weighted_eeg['targets'] = saved_data['targets']

    return weighted_eeg

import numpy as np
import deepdish as dd
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .utils import get_train_valid_test_split


class TorchDataset(Dataset):
    """All subject dataset class.

    Parameters
    ----------
    split_ids : list
        ids list of training or validation or traning data.

    Attributes
    ----------
    split_ids

    """
    def __init__(self, features, targets):
        super(TorchDataset, self).__init__()
        # Convert from numpy array
        self.features = torch.from_numpy(features).type(torch.float32)
        self.targets = torch.from_numpy(targets).type(torch.float32)
        print(self.targets.shape)

    def __getitem__(self, index):
        # Read only specific data and convert to torch tensors
        x = self.features[index, :, :]
        y = self.targets[index, :]
        return x, y

    def __len__(self):
        return self.features.shape[0]


def get_eeg_epochs_dataset(config):
    # Load the data
    read_path = config['processed_data']['eeg_epochs_path']
    data = dd.io.load(read_path)
    features = data['features']
    targets = data['targets']

    # Get train, valid, and test split
    indices = get_train_valid_test_split(targets, config)

    data_iterator = {}
    for key in indices:
        if indices[key] is not None:
            dataset = TorchDataset(features[indices[key]],
                                   targets[indices[key]])
            data_iterator[key] = DataLoader(dataset,
                                            batch_size=config['BATCH_SIZE'],
                                            shuffle=True,
                                            num_workers=10)

    return data_iterator


def get_psd_dataset(config):
    # Load the data
    read_path = config['processed_data']['psd_features_path']
    data = dd.io.load(read_path)
    features = data['features']
    targets = data['targets']

    # Get train, valid, and test split
    indices = get_train_valid_test_split(targets, config)

    data_iterator = {}
    for key in indices:
        if indices[key] is not None:
            dataset = TorchDataset(features[indices[key]],
                                   targets[indices[key]])
            data_iterator[key] = DataLoader(dataset,
                                            batch_size=config['BATCH_SIZE'],
                                            shuffle=True,
                                            num_workers=10)

    return data_iterator


def get_eeg_band_dataset(config):
    # Load the data
    read_path = config['processed_data']['eeg_band_epochs_path']
    data = dd.io.load(read_path)
    features = data['features']
    targets = data['targets']

    # Get train, valid, and test split
    indices = get_train_valid_test_split(targets, config)

    data_iterator = {}
    for key in indices:
        if indices[key] is not None:
            dataset = TorchDataset(features[indices[key]],
                                   targets[indices[key]])
            data_iterator[key] = DataLoader(dataset,
                                            batch_size=config['BATCH_SIZE'],
                                            shuffle=True,
                                            num_workers=10)

    return data_iterator


def get_weighted_eeg_band_dataset(config, exp_type='classification'):
    if exp_type == 'classification':
        # Load the data
        read_path = config['processed_data']['weighted_eeg_band_epochs_path']
    else:
        read_path = config['processed_data'][
            'weighted_eeg_band_epochs_reg_path']

    data = dd.io.load(read_path)
    features = data['features']
    targets = data['targets']

    data_iterator = {}
    dataset = TorchDataset(features, targets)
    data_iterator['testing'] = DataLoader(dataset,
                                          batch_size=config['BATCH_SIZE'],
                                          shuffle=True,
                                          num_workers=10)

    return data_iterator


def get_eeg_ir_matlab_dataset(config, leave_out_sub_id=None):
    # Load the data
    read_path = config['processed_data']['eeg_epochs_ir_matlab_path']
    data = dd.io.load(read_path)

    if leave_out_sub_id is not None:
        left_out_subject = config['subjects'][leave_out_sub_id]
        config['subjects'].pop(leave_out_sub_id)

    # Parameters
    data_iterator = {}
    x = []
    y = []

    for subject in config['subjects']:
        if subject not in config['test_subjects']:
            x_array = data[subject]['features']
            y_array = data[subject]['targets']
            x.append(x_array)
            y.append(y_array)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    if leave_out_sub_id is not None:
        # Training
        dataset = TorchDataset(x, y)
        data_iterator['training'] = DataLoader(dataset,
                                               batch_size=config['BATCH_SIZE'],
                                               shuffle=True,
                                               num_workers=10)
        # Testing
        dataset = TorchDataset(data[left_out_subject]['features'],
                               data[left_out_subject]['targets'])
        data_iterator['testing'] = DataLoader(dataset,
                                              batch_size=config['BATCH_SIZE'],
                                              shuffle=True,
                                              num_workers=10)
    else:
        # Get train, valid, and test split
        indices = get_train_valid_test_split(y, config)

        data_iterator = {}
        for key in indices:
            if indices[key] is not None:
                dataset = TorchDataset(x[indices[key]], y[indices[key]])
                data_iterator[key] = DataLoader(
                    dataset,
                    batch_size=config['BATCH_SIZE'],
                    shuffle=True,
                    num_workers=10)

    return data_iterator


def get_eeg_ir_matlab_subject_dataset(config, subject):
    # Load the data
    read_path = config['processed_data']['eeg_epochs_ir_matlab_path']
    data = dd.io.load(read_path)

    x = data[subject]['features']
    y = data[subject]['targets']

    # Get train, valid, and test split
    config['TEST_SIZE'] = 0.7
    indices = get_train_valid_test_split(y, config)

    data_iterator = {}
    for key in indices:
        if indices[key] is not None:
            dataset = TorchDataset(x[indices[key]], y[indices[key]])
            data_iterator[key] = DataLoader(dataset,
                                            batch_size=config['BATCH_SIZE'],
                                            shuffle=True,
                                            num_workers=10)
    return data_iterator


def get_eeg_ir_subject_wise_split_dataset(config):
    # Load the data
    read_path = config['processed_data']['eeg_epochs_ir_matlab_path']
    data = dd.io.load(read_path)

    # Parameters
    data_iterator = {}
    x_train, y_train = [], []
    x_test, y_test = [], []

    for subject in config['subjects']:
        x_array = data[subject]['features']
        y_array = data[subject]['targets']

        # Split train and test
        indices = get_train_valid_test_split(y_array, config)

        # Training data
        x_train.append(x_array[indices['training']])
        y_train.append(y_array[indices['training']])

        # Testing data
        x_test.append(x_array[indices['testing']])
        y_test.append(y_array[indices['testing']])

    # Concatenate them
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Torch data iterators
    data_iterator['training'] = DataLoader(TorchDataset(x_train, y_train),
                                           batch_size=config['BATCH_SIZE'],
                                           shuffle=True,
                                           num_workers=10)
    data_iterator['testing'] = DataLoader(TorchDataset(x_test, y_test),
                                          batch_size=config['BATCH_SIZE'],
                                          shuffle=True,
                                          num_workers=10)
    return data_iterator

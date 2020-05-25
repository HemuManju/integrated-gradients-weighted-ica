from pathlib import Path

import torch
import numpy as np
import deepdish as dd

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .utils import label_stats


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
    def __init__(self, features, labels, split_ids):
        super(TorchDataset, self).__init__()
        self.split_ids = split_ids
        self.features = features[self.split_ids]
        self.labels = labels[self.split_ids, :]

    def __getitem__(self, index):
        # Read only specific data and convert to torch tensors
        x = torch.from_numpy(self.features[index]).type(torch.float32)
        y = torch.from_numpy(self.labels[index, :]).type(torch.float32)
        return x, y

    def __len__(self):
        return self.features.shape[0]


def get_data_split_ids(labels, leave_tags, test_size=0.15):
    """Generators training, validation, and training
    indices to be used by Dataloader.

    Parameters
    ----------
    labels : array
        An array of labels.
    test_size : float
        Test size e.g. 0.15 is 15% of whole data.

    Returns
    -------
    dict
        A dictionary of ids corresponding to train, validate, and test.

    """

    # Create an empty dictionary
    split_ids = {}

    if (leave_tags == 0).any():
        train_id = np.nonzero(leave_tags)[0]
        test_id = np.nonzero(1 - leave_tags)[0]
        test_id, validate_id, _, _ = train_test_split(test_id,
                                                      test_id * 0,
                                                      test_size=0.5)
    else:
        ids = np.arange(labels.shape[0])
        train_id, test_id, _, _ = train_test_split(ids,
                                                   ids * 0,
                                                   test_size=2 * test_size)
        test_id, validate_id, _, _ = train_test_split(test_id,
                                                      test_id * 0,
                                                      test_size=0.5)
    split_ids['training'] = train_id
    split_ids['validation'] = validate_id
    split_ids['testing'] = test_id

    return split_ids


def train_test_iterator(config, test_subjects, leave_out=False, cov=False):
    """A function to get train, validation, and test data.

    Parameters
    ----------
    features : array
        An array of features.
    labels : array
        True labels.
    leave_tags : array
        An array specifying whether a subject was left out of training.
    config : yaml
        The configuration file.
    leave_out : bool
        Whether to leave out some subjects training and use them in testing

    Returns
    -------
    dict
        A dict containing the train and test data.

    """
    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    TEST_SIZE = config['TEST_SIZE']

    # Get the features and labels
    if leave_out:
        features, labels, leave_tags = subject_dependent_data(
            config, test_subjects)
    else:
        features, labels, leave_tags = subject_independent_data(config)

    # Get training, validation, and testing split_ids
    split_ids = get_data_split_ids(labels, leave_tags, test_size=TEST_SIZE)

    # Initialise an empty dictionary
    data_iterator = {}

    # Create train, validation, test datasets and save them in a dictionary
    train_data = TorchDataset(features, labels, split_ids['training'])
    data_iterator['training'] = DataLoader(train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=10)

    valid_data = TorchDataset(features, labels, split_ids['validation'])
    data_iterator['validation'] = DataLoader(valid_data,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=10)

    test_data = TorchDataset(features, labels, split_ids['testing'])
    data_iterator['testing'] = DataLoader(test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=10)

    return data_iterator


def subject_independent_data(config):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels, leave_leave_tags
        2 arrays features and labels.
        A tag determines whether the data point is used in training.

    """

    path = str(Path(__file__).parents[2] / config['erp_dataset_path'])
    data = dd.io.load(path)

    # Parameters
    TEST_SIZE = config['TEST_SIZE']
    BATCH_SIZE = config['BATCH_SIZE']
    subjects = config['subjects']
    session = config['session']

    # Empty array (list)
    x = []
    y = []

    for subject in subjects:
        x_temp = data['subject_' + subject][session]['epochs'].get_data()
        y_temp = data['subject_' + subject][session]['labels']
        x.append(x_temp)
        y.append(y_temp)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]

    # Create train and test
    ids = np.arange(labels.shape[0])
    train_id, test_id, _, _ = train_test_split(ids,
                                               ids * 0,
                                               test_size=TEST_SIZE)

    # Create train and test torch datasets and save them in a dictionary
    data_iterator = {}
    train_data = TorchDataset(features, labels, train_id)
    data_iterator['training'] = DataLoader(train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=10)
    # Create test dataiterator
    test_data = TorchDataset(features, labels, test_id)

    data_iterator['testing'] = DataLoader(test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=10)

    label_stats(data_iterator)

    return data_iterator


def subject_dependent_data(config, test_subjects):
    """Get subject dependent data.

    Parameters
    ----------
    config : yaml
        The configuration file

    Returns
    -------
    features, labels
        2 arrays features and labels

    """

    path = str(Path(__file__).parents[2] / config['erp_dataset_path'])
    data = dd.io.load(path)

    # Parameters
    subjects = config['subjects']
    session = config['session']

    # Empty array (list)
    x = []
    y = []
    leave_tags = np.empty((0, 1))

    for subject in subjects:
        x_temp = data['subject_' + subject][session]['epochs'].get_data()
        y_temp = data['subject_' + subject][session]['labels']
        x.append(x_temp)
        y.append(y_temp)
        if subject in test_subjects:
            leave_tags = np.concatenate((leave_tags, y_temp[:, 0:1] * 0),
                                        axis=0)
        else:
            leave_tags = np.concatenate((leave_tags, y_temp[:, 0:1] * 0 + 1),
                                        axis=0)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)

    # Store them in dictionary
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]
    leave_tags = leave_tags[rus.sample_indices_, :]

    return features, labels, leave_tags


def subject_specific_data(config, subjects):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    config : yaml
        The configuration file
    subject : list
        The subject ID for whom we need the data

    Returns
    -------
    data_iterator
        A pytorch data iterator

    """

    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    TEST_SIZE = config['TEST_SIZE']
    session = config['session']

    path = str(Path(__file__).parents[2] / config['erp_dataset_path'])
    data = dd.io.load(path)

    # Empty array (list)
    x = []
    y = []

    for subject in subjects:
        x_temp = data['subject_' + subject][session]['epochs'].get_data()
        y_temp = data['subject_' + subject][session]['labels']
        x.append(x_temp)
        y.append(y_temp)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]

    # Create train and test
    ids = np.arange(labels.shape[0])
    train_id, test_id, _, _ = train_test_split(ids,
                                               ids * 0,
                                               test_size=1 - 2 * TEST_SIZE)

    # Create train and test torch datasets and save them in a dictionary
    data_iterator = {}
    train_data = TorchDataset(features, labels, train_id)
    data_iterator['training'] = DataLoader(train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=10)
    # Create test dataiterator
    test_data = TorchDataset(features, labels, test_id)
    data_iterator['testing'] = DataLoader(test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=10)

    return data_iterator


def subject_specific_data_with_validation(config, subjects):
    """Get subject independent data (pooled data).

    Parameters
    ----------
    config : yaml
        The configuration file
    subject : list
        The subject ID for whom we need the data

    Returns
    -------
    data_iterator
        A pytorch data iterator

    """

    # Parameters
    BATCH_SIZE = config['BATCH_SIZE']
    TEST_SIZE = config['TEST_SIZE']
    session = config['session']

    path = str(Path(__file__).parents[2] / config['erp_dataset_path'])
    data = dd.io.load(path)

    # Empty array (list)
    x = []
    y = []

    for subject in subjects:
        x_temp = data['subject_' + subject][session]['epochs'].get_data()
        y_temp = data['subject_' + subject][session]['labels']
        x.append(x_temp)
        y.append(y_temp)

    # For validation
    x_valid = []
    y_valid = []

    remaining_subjects = np.setdiff1d(config['subjects'], subjects)
    for subject in remaining_subjects:
        x_temp = data['subject_' + subject][session]['epochs'].get_data()
        y_temp = data['subject_' + subject][session]['labels']
        x_valid.append(x_temp)
        y_valid.append(y_temp)

    # Convert to array
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    # Balance the dataset
    rus = RandomUnderSampler()
    rus.fit_resample(y, y)
    features = x[rus.sample_indices_, :, :]
    labels = y[rus.sample_indices_, :]

    # Create train and test
    ids = np.arange(labels.shape[0])
    train_id, test_id, _, _ = train_test_split(ids,
                                               ids * 0,
                                               test_size=1 - 2 * TEST_SIZE)

    # Create train and test torch datasets and save them in a dictionary
    data_iterator = {}
    train_data = TorchDataset(features, labels, train_id)
    data_iterator['training'] = DataLoader(train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=10)

    # Validation set from previously trained subjects
    valid_data = TorchDataset(x_valid, y_valid, np.arange(y_valid.shape[0]))
    data_iterator['validation'] = DataLoader(valid_data,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=10)
    # Create test dataiterator
    test_data = TorchDataset(features, labels, test_id)
    data_iterator['testing'] = DataLoader(test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=10)

    return data_iterator

import os
import mne
import numpy as np
from scipy.io import loadmat

from .utils import nested_dict


def get_data_path(subject, config):
    path_folder_subject = config['raw_data_path'] + '/subject_' + str(
        subject).zfill(2) + os.sep

    # filter the data regarding the experimental conditions
    subject_paths = []
    for session in [1, 2, 3]:
        path = 'subject_' + str(subject).zfill(2) + '_session_' + str(
            session).zfill(2) + '.mat'
        subject_paths.append(path_folder_subject + path)

    return subject_paths


def get_single_subject_data(subject, config):
    """Return data for a single subject

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    config : yaml
        The configuration file.

    Returns
    -------
    sessions : dict
        A dictionary
    """
    file_path_list = get_data_path(subject, config)

    sessions = {}
    for file_path, session in zip(file_path_list, [1, 2, 3]):

        session_name = 'session_' + str(session)
        sessions[session_name] = {}
        run_name = 'run_1'

        chnames = [
            'Fp1', 'Fp2', 'AFz', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
            'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
            'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'O1', 'Oz', 'O2', 'PO8',
            'PO9', 'PO10', 'STI 014'
        ]

        chtypes = ['eeg'] * 32 + ['stim']

        D = loadmat(file_path)['DATA'].T
        S = D[1:33, :]
        stim = D[-2, :] + D[-1, :]
        X = np.concatenate([S, stim[None, :]])

        info = mne.create_info(ch_names=chnames,
                               sfreq=512,
                               ch_types=chtypes,
                               montage='standard_1020',
                               verbose=False)
        raw = mne.io.RawArray(data=X, info=info, verbose=False)
        sessions[session_name][run_name] = raw

    return sessions


def create_erp_dataset(config):
    """Create the ERP dataset of all the subjects. Most of the code is from

    """
    erp_dataset = nested_dict()
    # NOTE: that subject 31 at session 3 has a few samples which are 'nan'
    # to avoid this problem I dropped the epochs having this condition

    # Load data
    for subject in config['subjects']:
        sessions = get_single_subject_data(subject, config)

        for session in sessions.keys():
            raw = sessions[session]['run_1']
            # Filter data and resample
            fmin = 1
            fmax = 24
            raw.filter(fmin, fmax, verbose=False)

            # detect the events and cut the signal into epochs
            events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
            event_id = {'NonTarget': 1, 'Target': 2}
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin=0.0,
                                tmax=0.8,
                                baseline=None,
                                verbose=False,
                                preload=True)
            epochs.pick_types(eeg=True)
            labels = epochs.events[:, -1] - 1

            erp_dataset[subject][session]['epochs'] = epochs
            erp_dataset[subject][session]['labels'] = labels

    return erp_dataset

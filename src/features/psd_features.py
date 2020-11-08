import deepdish as dd
import numpy as np
import pandas as pd

from scipy.integrate import simps
from sklearn.preprocessing import StandardScaler

import mne
from mne.time_frequency import psd_multitaper

from .utils import convert_to_array


def subject_psd_array(subject, trial, config):
    # Data path and Data
    read_path = config['processed_data']['clean_eeg_path']
    epochs = dd.io.load(read_path, group='/' + subject + '/eeg/' + trial)

    # PSD values and class targets
    psds, freqs = psd_multitaper(epochs,
                                 fmin=1.0,
                                 fmax=64.0,
                                 n_jobs=6,
                                 verbose=False)
    # Normalise the PSD values
    total_power = simps(psds, dx=np.mean(np.diff(freqs)))
    normalised_power = psds / total_power[:, :, None]
    scaled_power = normalised_power * config['SCALE_FACTOR']  # Scaled

    # Convert them to array
    x_array, y_array = convert_to_array(scaled_power, trial, config)

    return x_array, y_array


def eeg_psd_dataset(config):
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
    psd_features = {}
    x = []
    y = []

    for subject in config['subjects']:
        for trial in config['trials']:
            if subject not in config['test_subjects']:
                x_array, y_array = subject_psd_array(subject, trial, config)
                x.append(x_array)
                y.append(y_array)

    # Convert to array
    psd_features['features'] = np.concatenate(x, axis=0)
    psd_features['targets'] = np.concatenate(y, axis=0)

    return psd_features


def subject_eeg_psd_df(subject, config, z_score=False):
    """Calculate the band power of EEG signals.

    Parameters
    ----------
    subject : str
        String of subject ID e.g. 8801.
    config : yaml file
        Configuration file.
    z_score : bool
        whether to perform z-score transformation or not.

    Returns
    -------
    df : pandas dataframe
        6 band powers of at different sensor locations.
    """
    # Data path and Data
    read_path = config['processed_data']['clean_eeg_path']
    df_subject_psd = pd.DataFrame()
    scaler = StandardScaler()

    for trial in config['trials']:
        epochs = dd.io.load(read_path, group='/' + subject + '/eeg/' + trial)
        picks = mne.pick_types(epochs.info, eeg=True)
        ch_names = epochs.ch_names[picks[0]:picks[-1] + 1]
        psds, freqs = psd_multitaper(epochs,
                                     fmin=1.0,
                                     fmax=64.0,
                                     picks=picks,
                                     n_jobs=6,
                                     verbose=False,
                                     normalization='full')
        psd_band = []
        for freq_band in config['freq_bands']:
            psd_band.append(psds[:, :, (freqs >= freq_band[0]) &
                                 (freqs <= freq_band[1])].mean(axis=-1))
        # Form pandas dataframe
        data = np.concatenate(psd_band, axis=1)
        columns = [x + '_' + y for x in ch_names for y in config['band_names']]
        df = pd.DataFrame(np.log10(data), columns=columns)

        # Add class targets
        if (trial == 'HighFine') or (trial == 'LowGross'):
            df['class'] = [[1, 0]] * psds.shape[0]
            df['class_label'] = [0] * psds.shape[0]
        if (trial == 'HighGross') or (trial == 'LowFine'):
            df['class'] = [[0, 1]] * psds.shape[0]
            df['class_label'] = [1] * psds.shape[0]

        # Append
        df_subject_psd = df_subject_psd.append(df, ignore_index=True)

    # Normalization
    if z_score:
        z_score_array = scaler.fit_transform(df_subject_psd[columns])
        df_subject_psd[columns] = z_score_array
        df_subject_psd['subject'] = str(subject)
    return df_subject_psd


def eeg_psd_df(config, z_score=False):
    # Parameters
    df_psd = pd.DataFrame()
    for subject in config['subjects']:
        df_subject_psd = subject_eeg_psd_df(subject, config, z_score=z_score)
        df_psd = df_psd.append(df_subject_psd)

    return df_psd

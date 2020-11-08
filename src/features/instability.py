import numpy as np
import deepdish as dd
from scipy.signal import welch


def interaction_band_pow(epochs, config):
    """Get the band power (psd) of the interaction forces/moments.

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine, AdaptFine.

    Returns
    -------
    band_pow : array
        An array of band_pow of each epoch of data.
    freqs : array
        An array of frequencies in power spectral density.

    """
    data = epochs.get_data()
    band_pow = []
    for i in range(epochs.__len__()):
        # Last row in smooth force
        freqs, power = welch(data[i, -1, :],
                             fs=128,
                             nperseg=64,
                             nfft=128,
                             detrend=False)
        band_pow.append(power)
    psds = np.sqrt(np.array(band_pow) * 128 / 2)
    return psds, freqs


def instability_index(subject, trial, config):
    """Calculate instability index of the subject and trial.

    Parameters
    ----------
    subject : string
        subject ID e.g. 7707.
    trial : string
        trial e.g. HighFine, AdaptFine.
    config : yaml file
        The configuration file

    Returns
    -------
    ins_index : array
        An array of instability index calculated at each epochs.

    """
    # signature of data: x(n_epochs, 3 channels, frequencies: 0-128 Hz)
    read_path = config['interim_data']['robot_epoch_path']
    epochs = dd.io.load(read_path, group='/' + subject + '/robot/' + trial)

    data, freqs = interaction_band_pow(epochs, config)

    # Get frequency index between 3 Hz and 12 Hz
    f_critical_max = (freqs > 2.35) & (freqs <= 10.0)
    f_min_max = (freqs > 0.25) & (freqs <= 10.0)

    num = data[:, f_critical_max].sum(axis=-1)
    den = data[:, f_min_max].sum(axis=-1)
    ir_index = num / den

    return ir_index

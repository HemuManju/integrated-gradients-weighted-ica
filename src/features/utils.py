import numpy as np
import mne


def binary_mask_from_ig_weights(ig_weights, normalize=False):
    if normalize:
        ig_weight_mask = np.zeros(ig_weights.shape)
        for i in range(ig_weights.shape[1]):
            temp = ig_weights[i, :, :]

            # Now make 0-25% and 75-100% binary
            lower_tail = np.percentile(temp, 25)
            upper_tail = np.percentile(temp, 75)
            mask = (temp <= lower_tail) + (temp >= upper_tail)
            temp[mask] = 1
            temp[~mask] = 0
            ig_weight_mask[i, :, :] = temp
    else:
        ig_weights[ig_weights <= 0] = 0
        ig_weights[ig_weights > 0] = 1
        ig_weight_mask = ig_weights

    return ig_weight_mask


def eeg_epochs_from_csv(config, subject, trial):
    read_path = config['external_data'] + subject + '/' + trial
    data = np.genfromtxt(read_path + '_EEG.csv', delimiter=',')

    # Create mne epoch object
    info = mne.create_info(ch_names=[
        'Fp1', 'F7', 'F8', 'T4', 'T6', 'T5', 'T3', 'Fp2', 'O1', 'P3', 'Pz',
        'F3', 'Fz', 'F4', 'C4', 'P4', 'POz', 'C3', 'Cz', 'O2'
    ],
                           ch_types=['misc'] * data.shape[0],
                           sfreq=256.0)
    raw = mne.io.RawArray(data, info, verbose=False)
    events = mne.make_fixed_length_events(raw,
                                          duration=config['epoch_length'],
                                          overlap=config['overlap'])
    epochs = mne.Epochs(raw,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        baseline=(0, 0),
                        verbose=False)
    return epochs


def convert_to_array(epoch, trial, config):
    """Converts the edf files in eeg and robot dataset into arrays.

    Parameters
    ----------
    subject : string
        Subject ID e.g. 7707.
    trial : string
        e.g. HighFine, HighGross, LowFine, LowGross, AdoptComb, HighComb etc.

    Returns
    -------
    tensors
        x and y arrays corresponding to the subject and trial.

    """
    # Parameters
    n_electrodes = config['n_electrodes']

    try:
        x = epoch.get_data()
    except Exception:
        x = epoch

    y = np.zeros((x.shape[0], config['n_classes']))

    # Convert class targets to one_hot
    if (trial == 'HighFine') or (trial == 'LowGross'):
        y[:, 0] = 1  # [1, 0]
    if (trial == 'HighGross') or (trial == 'LowFine'):
        y[:, 1] = 1  # [0, 1]

    x_array = np.float32(x[:, 0:n_electrodes, :])
    y_array = y

    return x_array, y_array

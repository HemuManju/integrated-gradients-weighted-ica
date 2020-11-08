import mne
import deepdish as dd

from .utils import nested_dict


def eeg_epochs_dataset(subject, trial, config, preload=True):
    """Get the epoched eeg data excluding unnessary channels
    from fif file and also filter the signal.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    epochs  : epoched data

    """
    epoch_length = config['epoch_length']
    overlap = config['overlap']

    read_path = config['raw_data']['raw_eeg_path']
    raw_eeg = dd.io.load(read_path, group='/' + subject + '/eeg/' + trial)

    events = mne.make_fixed_length_events(raw_eeg,
                                          duration=epoch_length,
                                          overlap=epoch_length * overlap)
    epochs = mne.Epochs(raw_eeg,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        verbose=False,
                        baseline=(0, 0),
                        preload=preload)

    return epochs


def create_eeg_epochs(config):
    """Create the data with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    eeg_epoch_dataset : dataset of all the subjects with different conditions

    """
    eeg_epoch_dataset = {}
    for subject in config['subjects']:
        data = nested_dict()
        for trial in config['trials']:
            epochs = eeg_epochs_dataset(subject, trial, config)
            data['eeg'][trial] = epochs
        eeg_epoch_dataset[subject] = data

    return eeg_epoch_dataset


def robot_epochs(subject, trial, config, preload=True):
    """Get the epoched eeg data excluding unnessary channels
    from fif file and also filter the signal.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    epochs  : epoched data

    """
    epoch_length = config['epoch_length']
    overlap = config['overlap']

    # Load raw force data
    read_path = config['raw_data']['raw_robot_path']
    raw_force = dd.io.load(read_path, group='/' + subject + '/robot/' + trial)

    events = mne.make_fixed_length_events(raw_force,
                                          duration=epoch_length,
                                          overlap=epoch_length * overlap)
    epochs = mne.Epochs(raw_force,
                        events,
                        tmin=0,
                        tmax=config['epoch_length'],
                        verbose=False,
                        baseline=(0, 0),
                        preload=preload)

    # Load cleaned EEG data to sync the force data
    read_path = config['processed_data']['clean_eeg_path']
    eeg_epochs = dd.io.load(read_path, group='/' + subject + '/eeg/' + trial)
    drop_id = [id for id, val in enumerate(eeg_epochs.drop_log) if val]

    if len(eeg_epochs.drop_log) != len(epochs.drop_log):
        raise Exception('Two epochs are not of same length!')
    else:
        epochs.drop(drop_id)

    return epochs


def create_robot_epochs(config):
    """Create the data with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross

    Returns
    ----------
    eeg_epoch_dataset : dataset of all the subjects with different conditions

    """
    robot_epoch_dataset = {}
    for subject in config['subjects']:
        data = nested_dict()
        for trial in config['trials']:
            epochs = robot_epochs(subject, trial, config)
            data['robot'][trial] = epochs
        robot_epoch_dataset[subject] = data

    return robot_epoch_dataset

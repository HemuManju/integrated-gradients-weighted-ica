import mne
import deepdish as dd
from autoreject import AutoReject, get_rejection_threshold

from .utils import nested_dict


def autoreject_repair_epochs(epochs, reject_plot=False):
    """Rejects the bad epochs with AutoReject algorithm

    Parameter
    ----------
    epochs : Epoched, filtered eeg data
    Returns
    ----------
    epochs : Epoched data after rejection of bad epochs

    """
    # Cleaning with autoreject
    picks = mne.pick_types(epochs.info, eeg=True)  # Pick EEG channels
    ar = AutoReject(n_interpolate=[1, 4, 8],
                    n_jobs=6,
                    picks=picks,
                    thresh_func='bayesian_optimization',
                    cv=10,
                    random_state=42,
                    verbose=False)

    cleaned_epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    if reject_plot:
        reject_log.plot_epochs(epochs, scalings=dict(eeg=40e-6))

    return cleaned_epochs


def append_eog_index(epochs, ica):
    """Detects the eye blink aritifact indices and adds that information to ICA

    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    ica    : ica object from mne
    Returns
    ----------
    ICA : ICA object with eog indices appended

    """
    # Find bad EOG artifact (eye blinks) by correlating with Fp1
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp1',
                                             verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.65]
    ica.exclude += id_eog

    # Find bad EOG artifact (eye blinks) by correlation with Fp2
    eog_inds, scores_eog = ica.find_bads_eog(epochs,
                                             ch_name='Fp2',
                                             verbose=False)
    eog_inds.sort()
    # Append only when the correlation is high
    id_eog = [i for i, n in enumerate(scores_eog.tolist()) if abs(n) >= 0.75]
    ica.exclude += id_eog

    return ica


def clean_with_ica(epochs, show_ica=False):
    """Clean epochs with ICA.

    Parameter
    ----------
    epochs : Epoched, filtered, and autorejected eeg data
    Returns
    ----------
    ica     : ICA object from mne
    epochs  : ICA cleaned epochs

    """

    picks = mne.pick_types(epochs.info,
                           meg=False,
                           eeg=True,
                           eog=False,
                           stim=False,
                           exclude='bads')
    ica = mne.preprocessing.ICA(n_components=None,
                                method="picard",
                                verbose=False)
    # Get the rejection threshold using autoreject
    reject_threshold = get_rejection_threshold(epochs)
    ica.fit(epochs, picks=picks, reject=reject_threshold)

    ica = append_eog_index(epochs, ica)  # Append the eog index to ICA
    # mne pipeline to detect artifacts
    ica.detect_artifacts(epochs, eog_criterion=range(2))
    if show_ica:
        ica.plot_components(inst=epochs)
    ica.apply(epochs)  # Apply the ICA

    return epochs, ica


def clean_eeg_epochs(config):
    """Create cleaned dataset (by running autoreject and ICA)
    with each subject data in a dictionary.

    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross
    Returns
    ----------
    clean_eeg_dataset : dataset of all the subjects with different conditions

    """

    clean_eeg_dataset = {}
    read_path = config['interim_data']['eeg_epoch_path']
    raw_eeg = dd.io.load(str(read_path))  # load the epoch eeg

    for subject in config['subjects']:
        data = nested_dict()
        for trial in config['trials']:
            epochs = raw_eeg[subject]['eeg'][trial]
            ica_epochs, ica = clean_with_ica(epochs)
            repaired_eeg = autoreject_repair_epochs(ica_epochs)
            data['eeg'][trial] = repaired_eeg
            data['ica'][trial] = ica
        clean_eeg_dataset[subject] = data

    return clean_eeg_dataset

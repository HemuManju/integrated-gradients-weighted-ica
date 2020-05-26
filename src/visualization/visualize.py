import numpy as np
import mne

import matplotlib.pyplot as plt


def visualize_seperated_epochs(results, V):
    epochs = results['epochs']
    labels = results['labels']

    non_target = np.empty(epochs.shape)
    target = np.empty(epochs.shape)

    # Recovered signal
    for i, epoch in enumerate(epochs):
        if labels[i] > 0:
            target[i, :, :] = np.dot(V, epoch)
        else:
            non_target[i, :, :] = np.dot(V, epoch)

    # Convert to epoch
    ch_names = [
        'Fp1', 'Fp2', 'AFz', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
        'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7',
        'P3', 'Pz', 'P4', 'P8', 'PO7', 'O1', 'Oz', 'O2', 'PO8', 'PO9', 'PO10'
    ]
    ch_types = ['eeg'] * 32
    sfreq = 256  #epochs.shape[2]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    chname2idx = {}
    for i, chn in enumerate(ch_names):
        chname2idx[chn] = i

    # Non target and target epochs
    non_target_epochs = mne.EpochsArray(non_target, info=info)
    target_epochs = mne.EpochsArray(target, info=info)

    # Evoked plot
    plt.style.use('clean')
    fig, ax = plt.subplots(facecolor='white', figsize=(10.9, 7.6))
    evkTarget = target_epochs.average().data[chname2idx['Cz'], :]
    evkNonTarget = non_target_epochs.average().data[chname2idx['Cz'], :]

    t = np.arange(len(evkTarget)) / target_epochs.info['sfreq']
    ax.plot(t, evkTarget, label='Target')
    ax.plot(t, evkNonTarget, label='NonTarget')
    plt.show()

    # # Visualize
    # mne.viz.plot_epochs_image(target_epochs, picks='Cz', show=False)
    # mne.viz.plot_epochs_image(non_target_epochs, picks='Cz')
    return None

import numpy as np
import mne
import torch

from captum.attr import IntegratedGradients, FeatureAblation


def compute_ig_attributions(config,
                            trained_model,
                            data_iterator,
                            key='testing',
                            exp_type='classification'):
    """Computer attributions using integrated gradients method

    Parameters
    ----------
    trained_model : pytorch model
        A trained pytorch model
    data_iterator : dict
        A dictionary contaning data iterators where the key specifies which
        type of data to use i.e., training or validation or testing data
    key : str
        A string specifying which data (training, validation, or testing)
    Returns
    -------
    dict
        A dictionary containing attributes, targets, and input data.
    """
    # Random seeds
    torch.manual_seed(37)
    np.random.seed(37)

    # Device used for computation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Intergrated gradients algorithm
    ig = IntegratedGradients(trained_model)

    attributions, targets, inputs = [], [], []

    for x_batch, y_batch in data_iterator[key]:
        # Input and baseline
        batch_inputs = x_batch
        batch_baseline = (x_batch * 0)

        if exp_type == 'classification':

            batch_targets = (torch.max(y_batch, dim=1)[1])
            batch_attributions, delta = ig.attribute(
                inputs=batch_inputs.to(device),
                baselines=batch_baseline,
                target=batch_targets,
                return_convergence_delta=True)
        else:
            batch_targets = y_batch
            batch_attributions, delta = ig.attribute(
                inputs=batch_inputs.to(device),
                baselines=batch_baseline.to(device),
                return_convergence_delta=True)

        # Append them
        attributions.append(batch_attributions)
        targets.append(y_batch)
        inputs.append(x_batch)

    # Concatenate the attributes, targets, and inputs
    attributions = torch.cat(attributions, dim=0).cpu().detach().numpy()
    targets = torch.cat(targets, dim=0).cpu().detach().numpy()
    inputs = torch.cat(inputs, dim=0).cpu().detach().numpy()

    # Convert inputs to Mne Epochs
    info = mne.create_info(ch_names=[
        'Fp1', 'F7', 'F8', 'T4', 'T6', 'T5', 'T3', 'Fp2', 'O1', 'P3', 'Pz',
        'F3', 'Fz', 'F4', 'C4', 'P4', 'POz', 'C3', 'Cz', 'O2'
    ],
                           ch_types=['eeg'] * inputs.shape[1],
                           sfreq=250.0)
    epochs = mne.EpochsArray(inputs / (1e6), info=info)  # Convert to μV

    # Consolidate the results
    results = {}
    results['ig_attributions'] = attributions
    results['targets'] = targets
    results['inputs'] = inputs
    results['epochs'] = epochs

    return results


def compute_ablation_scores(config,
                            trained_model,
                            data_iterator,
                            key='testing',
                            exp_type='classification'):
    """Computer attributions using integrated gradients method

    Parameters
    ----------
    trained_model : pytorch model
        A trained pytorch model
    data_iterator : dict
        A dictionary contaning data iterators where the key specifies which
        type of data to use i.e., training or validation or testing data
    key : str
        A string specifying which data (training, validation, or testing)
    Returns
    -------
    dict
        A dictionary containing attributes, targets, and input data.
    """
    # Random seeds
    torch.manual_seed(37)
    np.random.seed(37)

    # Device used for computation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Intergrated gradients algorithm
    ablator = FeatureAblation(trained_model)

    attributions, targets, inputs = [], [], []
    temp = np.repeat(np.arange(20)[:, np.newaxis],
                     repeats=250 * config['epoch_length'],
                     axis=1)
    feature_mask = torch.from_numpy(temp).type(torch.int)

    for x_batch, y_batch in data_iterator[key]:
        # Input and baseline
        batch_inputs = x_batch
        attr_temp = []

        for ch_id, channel in enumerate(config['freq_bands']):
            input_ch = batch_inputs[:, -1, :, :]  # Unfiltered epoch

            if exp_type == 'classification':
                # Baseline is changed according to the input channel
                batch_targets = (torch.max(y_batch, dim=1)[1])
                baselines = batch_inputs[:, ch_id, :, :]

                # Compute attributions
                channel_attributes = ablator.attribute(
                    input_ch.to(device),
                    target=batch_targets.to(device),
                    baselines=baselines.to(device),
                    feature_mask=feature_mask.to(device))
            else:
                baselines = batch_inputs[:, ch_id, :, :]
                batch_targets = y_batch
                dummy = (y_batch * 0).to(device)
                channel_attributes = ablator.attribute(
                    inputs=(input_ch.to(device), dummy),
                    baselines=(baselines.to(device), dummy),
                    feature_mask=(feature_mask.to(device),
                                  dummy.type(torch.int)))

            if isinstance(channel_attributes, tuple):
                attr_temp.append(channel_attributes[0][:, None, :, :])
            else:
                attr_temp.append(channel_attributes[:, None, :, :])

        # Convert to array
        attr_temp = torch.cat(attr_temp, dim=1)

        # Append
        attributions.append(attr_temp)
        targets.append(batch_targets)
        inputs.append(batch_inputs)

    # Concatenate the attributes, targets, and inputs
    attributions = torch.cat(attributions, dim=0).cpu().detach().numpy()
    targets = torch.cat(targets, dim=0).cpu().detach().numpy()
    inputs = torch.cat(inputs, dim=0).cpu().detach().numpy()

    # Consolidate the results
    results = {}
    results['attributions'] = attributions
    results['targets'] = targets
    results['epochs'] = inputs
    return results

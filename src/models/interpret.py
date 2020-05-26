import numpy as np

import torch

from captum.attr import IntegratedGradients
# import matplotlib.pyplot as plt


def compute_attributions(trained_model, data_iterator):
    torch.manual_seed(37)
    np.random.seed(37)

    # Intergrated gradients algorithm
    trained_model.eval()
    ig = IntegratedGradients(trained_model)
    attributions, targets, inputs = [], [], []

    for x_batch, y_batch in data_iterator['testing']:
        # Input and baseline
        batch_inputs = x_batch
        batch_baseline = x_batch * 0
        batch_targets = (torch.max(y_batch, dim=1)[1])
        batch_attributions, delta = ig.attribute(batch_inputs,
                                                 batch_baseline,
                                                 target=batch_targets,
                                                 return_convergence_delta=True)
        attributions.append(batch_attributions)
        targets.append(batch_targets)
        inputs.append(batch_inputs)

    # Concatenate the attributes, targets, and inputs
    attributions = torch.cat(attributions, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    inputs = torch.cat(inputs, dim=0).numpy()

    # Consolidate the results
    results = {}
    results['attributions'] = attributions
    results['labels'] = targets
    results['epochs'] = inputs
    return results

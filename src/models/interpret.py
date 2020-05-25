import numpy as np

import torch

from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt


def compute_attribution(trained_model, data_iterator):
    torch.manual_seed(37)
    np.random.seed(37)

    # Intergrated gradients algorithm
    trained_model.eval()
    ig = IntegratedGradients(trained_model)

    for x_batch, y_batch in data_iterator['training']:
        # Input and baseline
        inputs = x_batch
        baseline = x_batch * 0
        targets = (torch.max(y_batch, dim=1)[1])
        attributions, delta = ig.attribute(inputs,
                                           baseline,
                                           target=targets,
                                           return_convergence_delta=True)
        for i, target in enumerate(targets):
            if target > 0:
                test = attributions[i].numpy()
                plt.imshow(test, aspect='auto')
                plt.show()
    return None

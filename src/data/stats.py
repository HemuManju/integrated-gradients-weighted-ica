import numpy as np
import pandas as pd

from models.utils import load_saved_checkpoint


def best_avg_metrics(config, checkpoint_dirpath, metric_type='classification'):

    columns = ['metrics_mean', 'metrics_std']
    df_accuracy = pd.DataFrame(columns=columns)
    accuracy = []
    for i in range(5):
        read_path = [checkpoint_dirpath, 'iter_' + str(i + 1)]
        saved_ckpt = load_saved_checkpoint(config, 0, read_path, best=True)
        accuracy.append(saved_ckpt['metric']['testing'])
    if metric_type == 'classification':
        accuracy = np.asarray(accuracy) * 100
    else:
        accuracy = np.asarray(accuracy)

    data = [
        np.mean(accuracy),
        np.std(accuracy),
    ]
    # Populate the data frame
    df_accuracy.loc[0] = data

    return df_accuracy

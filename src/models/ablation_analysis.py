import numpy as np
from data.utils import read_dataset


def selected_psd_features(config, exp_type, n_features=2):
    if exp_type == 'classification':
        read_path = config['processed_data']['albation_attr_classify_path']
    else:
        read_path = config['processed_data']['albation_attr_regress_path']

    # Read the data
    dataset = read_dataset(read_path)
    attributions = dataset['attributions']

    ch_names = [
        'Fp1', 'F7', 'F8', 'T4', 'T6', 'T5', 'T3', 'Fp2', 'O1', 'P3', 'Pz',
        'F3', 'Fz', 'F4', 'C4', 'P4', 'POz', 'C3', 'Cz', 'O2'
    ]
    mean_electrode = np.mean(attributions, axis=-1)
    avg = np.mean(mean_electrode, axis=0)

    # Pooled features
    pooled_mean_sorted = np.argsort(avg.flatten())
    pooled_features = []
    for ele in pooled_mean_sorted:
        ch_name = ele % 20
        freq = ele // 20
        feature = ch_names[ch_name] + '_' + config['band_names'][freq]
        pooled_features.append(feature)

    selected_features = []
    for i in range(len(config['freq_bands'])):
        if exp_type == 'classification':
            temp = np.argsort(avg[i])  # We want the prob to increase
        else:
            temp = np.argsort(-avg[i])  # We want error to decrease
        band_name = config['band_names'][i]
        features = [
            ch_names[temp[n]] + '_' + band_name for n in range(n_features)
        ]
        selected_features.append(features)
    selected_features = sum(selected_features, [])  # flatten out the list
    return selected_features, pooled_features

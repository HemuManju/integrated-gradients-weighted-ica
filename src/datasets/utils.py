import numpy as np
import pprint


def label_stats(data_iterator):
    info = {}
    for key in data_iterator.keys():
        info[key] = {}
        labels = data_iterator[key].dataset.labels
        _, counts = np.unique(labels[:, 0], return_counts=True)
        info[key]['class_distribution'] = counts / len(labels)
        info[key]['class_count'] = counts

        # Display the information
    pprint.pprint(info)
    return info

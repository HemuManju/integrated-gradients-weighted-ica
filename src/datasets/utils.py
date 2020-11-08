import numpy as np
from sklearn.model_selection import train_test_split


def get_train_valid_test_split(targets, config):
    ids = np.arange(targets.shape[0])
    if config['VALID_SIZE'] > 0:
        train_id, test_id, _, _ = train_test_split(
            ids, ids * 0, test_size=config['TEST_SIZE'] + config['VALID_SIZE'])
        valid_id, test_id, _, _ = train_test_split(
            test_id, test_id * 0, test_size=config['TEST_SIZE'])

    else:
        train_id, test_id, _, _ = train_test_split(
            ids, ids * 0, test_size=config['TEST_SIZE'])
        valid_id = None

    indices = {}
    indices['training'] = train_id
    indices['validation'] = valid_id
    indices['testing'] = test_id

    return indices

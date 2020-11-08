import collections
from pathlib import Path
import deepdish as dd

import pandas as pd


def nested_dict():
    return collections.defaultdict(nested_dict)


def save_dataset(path, dataset, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool

    """
    save_path = Path(__file__).parents[2] / path
    if save:
        dd.io.save(save_path, dataset)

    return None


def compress_dataset(path):
    """compress the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool
    """

    dataset = dd.io.load(path)
    # New name
    file_name = path.split('.')
    file_name[-2] = file_name[-2] + '_compressed.'
    save_path = ''.join(file_name)
    dd.io.save(save_path, dataset, compression=('blosc', 5))

    return None


def save_dataframe(path, dataframe, save):
    save_path = Path(__file__).parents[2] / path
    if save:
        dataframe.to_csv(save_path, index=False)
    return None


def read_dataframe(path):
    read_path = Path(__file__).parents[2] / path
    df = pd.read_csv(read_path)
    return df


def read_dataset(path):
    """Read the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool

    """
    read_path = Path(__file__).parents[2] / path
    data = dd.io.load(read_path)
    return data

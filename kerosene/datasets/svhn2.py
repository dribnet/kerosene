# -*- coding: utf-8 -*-
from fuel.datasets.svhn import SVHN
from .data_utils import get_datasets, fuel_datasets_into_lists

fileinfo = [
    "svhn_format_2.hdf5",
    "https://archive.org/download/kerosene_mnist/svhn_format_2.hdf5"
]

def load_data(sets=['train', 'test'], sources=['features', 'targets']):
    def load_data_callback():
        return map(lambda s: SVHN(which_format=2, which_sets=[s], sources=sources), sets)

    fuel_datasets = get_datasets(load_data_callback, *fileinfo)
    return fuel_datasets_into_lists(fuel_datasets)
    
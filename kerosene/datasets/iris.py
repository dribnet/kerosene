# -*- coding: utf-8 -*-
from fuel.datasets import Iris
from .data_utils import get_datasets, fuel_datasets_into_lists

fileinfo = [
    "iris.hdf5",
    "https://archive.org/download/kerosene_mnist/iris.hdf5"
]

def load_data(sets=['all'], sources=['features', 'targets']):
    def load_data_callback():
        return map(lambda s: Iris(which_sets=[s], sources=sources), sets)

    fuel_datasets = get_datasets(load_data_callback, *fileinfo)
    return fuel_datasets_into_lists(fuel_datasets, shuffle=True)

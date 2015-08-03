# -*- coding: utf-8 -*-
from fuel.datasets.cifar10 import CIFAR10
from .data_utils import get_datasets, fuel_datasets_into_lists

fileinfo = [
    "cifar10.hdf5",
    "https://archive.org/download/kerosene_mnist/cifar10.hdf5"
]

def load_data(sets=['train', 'test'], sources=['features', 'targets']):
    def load_data_callback():
        return map(lambda s: CIFAR10(which_sets=[s], sources=sources), sets)

    fuel_datasets = get_datasets(load_data_callback, *fileinfo)
    return fuel_datasets_into_lists(fuel_datasets)
    
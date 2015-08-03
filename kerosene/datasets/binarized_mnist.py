# -*- coding: utf-8 -*-
from fuel.datasets.binarized_mnist import BinarizedMNIST
from .data_utils import get_datasets, fuel_datasets_into_lists

fileinfo = [
    "binarized_mnist.hdf5",
    "https://archive.org/download/kerosene_mnist/binarized_mnist.hdf5"
]

def load_data(sets=['train', 'test'], sources=['features']):
    def load_data_callback():
        return map(lambda s: BinarizedMNIST(which_sets=[s], sources=sources), sets)

    fuel_datasets = get_datasets(load_data_callback, *fileinfo)
    return fuel_datasets_into_lists(fuel_datasets)

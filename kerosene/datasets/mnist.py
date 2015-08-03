# -*- coding: utf-8 -*-
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from .data_utils import get_datasets, fuel_data_to_list

def load_data(sets=['train', 'test'], sources=['features','targets']):
    def load_data_callback():
        return map(lambda s: MNIST(which_sets=[s], sources=sources), sets)

    fuel_datasets = get_datasets(load_data_callback, "mnist.hdf5", "https://archive.org/download/kerosene_mnist/mnist.hdf5")

    return map(fuel_data_to_list, fuel_datasets) # (X_train, y_train), (X_test, y_test)

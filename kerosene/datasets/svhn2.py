# -*- coding: utf-8 -*-
from fuel.datasets.svhn import SVHN
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from .data_utils import get_dataset

def load_train_data():
    return SVHN(which_format=2,which_sets=['train'], sources=['features', 'targets'])

def load_test_data():
    return SVHN(which_format=2,which_sets=['test'], sources=['features', 'targets'])

def load_data():
    train_data = get_dataset(load_train_data, "svhn_format_2.hdf5", "https://archive.org/download/kerosene_svhn2/svhn_format_2.hdf5")
    test_data = get_dataset(load_test_data, "svhn_format_2.hdf5", "https://archive.org/download/kerosene_svhn2/svhn_format_2.hdf5")

    train_data_stream = DataStream.default_stream(train_data,
        iteration_scheme=SequentialScheme(train_data.num_examples, train_data.num_examples))
    train_list = list(train_data_stream.get_epoch_iterator())

    test_data_stream = DataStream.default_stream(test_data,
        iteration_scheme=SequentialScheme(test_data.num_examples, test_data.num_examples))
    test_list = list(test_data_stream.get_epoch_iterator())

    return train_list[0], test_list[0] # (X_train, y_train), (X_test, y_test)

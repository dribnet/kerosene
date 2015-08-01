# -*- coding: utf-8 -*-
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

def load_data():
    train_data = MNIST(("train",))
    train_data_stream = DataStream.default_stream(train_data,
        iteration_scheme=SequentialScheme(train_data.num_examples, train_data.num_examples))
    train_list = list(train_data_stream.get_epoch_iterator())

    test_data = MNIST(("test",))
    test_data_stream = DataStream.default_stream(test_data,
        iteration_scheme=SequentialScheme(test_data.num_examples, test_data.num_examples))
    test_list = list(test_data_stream.get_epoch_iterator())

    return train_list[0], test_list[0] # (X_train, y_train), (X_test, y_test)

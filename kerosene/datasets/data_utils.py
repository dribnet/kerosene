# -*- coding: utf-8 -*-
import os

import fuel
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from keras.datasets import data_utils

def get_datasets(fetch_function, local_file, url):
    try:
        # first try just loading it from fuel's centralized location
        datasets = fetch_function()
    except IOError as e_not_found:
        # fallback to loading it from keras data subdirectory
        kerosenedir = os.path.expanduser(os.path.join('~', '.keras', 'datasets', "kerosene"))
        if not os.path.exists(kerosenedir):
            os.makedirs(kerosenedir)
        path = data_utils.get_file("kerosene/{}".format(local_file), origin=url)
        # override fuel's centralized location temporarily
        remember_path, fuel.config.data_path = fuel.config.data_path, kerosenedir
        datasets = fetch_function()
        # restore (in case we load other datasets from the central location)
        fuel.config.data_path = remember_path

    return datasets

def fuel_data_to_list(fuel_data):
    fuel_data_stream = DataStream.default_stream(fuel_data,
        iteration_scheme=SequentialScheme(fuel_data.num_examples, fuel_data.num_examples))
    return list(fuel_data_stream.get_epoch_iterator())[0]

def fuel_datasets_into_lists(fuel_datasets):
    return map(fuel_data_to_list, fuel_datasets)
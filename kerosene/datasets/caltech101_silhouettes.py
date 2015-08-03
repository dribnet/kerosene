# -*- coding: utf-8 -*-
from fuel.datasets.caltech101_silhouettes import CalTech101Silhouettes
from .data_utils import get_datasets, fuel_datasets_into_lists

fileinfo_16 = [
    "caltech101_silhouettes16.hdf5",
    "https://archive.org/download/kerosene_mnist/caltech101_silhouettes16.hdf5"
]

fileinfo_28 = [
    "caltech101_silhouettes28.hdf5",
    "https://archive.org/download/kerosene_mnist/caltech101_silhouettes28.hdf5"
]

def load_data(size=28, sets=['train', 'test'], sources=['features', 'targets']):
    def load_data_callback():
        return map(lambda s: CalTech101Silhouettes(which_sets=[s], sources=sources), sets)

    fileinfo = (fileinfo_16 if size==16 else fileinfo_28)
    fuel_datasets = get_datasets(load_data_callback, *fileinfo)
    return fuel_datasets_into_lists(fuel_datasets)

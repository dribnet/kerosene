# -*- coding: utf-8 -*-
import os
import fuel
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.downloaders.base import default_downloader

########## UTILS 

# cache this incase we load multiple datasets
initial_fuel_config_path = fuel.config.data_path

# helper class to construct local and remote filenames from metadata
def paths_from_metadata(basename, version, url_dir):
    filename = "{}-{}.hdf5".format(basename, version)
    url_dir = "{}{}".format(url_dir, filename)
    return (filename, url_dir)

# this assumes that the load should happen from kerosene path and
# will include a download of the file if not already present
# returns fuel path to local file
def ensure_dataset_ready(basename, version, url_dir):
    # setup names
    filename, url = paths_from_metadata(basename, version, url_dir)
    kerosenedir = os.path.expanduser(os.path.join('~', '.kerosene', 'datasets'))
    filetarget = os.path.join(kerosenedir, filename)
    # if file is not present, download it (also created directories if needed)
    if not os.path.isfile(filetarget):
        default_downloader(kerosenedir, [url], [filename])
    # override fuel's centralized location temporarily
    fuel.config.data_path = kerosenedir
    return filename

def restore_fuel_data_path():
    fuel.config.data_path = initial_fuel_config_path

def fuel_data_to_list(fuel_data, shuffle):
    if(shuffle):
        scheme = ShuffledScheme(fuel_data.num_examples, fuel_data.num_examples)
    else:
        scheme = SequentialScheme(fuel_data.num_examples, fuel_data.num_examples)
    fuel_data_stream = DataStream.default_stream(fuel_data, iteration_scheme=scheme)
    return next(fuel_data_stream.get_epoch_iterator())

def fuel_datasets_unpacked(fuel_datasets, shuffle=False):
    return map(lambda x: fuel_data_to_list(x, shuffle=shuffle), fuel_datasets)

########### Base class

class Dataset(object):
    # these values are intended to be immutable per subclass
    basename = "your_dataset_name_here"
    version = "0.1.0"
    url_dir = "https://archive.org/download/kerosene_201508/"
    default_sets = ['train', 'test']
    default_sources = ['features', 'targets']

    # set this if a fuel class needs a "filename" override patch
    class_for_filename_patch = None

    # subclasses override this to build the requested datasets from hdf5 file
    # (self is generally not needed, can be a pure function)
    def build_data(self, sets, sources):
        return None

    def apply_transforms(self, datasets):
        return datasets

    def load_data(self, sets=None, sources=None, fuel_dir=False):
        sets = self.default_sets if sets is None else sets
        sources = self.default_sources if sources is None else sources
        if(fuel_dir):
            restore_fuel_data_path()
        else:
            local_file = ensure_dataset_ready(self.basename, self.version, self.url_dir)
            if(self.class_for_filename_patch):
                self.class_for_filename_patch.filename = property(lambda self: local_file)
        datasets = self.build_data(sets, sources)
        datasets = fuel_datasets_unpacked(datasets, shuffle=True)
        datasets = self.apply_transforms(datasets)
        return datasets

# again, this is an template for subclasses
def load_data(sets=None, sources=None, fuel_dir=False):
    return Dataset().load_data(sets, sources, fuel_dir);

# Kerosene: Clean Burning Fuel

Kerosene is a growing set of publicly available verisioned datasets in hdf5 format with a dead-simple interface.

## Show me

The default interface is a drop-in clone of keras.datasets: a minimal interface to provide features and labels in a test train split.

```python
from kerosene.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

Datasets can have one or more sources, such as secondary features.

```python
from kerosene.datasets import cifar100
# default labels for cifar100 are 'coarse_labels'
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
# but you can also subsequently grab the 'fine_labels'
(z_train,), (z_test,) = cifar100.load_data(sources=['fine_labels'])
```

Many datasets have multiple sets (aka "splits") other than test/train.

```python
from kerosene.datasets import svhn2
import numpy as np
# street view house numbers defaults: 73,257 train / 26,032 test
(X_train, y_train), (X_test, y_test) = svhn2.load_data()
# but maybe you have time to burn? use 'extra' and train on > 600,000!
(X_train, y_train), (X_extra, y_extra), (X_test, y_test) = svhn2.load_data(sets=['train', 'extra', 'test'])
X_train = np.concatenate([X_train, X_extra])
y_train = np.concatenate([y_train, y_extra])
```

And for some datasets less is more - perhaps only one source...

```python
from kerosene.datasets import binarized_mnist
# what no labels? send in the autoencoder
(X_train,), (X_test,) = binarized_mnist.load_data()
```

or one set.

```python
from kerosene.datasets import iris
(X_all, y_all), = iris.load_data()
...
# keras to the rescue
model.fit(X_all, Y_all, validation_split=0.25)
```

Just like keras.datasets, downloads are automatic and cached on your local drive.

## OK, what is this again?

Kerosene provides a collection of versioned, immutable, publicly available fuel-compatible datasets in hdf5 format along with a minimalist interface for Keras. Let's go through that quickly.

  * Semantic Versioning. Just like software. There will be bugs. There will be changes. We'll be ready.
  * Immutable. Once a version released to the wild, it is never rewritten.
  * Publicly available. Reproducable experiments depend on unencombered access.
  * fuel-compatible - borrows heavily and remains compatible with the [fuel data pipeline frameworkk](https://github.com/mila-udem/fuel)
  * hdf5 format. welcome to a saner world free of pickled python objects.
  * interface: As simple as possible. Automatic downloads and sensible defaults.

Kerosene itself includes wrappers only for the datasets that are built into the fuel libraries. But when
used as a dependency, it allows similar access to any third party fuel hdf5 file. As an example, see
the lfw_fuel repo which provides keras and blocks access to the Labeled Faces in the Wild dataset
in several formats.

## Installation

Currently depends on official Keras release and current version of fuel. Not yet installable via pip.

You probably know how to install Keras, so to get fuel just

```bash
pip install git+git://github.com/mila-udem/fuel.git@0653e5b

```

And then from this repo

```
python setup.py install
```

After that you should be able to run any of the examples in the examples folder

```bash
python ./examples/mnist.py
```

## What's included

Currently the six datasets are wrappers around those provided by fuel. Each has corresponding
example in the examples directory which is meant to be a high performance representative use of that
dataset.

There's also small wrapper scripts `kero-download` and `kero-convert`, which are used to run `fuel-download`
and `fuel-convert` on datasets that are not part of the fuel distribution - such as lfw_fuel.

## Issues

This project is just getting started, so the API is subject to change, documentation is lacking, and options are not necessarily discoverable. I'm not so pleased with the hdf5 file sizes. The dev fuel dependency isn't great,
but this cannot be fixed until a fuel release. The overall software architecture is also rough, but it functions
fine a s proof of concept that can be refined if useful.

## Feedback:

Kerosene is currently an experiment in making datasets large and small easily sharable. Feedback welcome via github issues or email.

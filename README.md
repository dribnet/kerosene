### Note: This library depends on Fuel - which is no longer being developed or maintained. So this library is also orphaned and is no longer under active development. 

# Kerosene: Clean Burning Fuel

Provides verisioned datasets to Machine Learning projects in hdf5 format with a dead-simple interface.

## Show me

Without optional arguments, kerosene provides a minimal interface to get features and labels in a test / train split. Below are examples of using it with [keras](https://github.com/fchollet/keras), but it can work with any machine learning library.

```python
# MNIST example
from keras.models import Sequential
from kerosene.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# (build the perfect model here)

model.fit(X_train, Y_train, show_accuracy=True, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
```

Kerosene datasets support one or more sources, such as secondary labels.

```python
# CIFAR100 example
from kerosene.datasets import cifar100
# default labels for cifar100 are 'coarse_labels'
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
# but you can also subsequently grab the 'fine_labels'
(z_train,), (z_test,) = cifar100.load_data(sources=['fine_labels'])
```

And additionally can have multiple sets (aka "splits") other than test/train.

```python
# Street View House Numbers example
from kerosene.datasets import svhn2
import numpy as np
# street view house numbers defaults: 73,257 train / 26,032 test
(X_train, y_train), (X_test, y_test) = svhn2.load_data()
# have time to burn? use 'extra' and train on > 600,000 examples!
(X_extra, y_extra), = svhn2.load_data(sets=['extra'])
X_train = np.concatenate([X_train, X_extra])
y_train = np.concatenate([y_train, y_extra])
```

And for some datasets less is more - perhaps only one source

```python
# Binarized MNIST example
from kerosene.datasets import binarized_mnist
# what no labels? send in the autoencoder
(X_train,), (X_test,) = binarized_mnist.load_data()
```

or one set.

```python
# Iris example
from kerosene.datasets import iris
(X_all, y_all), = iris.load_data()
# then later ... keras to the rescue
model.fit(X_all, Y_all, validation_split=0.25)
```

Kerosene downloads are automatic and cached on your local drive.

## OK, what is this again?

Kerosene provides a collection of versioned, immutable, publicly available fuel-compatible datasets in hdf5 format along with a minimalist interface for Keras. So --

  * semantic versioning: Just like software - there will be bugs and changes. 
  * immutable: Once a version released to the wild, it is never rewritten.
  * publicly available: Reproducible experiments depend on unencumbered data access.
  * fuel-compatible - Borrows from and stays compatible with the [fuel data pipeline framework](https://github.com/mila-udem/fuel)
  * hdf5 format: Hoping a pipeline free of pickled python objects will be a saner one.
  * interface: As simple as possible. Automatic downloads and sensible defaults.

Kerosene includes wrappers for most of the datasets that are built into the fuel libraries.
When used as a dependency, it similarly provides access to any third party fuel hdf5 file in a way
intended to be useful to any ML library, such as [keras](https://github.com/fchollet/keras) and [blocks](https://github.com/mila-udem/blocks). As an example, see
the [lfw_fuel](https://github.com/dribnet/lfw_fuel) repo which provides simple
access to the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset in several formats.

## Installation

```bash
pip install kerosene
```

Kerosene depends on the [fuel](https://github.com/mila-udem/fuel) library, which will be installed automatically if needed.

Sometimes sudo is necessary for the pip command.

If you have keras, you should be able to run any of the examples in the examples folder with the most recent version of keras (0.3.0).

```bash
pip install keras
python ./examples/mnist.py
```

## What's included

Currently the six datasets are wrappers around those provided by fuel. Each has a corresponding
keras based example in the examples directory which is intended to be a high performance representative use of that dataset.

| Dataset | # records | % Accuracy Score |
|---------|-------------------|-----------------|
| binarized_mnist | 70,000    |     0.2358 (loss)   |
| [cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)         | 60,000    |     74.98       |
| [cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)        | 60,000    |  53.14 (coarse) / 44.02 (fine) |
| [iris](https://en.wikipedia.org/wiki/Iris_flower_data_set)            |    150    |     63.16       |
| [mnist](http://yann.lecun.com/exdb/mnist/)           | 70,000    |     99.10       |
| [svhn2](http://ufldl.stanford.edu/housenumbers/)       | >600,000  |  93.05 (train) / 96.40 (+extra) |

Merge requests for any of these examples that are more accurate, run faster, and/or are written clearer are definitely welcome.

It is also possible to use `fuel-download` and `fuel-convert` on datasets that are not part of the fuel distribution, making them kerosene and fuel compatible. An example is [lfw_fuel](https://github.com/dribnet/lfw_fuel), which creates a fuel-compatible dataset.


## Issues

The next planned improvements are:

 * support for compressed downloads (because the hdf5 files are large)
 * interface for iterating over a dataset without loading it into memory

Documentation is lacking, options are not easily discoverable, and the software design is rough. These areas can be improved if others find this library useful.

## License

MIT

## Feedback:

Kerosene is currently an experiment in making datasets large and small easily sharable. Feedback welcome via github issues or email.

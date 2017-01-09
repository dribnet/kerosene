from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from kerosene.datasets import binarized_mnist
from keras.models import Sequential
from keras.layers.core import Dense, AutoEncoder
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import containers
import keras.backend as K

K.set_image_dim_ordering('th')

'''
    Iain Murray's conversion of mnist to binary format. The
    provenance is from Hugo Larochelle's website.

    Ruslan Salakhutdinov and Iain Murray, *On the Quantitative
        Analysis of Deep Belief Networks*, Proceedings of the 25th
        international conference on Machine learning, 2008, pp. 872-879.

    http://www.cs.toronto.edu/~larocheh/public/datasets/
        binarized_mnist/binarized_mnist_{train,valid,test}.amat

    Note that this dataset is not quite what you might expect:

        * randomized dithering limits compression roughly 50%
        * no labels
        * validation could have "writer" information leakage

    But at this point this dataset is in standard use so I'm
    making it available for comparison to published results.

    Since I don't have labels, I'm just running it through the
    autoencoder with reconstruction; Seems to converge, but haven't
    verified it's doing the right thing.
    
    This version gets stuck at 0.2358 on the first epoch and remains
    until epoch 12. 1 second per epoch on a GeForce GTX 680 GPU.
    (This is probably a bug that needs to be fixed).
'''

batch_size = 128
nb_epoch = 12

# the data, shuffled and split between tran and test sets
(X_train,), (X_test,) = binarized_mnist.load_data()

# print shape of data while model is building
train_shape = X_train.shape
test_shape = X_test.shape
print("{1} train samples, {2} channel{0}, {3}x{4}".format("" if train_shape[1] == 1 else "s", *train_shape))
print("{1}  test samples, {2} channel{0}, {3}x{4}".format("" if test_shape[1] == 1 else "s", *test_shape))

# flatten
num_train_samples = train_shape[0]
num_test_samples = test_shape[0]
input_dim = train_shape[1] * train_shape[2] * train_shape[3]
X_train = X_train.reshape(num_train_samples, input_dim)
X_test = X_test.reshape(num_test_samples, input_dim)
# parameters for autoencoder
hidden_dim = int(input_dim / 2)
final_dim = int(hidden_dim / 2)
activation = 'linear'

# build model
model = Sequential()
encoder = containers.Sequential([
    Dense(hidden_dim, activation=activation, input_shape=(input_dim,)),
    Dense(final_dim,  activation=activation)])
decoder = containers.Sequential([
    Dense(hidden_dim, activation=activation, input_shape=(final_dim,)),
    Dense(input_dim, activation=activation)])
model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, X_test))

# not sure if this is a valid way to evaluate the autoencoder...
score = model.evaluate(X_test, X_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

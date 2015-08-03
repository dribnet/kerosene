from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from kerosene.datasets import caltech101_silhouettes
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

'''
    CalTech101 Silhouette dataset prepared by Benjamin M. Marlin.

        https://people.cs.umass.edu/~marlin/data.shtml

    Here we test the 28x28 pixel sized version. It's rather small - so
    we'll just combine the 4100 train and 2264 validation splits,
    which are run against the 2307 images in the test split.

    This version can get to 72.95% test accuracy after 12 epochs.
    1 second per epoch on a GeForce GTX 680 GPU.
'''

batch_size = 128
nb_classes = 102 # they actually go from [1, 101] inclusive
nb_epoch = 12

# the data, shuffled and split between tran and test sets
(X_train, y_train),  (X_valid, y_valid), (X_test, y_test) = \
	caltech101_silhouettes.load_data(sets=['train', 'valid', 'test'])
X_train = np.concatenate([X_train, X_valid])
y_train = np.concatenate([y_train, y_valid])

# print shape of data while model is building
print("{1} train samples, {2} channel{0}, {3}x{4}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
print("{1}  test samples, {2} channel{0}, {3}x{4}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32*196, 128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

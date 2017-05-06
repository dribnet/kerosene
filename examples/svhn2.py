from __future__ import absolute_import
from __future__ import print_function
import os
import itertools
import numpy as np
np.random.seed(1337)  # for reproducibility

from kerosene.datasets import svhn2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
import keras.backend as K

K.set_image_dim_ordering('th')

'''
    Train a convnet on the cropped Stanford Street View House Numbers dataset.

    http://ufldl.stanford.edu/housenumbers/

    This version can get to 93.05% test accuracy after 12 epochs.
    35 seconds per epoch on a GeForce GTX 680 GPU.

    Or you can try:
      USE_EXTRA=1 python svhn2.py

    to also train on the much larger set that includes "extra" data.

    With this extra data, this gets to 96.40% test accuracy after 12 epochs.
    273 seconds per epoch on a GeForce GTX 680 GPU.
'''

batch_size = 128
nb_classes = 11
nb_epoch = 12

if "USE_EXTRA" not in os.environ:
    # standard split is 73,257 train / 26,032 test
    (X_train, y_train), (X_test, y_test) = svhn2.load_data()
else:
    # svhn2 extra split has an additional 531,131 (!) examples
    (X_train, y_train), (X_extra, y_extra), (X_test, y_test) = svhn2.load_data(sets=['train', 'extra', 'test'])
    X_train = np.concatenate([X_train, X_extra])
    y_train = np.concatenate([y_train, y_extra])

# print shape of data while model is building
print("{1} train samples, {2} channel{0}, {3}x{4}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
print("{1}  test samples, {2} channel{0}, {3}x{4}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))

# input image dimensions
_, img_channels, img_rows, img_cols = X_train.shape

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from kerosene.datasets import iris
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

'''
    Train something simple on the classic iris dataset.

    This version can get to 63.16% test accuracy after 3 epochs,
    and it remains there for future epochs, though the test loss
    continues to fall for all 12 epochs.
    0 seconds per epoch, even on your CPU.
'''

batch_size = 1
nb_classes = 3
nb_epoch = 12

# the data, shuffled and split between tran and test sets
(X_all, y_all), = iris.load_data()

# print shape of data while model is building
print("{0} samples in all, {1} columns".format(*X_all.shape))

# convert class vectors to binary class matrices
Y_all = np_utils.to_categorical(y_all, nb_classes)

model = Sequential()                                                       
model.add(Dense(4, 3, init='uniform'))                                   
model.add(Activation('softmax'))                                           
model.compile(loss='mean_squared_error', optimizer='rmsprop')                    

model.fit(X_all, Y_all, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, validation_split=0.25, verbose=1)

from keras.datasets import mnist  # subroutines for fetching the MNIST dataset
from keras.models import Model  # basic class for specifying and training a neural network
from keras.layers import Input, Dense  # the two types of neural network layer we will be using
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values
from keras.models import model_from_json
from config import config
import os
import json
from prepare_data import get_file_content
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation



def create_flat_vector(x_train, length_samples):
    new_x_train = np.zeros((length_samples, 448))
    i = 0
    for x_class in x_train:
        for x_class_sample in x_class:
            x_vector_sample = np.array([])

            for x_frame in x_class_sample:
                if len(x_vector_sample) == 448:
                    break
                x_vector_sample = np.append(x_vector_sample, np.asarray(x_frame))

            for j in range(448):
                new_x_train[i, j] = x_vector_sample[j]

            i += 1
    return new_x_train


x_path = os.path.join(config.DATA_PATH, "X_TRAIN.json")
y_path = os.path.join(config.DATA_PATH, "Y_TRAIN.json")

x_path_test = os.path.join(config.DATA_PATH, "X_TEST.json")
y_path_test = os.path.join(config.DATA_PATH, "Y_TEST.json")

X_train = json.loads(get_file_content(x_path))
Y_train = json.loads(get_file_content(y_path))

X_test = json.loads(get_file_content(x_path_test))
Y_test = json.loads(get_file_content(y_path_test))

# # delete trash
# del X_train[-1]
# del Y_train[-85:]
#
# # delete trash
# del X_test[-1]
# del Y_test[-85:]


# need array 850, 448 X
X_train = create_flat_vector(X_train,1650)
X_test = create_flat_vector(X_test,500)
# need array 850, 1   Y

X_train = np.array(X_train)
Y_train = np.array(Y_train)

#Y_train = Y_train[:, :10]


X_test = np.array(X_test)
Y_test = np.array(Y_test)

batch_size = 64  # in each iteration, we consider 128 training examples at once
num_epochs = 60  # we iterate twenty times over the entire training set
hidden_size = 512  # there will be 512 neurons in both hidden layers

num_train = 1650 # there are 60000 training examples in MNIST
num_test = 500 # there are 10000 test examples in MNIST
#
num_classes = 10 # there


model = Sequential()
input_size = num_train
model.add(Dense(32, input_dim=input_size))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy',
optimizer='rmsprop',
metrics=['accuracy'])

model.fit(X_train, Y_train, # Train the model using the training set...
batch_size=batch_size, nb_epoch=num_epochs,
verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

print "evaluate"
score = model.evaluate(X_test, Y_test,verbose=1) # Evaluate the trained model on the test set!
print "\nResult\n"
print score


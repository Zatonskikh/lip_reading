'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
from config import config
import os
import json
from prepare_data import get_file_content


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 448
epochs = 25
# number of elements ahead that are used to make the prediction
lahead = 1

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


# need array 850, 448 X
X_train = create_flat_vector(X_train,850)
X_test = create_flat_vector(X_test,300)
# need array 850, 1   Y

X_train = np.array(X_train)
Y_train = np.array(Y_train)

Y_train = Y_train[:, :10]


X_test = np.array(X_test)
Y_test = np.array(Y_test)

Y_test = Y_test[:, :10]



print('Creating Model...')
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))

print('Plotting Results')
plt.subplot(2, 1, 1)
plt.plot(Y_test)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()


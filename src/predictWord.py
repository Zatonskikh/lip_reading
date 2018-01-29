from __future__ import print_function
from __future__ import print_function
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout,Embedding,Conv1D,MaxPooling1D
import numpy as np
from config import config
import os
import json
from prepare_data import get_file_content

def create_flat_vector(x_train, length_samples):
    new_x_train = []
    for x_class in x_train:
        new_x_train += x_class

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
X_train = create_flat_vector(X_train,1650)
X_test = create_flat_vector(X_test,500)
# need array 850, 1   Y

X_train = np.array(X_train)
Y_train = np.array(Y_train)

#Y_train = Y_train[:, :10]


#X_test = np.array(X_test)
#Y_test = np.array(Y_test)

#Y_test = Y_test[:, :10]


data_dim = 56
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(16, return_sequences=True,dropout_W=0.1,
               input_shape=(timesteps, data_dim)))
# returns a sequence of vectors of dimension 32
model.add(LSTM(32,dropout_U=0.1, return_sequences=True))# returns a sequence of vectors of dimension 32
model.add(LSTM(64,dropout_U=0.1,dropout_W=0.1)) # return a single vector of dimension 32

model.add(Dense(512,activation="relu"))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.load_weights('data/savedmodel_')
model.predict_classes()

if __name__ == '__main__':


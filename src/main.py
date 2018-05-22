import os
import src.config
from src.markup_data import MarkUp
from uuid import uuid4
from src.utils.utils import start_collect_data
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout,Embedding,Conv1D,MaxPooling1D
from src.facerec_from_video_file import face_recognition_custom

def create_flat_vector(x_train, length_samples):
    new_x_train = []
    for x_class in x_train:
        new_x_train += x_class

    return new_x_train

def model(x):
    data_dim = 56
    timesteps = 8
    num_classes = 10

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(16, return_sequences=True, dropout_W=0.1,
                   input_shape=(timesteps, data_dim)))
    # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, dropout_U=0.1, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64, dropout_U=0.1, dropout_W=0.1))  # return a single vector of dimension 32

    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.load_weights('../data/savedmodel_new')
    a = model.predict_classes(x)
    return config.WORDS[int(a)]

if __name__ == '__main__':

    instance = MarkUp()
    if config.COLLECT_DATA:
        start_collect_data(instance)
    else:
        random_str = str(uuid4())
        path = "2446.avi"
        dist = instance.mark_up_video(os.path.join(config.DATA_PATH, path), random_str)
        x = []
        for d in dist:
            x.append(d['vector'])
        x = [[x]]
        x_val = np.array(create_flat_vector(x, 1))
        word = model(x_val)
        # if word == "shifrovanie":
        #     print("Error paswword")
        try:
            face_recognition_custom(path, word)
        except:
            print("NOT VALID PERSON")
    print ("Not found %d" % instance.error_count)

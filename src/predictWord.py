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


# x_path = os.path.join(config.DATA_PATH, "X_TRAIN.json")
# y_path = os.path.join(config.DATA_PATH, "Y_TRAIN.json")
#
# x_path_test = os.path.join(config.DATA_PATH, "X_TEST.json")
# y_path_test = os.path.join(config.DATA_PATH, "Y_TEST.json")
#
# X_train = json.loads(get_file_content(x_path))
# Y_train = json.loads(get_file_content(y_path))
#
# X_test = json.loads(get_file_content(x_path_test))
# Y_test = json.loads(get_file_content(y_path_test))
#
#
# # need array 850, 448 X
# X_train = create_flat_vector(X_train,1650)
# X_test = create_flat_vector(X_test,500)
# # need array 850, 1   Y
#
# X_train = np.array(X_train)
# Y_train = np.array(Y_train)

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

model.load_weights('../data/savedmodel_new')
x_list = [[[0.15426121636932932, 0.4453753922096171, 1.1376472043977222, 0.5053054419187875, 0.17157122360003363, 0.27444208149161153, 0.3140145365160312, 0.15426121636932932, 0.07024361211052849, 0.40276011118272415, 0.011702193433637875, -0.08151997718340276, 0.0717298554444476, -0.05230546768078809, 0.4453753922096171, 0.07024361211052849, 0.12815222701116058, -0.08147991044159308, -0.02292436277108867, 0.06343741712941597, -0.053030327035591185, 1.1376472043977222, 0.40276011118272415, 0.12815222701116058, 0.09949571021042658, 0.37747763861379724, 0.21743221889133268, 0.2741764990877056, 0.5053054419187875, 0.011702193433637875, -0.08147991044159308, 0.09949571021042658, 0.08200948725075141, -0.08553808890998577, 0.04442416777523789, 0.17157122360003363, -0.08151997718340276, -0.02292436277108867, 0.37747763861379724, 0.08200948725075141, -0.09643687750385516, 0.02792296674629091, 0.27444208149161153, 0.0717298554444476, 0.06343741712941597, 0.21743221889133268, -0.08553808890998577, -0.09643687750385516, -0.03649780790684067, 0.3140145365160312, -0.05230546768078809, -0.053030327035591185, 0.2741764990877056, 0.04442416777523789, 0.02792296674629091, -0.03649780790684067], [-0.0351303320202172, -0.19652638121293453, -0.46133750776948634, -0.27064164413558434, -0.05851543924160496, -0.125132307885796, -0.12781943342936675, -0.0351303320202172, -0.045698688252367725, -0.1515854541316961, -0.030089762500727035, 0.03171771457210393, 0.009297262077545498, 0.007511717743025559, -0.19652638121293453, -0.045698688252367725, -0.015610610699430039, 0.02908033908174512, -0.02665153294174538, 0.0026756151706184284, 0.002494214735091538, -0.46133750776948634, -0.1515854541316961, -0.015610610699430039, -0.015960710776049325, -0.19863351068454227, -0.10458481255474883, -0.10006542359593507, -0.27064164413558434, -0.030089762500727035, 0.02908033908174512, -0.015960710776049325, -0.0736031569731781, -0.026468969544127197, -0.021120815941030824, -0.05851543924160496, 0.03171771457210393, -0.02665153294174538, -0.19863351068454227, -0.0736031569731781, -0.01727062151539449, -0.01399499304000651, -0.125132307885796, 0.009297262077545498, 0.0026756151706184284, -0.10458481255474883, -0.026468969544127197, -0.01727062151539449, -0.0008754837250058156, -0.12781943342936675, 0.007511717743025559, 0.002494214735091538, -0.10006542359593507, -0.021120815941030824, -0.01399499304000651, -0.0008754837250058156], [-0.17937817728360383, -0.3144406607933057, -0.6539202168987348, -0.2450406483574885, -0.07267559864113204, -0.14995453154537786, -0.1896699167949798, -0.17937817728360383, -0.03168271114461896, -0.28788271203820415, -0.03800135323818865, 0.020184111848345676, -0.0924095063971753, 0.040512093194981946, -0.3144406607933057, -0.03168271114461896, -0.13334872365053685, 0.022210327982917555, 0.013332928473578809, -0.08192169537695793, 0.05119370056961681, -0.6539202168987348, -0.28788271203820415, -0.13334872365053685, -0.052536727263940985, -0.1885089024545803, -0.1265303024994764, -0.16980816876156068, -0.2450406483574885, -0.03800135323818865, 0.022210327982917555, -0.052536727263940985, -0.0355408920284605, 0.0966802832872602, -0.04420848936372318, -0.07267559864113204, 0.020184111848345676, 0.013332928473578809, -0.1885089024545803, -0.0355408920284605, 0.1143650872883668, -0.029736636783207926, -0.14995453154537786, -0.0924095063971753, -0.08192169537695793, -0.1265303024994764, 0.0966802832872602, 0.1143650872883668, 0.04331410925548883, -0.1896699167949798, 0.040512093194981946, 0.05119370056961681, -0.16980816876156068, -0.04420848936372318, -0.029736636783207926, 0.04331410925548883], [0.1182676669512075, 0.2766628406162486, 0.7319806460240406, 0.34600256036604415, 0.10806942764756755, 0.15854358021353, 0.2543142199984756, 0.1182676669512075, 0.03223487006310344, 0.24867398587334755, 0.04708717768322468, -0.05718079132458276, 0.04299201674435996, -0.037987007052556976, 0.2766628406162486, 0.03223487006310344, 0.09475618078295822, -0.01591446413970843, -0.033885685676702915, 0.04559754329212229, -0.06717782173144038, 0.7319806460240406, 0.24867398587334755, 0.09475618078295822, 0.06599183411481946, 0.2304099901882839, 0.15826346355969667, 0.12566307716620706, 0.34600256036604415, 0.04708717768322468, -0.01591446413970843, 0.06599183411481946, 0.05543483558630519, -0.03344689679197932, 0.043288498358825644, 0.10806942764756755, -0.05718079132458276, -0.033885685676702915, 0.2304099901882839, 0.05543483558630519, -0.08178101377218311, 0.037687480936719975, 0.15854358021353, 0.04299201674435996, 0.04559754329212229, 0.15826346355969667, -0.03344689679197932, -0.08178101377218311, -0.03239659193310924, 0.2543142199984756, -0.037987007052556976, -0.06717782173144038, 0.12566307716620706, 0.043288498358825644, 0.037687480936719975, -0.03239659193310924], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
x_list = np.array(x_list)
a = model.predict_classes(X_train)
b = list(a)
print (b)




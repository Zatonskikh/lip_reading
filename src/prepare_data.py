import config.config as config
import os
import json
import glob
import re


def get_file_content(file_name):
    current_file = open(file_name, "r")
    content = current_file.read()
    current_file.close()
    return content


def get_vector_from_json(path):
    json_str = get_file_content(path)
    json_content = json.loads(json_str)
    assert 'vector' in json_content
    return json_content['vector']


def sort_frames(text):
    return int(re.split('(\d+)', os.path.basename(text))[1])


def get_x_sample(word_folder_path, word, y, y_sample):
    folders_x = glob.glob(os.path.join(word_folder_path, "%s_*/" % word))
    x_samples = []
    for folder in folders_x:
        frames = glob.glob(os.path.join(folder, "*.json"))
        frames = sorted(frames, key=sort_frames)
        print(frames)
        x_sample = []
        for frame in frames:
            x_sample.append(get_vector_from_json(frame))
        y.append(y_sample)
        x_samples.append(x_sample)
    return x_samples, y


def get_y_sample(word_folder_path, word):
    meta_y_path = os.path.join(word_folder_path, word + "_meta.json")
    if os.path.exists(meta_y_path):
        return get_vector_from_json(meta_y_path)


def get_x_y_vectors():
    x = []
    y = []
    for word in config.NN_CLASSES_ID.keys():
        word_folder_path = os.path.join(config.DATA_PATH, word)
        if os.path.exists(word_folder_path):
            y_sample = get_y_sample(word_folder_path, word)
            x_sample, y = get_x_sample(word_folder_path, word, y, y_sample)
            x.append(x_sample)
    return x, y


def save_data(vec, name):
    path = os.path.join(config.DATA_PATH, name + ".json")
    file = open(path, "w")
    json.dump(vec, file)
    file.close()


def start_processing_date():
    print("DATA PROCESSING STARTED")
    x_train, y_train = get_x_y_vectors()
    save_data(x_train, "X_TRAIN")
    save_data(y_train, "Y_TRAIN")
    print("DATA PROCESSING FINISHED")

# start_processing_date()
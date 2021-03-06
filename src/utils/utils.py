import cv2
import json
import os
import glob
import numpy as np
import src.config as config
import re
from uuid import uuid4

def show_image(name, image):
    cv2.imshow(name, image)

def draw_lips(lips, image):
    for (x, y) in lips:
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

def calc_dist(fp, sp, h, w):
    return float(((fp[0] - sp[0])**2 + (fp[1] - sp[1])**2))*(h**2 + w**2)/ (h**2 * w**2)

def connect_dots(lips, image):
    for i in range(0, len(lips)):
        cv2.line(image,lips[i],lips[(i+1)%len(lips)],(255,255,0))

def save_matrix_in_file(mat, frame_number, folder_name):
    height, width = mat.shape[:2]
    result = {
        'width': width,
        'height': height,
        'frame_number': frame_number,
        'vector': []
    }
    file_name = "frame_%s.json" % str(frame_number)
    for row in range(0, len(mat)):
        for col in range(0, len(mat[0])):
            if row != col:
                result['vector'].append(mat[row][col])
    #data_file = open(os.path.join(folder_name, file_name), 'w')
    #json.dump(result, data_file)
    #data_file.close()
    return result


def delete_extra_lip_points(lip_array):
    lip_array = np.delete(lip_array, np.s_[1::2], 0)
    lip_array = np.delete(lip_array, 6, 0)
    lip_array = np.delete(lip_array,7,0)
    return lip_array


def create_train_element_json(word, path):
    meta_file_name = os.path.join(path, "%s_meta.json" % word)
    if os.path.exists(meta_file_name):
        return
    vector = [0] * config.WORDS_COUNT
    if word in config.NN_CLASSES_ID:
        vector[config.NN_CLASSES_ID[word] - 1] = 1
    result = {
        "word": word,
        "vector": vector
    }
    meta_file = open(meta_file_name, "w")
    json.dump(result, meta_file)
    meta_file.close()


def get_file_content(file_name):
    current_file = open(file_name, "r")
    content = current_file.read()
    current_file.close()
    return content


def get_extra_vector():
    result = []
    for i in range(0, config.FRAME_VECTOR_SIZE):
        result.append(0)
    return result


def fill_to_equal(number, x_sample_folder):
    extra_vector = get_extra_vector()
    while number <= config.FRAME_DIFFERENCE_AMOUNT_LIMIT:
        x_sample_frame = os.path.join(x_sample_folder, str(number) + "_frame.json")
        frame_file = open(x_sample_frame, "w")
        result = {
            "minuend": 0,
            "subtrahend": 0,
            "vector": extra_vector
        }
        json.dump(result, frame_file)
        frame_file.close()
        number += 1

def create_X_train(path=None, word=None, random_str=None, mat = None):
    if mat:
        lenh = len(mat)
        results = []
    else:
        files = glob.glob(os.path.join(path, "*.json"))
        files = sorted(files, key=sort_frames)
        x_sample_folder = os.path.join(path, "../" + word + "_" + random_str)
        os.mkdir(x_sample_folder)
        lenh = len(files)
    number = 0
    for i in range(config.OFFSET_FRAMES, lenh - config.SPLIT_COUNT_FRAMES, config.SPLIT_COUNT_FRAMES):
        number += 1
        if number > config.FRAME_DIFFERENCE_AMOUNT_LIMIT:
            break
        if mat:
            current_vector = mat[i-1]['vector']
            next_vector = mat[i - 1 + config.SPLIT_COUNT_FRAMES]['vector']
        else:
            current_vector = json.loads(get_file_content(files[i]))['vector']
            next_vector = json.loads(get_file_content(files[i + config.SPLIT_COUNT_FRAMES]))['vector']
        subtract_vector = list(np.array(next_vector) - np.array(current_vector))
        result = {
            "minuend": i + config.SPLIT_COUNT_FRAMES,
            "subtrahend": i,
            "vector": subtract_vector
        }
        if mat:
            results.append(result)
        else:
            x_sample_frame = os.path.join(x_sample_folder, str(number) + "_frame.json")
            frame_file = open(x_sample_frame, "w")

            json.dump(result, frame_file)
            frame_file.close()
    if number < config.FRAME_DIFFERENCE_AMOUNT_LIMIT:
        if mat:


            while number <= config.FRAME_DIFFERENCE_AMOUNT_LIMIT:
                result = {
                    "minuend": 0,
                    "subtrahend": 0,
                    "vector": get_extra_vector()
                }
                results.append(result)
                number += 1
        else:
            fill_to_equal(number, x_sample_folder)
    return results[:config.FRAME_DIFFERENCE_AMOUNT_LIMIT]


def sort_frames(text):
    return int(re.split('(\d+)', os.path.basename(text))[1])


def get_video_paths(path):
    videos = []
    for video_format in config.VIDEO_FORMATS:
        videos += glob.glob(os.path.join(path, "*.%s" % video_format))
    return videos


def start_collect_data(instance):
    for word in config.NN_CLASSES_ID.keys():
        folder_path_word = os.path.join(config.DATA_PATH, word)
        if os.path.exists(folder_path_word):
            video_paths = get_video_paths(folder_path_word)
            for video_path in video_paths:
                random_str = str(uuid4())
                instance.mark_up_video(video_path, random_str)


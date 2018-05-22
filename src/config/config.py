import os
# -*- coding: utf-8 -*-
# coding=utf-8
# PROJECT CONFIG
CURRENT_PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(CURRENT_PROJECT_PATH, "../")
# LIP FINDER
PATH_TO_SHAPE_LANDMARK = "config/shape_predictor_68_face_landmarks.dat"
ASPECT_RATIO = 7

VIDEO_FORMATS = ["MOV", "mp4", "mpg","avi"]

COLLECT_DATA = False

FRAME_VECTOR_SIZE = 56
FRAME_DIFFERENCE_AMOUNT_LIMIT = 8

DEBUG = True

# IMAGE UTILS CONFIG
MEDIAN_BLUR_KERNEL_SIZE = 3

NN_CLASSES_ID = {
    "start": 1,
    "stop": 2,
    "zablokiruy": 3,
    "razblokiruy": 4,
    "shifrovanie": 5,
    "edenica": 6,
    "aksioma": 7,
    "autentifikatsia": 8,
    "issledovanie": 9,
    "otsutstvie": 10
}

WORDS = ["start", "stop", "zablokiruy", "razblokiruy", "shifrovanie", "edenica", "aksioma", "autentifikatsia", "issledovanie", "otsutstvie"]
# NN_CLASSES_ID = {
#     "lol": 1
# }

# NN_CLASSES_ID = {
#     "start_test": 1,
#     "stop_test": 2,
#     "zablokiruy_test": 3,
#     "razblokiruy_test": 4,
#     "shifrovanie_test": 5,
#     "edenica_test": 6,
#     "aksioma_test": 7,
#     "autentifikatsia_test": 8,
#     "issledovanie_test": 9,
#     "otsutstvie_test": 10
# }

WORDS_COUNT = 10

OFFSET_FRAMES = 3
SPLIT_COUNT_FRAMES = 3

import os
# -*- coding: utf-8 -*-
# coding=utf-8
# PROJECT CONFIG
CURRENT_PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(CURRENT_PROJECT_PATH, "../data")
# LIP FINDER
PATH_TO_SHAPE_LANDMARK = "config/shape_predictor_68_face_landmarks.dat"
ASPECT_RATIO = 7

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
    "edinitsa": 6,
    "axioma": 7,
    "autentificatsia": 8,
    "issledovanie": 9,
    "otsutstvie": 10,
    "samples": 11
}

OFFSET_FRAMES = 3
SPLIT_COUNT_FRAMES = 3

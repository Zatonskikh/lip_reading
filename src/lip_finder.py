import dlib
import cv2
import config
import os
import utils
from imutils import face_utils
from imutils import resize
from utils import  delete_extra_lip_points
import numpy as np

class LipFinder:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(config.CURRENT_PROJECT_PATH,
                                                      config.PATH_TO_SHAPE_LANDMARK))

    def find_lips(self, image):
        img = utils.image_processing(image)
        gray = utils.get_gray_image(img)

        rects = self.detector(gray, 0)

        # TODO: bug with many faces
        # solution: manipulate with one face or push roi in array
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            self.lips = delete_extra_lip_points(shape[48:])


            (x, y, w, h) = cv2.boundingRect(np.array([self.lips]))
            self.roi = img[y:y + h, x:x + w]
            #self.roi = resize(self.roi, width=w*config.ASPECT_RATIO)

            self.lips = self.recalculate_lips_cor(self.lips, x, y)

    def get_lips_image(self):
        return self.roi

    def get_lips_cor(self):
        return self.lips

    def recalculate_lips_cor(self, lips, x_rect, y_rect):
        new_lips = np.ndarray(self.lips.shape, np.int64)
        for i in range(0, len(lips)):
            x = int(round((lips[i][0] - x_rect)))
            y = int(round((lips[i][1] - y_rect)))
            new_lips[i] = [x, y]

        return new_lips

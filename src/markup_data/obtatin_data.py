import cv2
import numpy as np
import os
from src.utils import show_image
from src.lip_finder import LipFinder
from src.utils import draw_lips
from src.utils import calc_dist, save_matrix_in_file, create_train_element_json, create_X_train
import shutil
import src.config as config
import imutils

class MarkUp:
    def __init__(self):
        self._lip_finder = LipFinder()
        self.error_count = 0

    def mark_up_video(self, path=0, random_str="1"):
        self._path = path
        self._video = cv2.VideoCapture(self._path)
        self.folder_name = None
        if path:
            self.folder_name = os.path.join(os.path.dirname(path),
                                            ".".join(os.path.basename(path).split(".")[:-1]) + "_meta")
            if os.path.exists(self.folder_name) and config.DEBUG:
                shutil.rmtree(self.folder_name)
            os.mkdir(self.folder_name)
        # TOD0: need in main class?
        word = os.path.basename(os.path.normpath(os.path.join(self.folder_name, "..")))
        create_train_element_json(word, os.path.normpath(os.path.join(self.folder_name, "..")))
        self.frame_number = 0
        while (self._video.isOpened()):
            ret, frame = self._video.read()

            self.frame_number+=1
            if (frame is None):
                break

            # if self._path.split(".")[-1]=="MOV":
            #     frame = imutils.rotate(frame,270)

            try:
                self._lip_finder.find_lips(frame)
                img_lips = self._lip_finder.get_lips_image()
            except:
                print self._path
                self.error_count += 1
                return
            h, w = img_lips.shape[:2]
            matrix_dist = self.calc_dist_point_from_other(self._lip_finder.get_lips_cor(), h, w)
            save_matrix_in_file(matrix_dist, self.frame_number, self.folder_name)

            draw_lips(self._lip_finder.get_lips_cor(), img_lips)
            if not config.COLLECT_DATA:
                show_image("lips", img_lips)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # TODO: delete this line
        cv2.destroyAllWindows()
        create_X_train(self.folder_name, word, random_str)

    #TODO: rewrite after building model of lips
    def calc_dist_point_from_other(self, lips, h, w):
        matrix_dist = np.zeros(shape=(len(lips), len(lips)))
        for i in range(len(lips)):
            for j in range(len(lips)):
                if i != j:
                    matrix_dist[i][j] = calc_dist(lips[i], lips[j], h, w)

        return matrix_dist

import numpy as np
import cv2
import imutils
import os
import re
import math

def rotateImage(image, angle, scale = 1.):
    width = image.shape[1]
    height = image.shape[0]

    radAngle = np.deg2rad(angle)

    newWidth = (abs(np.sin(radAngle) * height) + abs(np.cos(radAngle) * width)) * scale
    newHeight = (abs(np.cos(radAngle) * height) + abs(np.sin(radAngle) * width)) * scale

    rot_mat = cv2.getRotationMatrix2D((newWidth * 0.5, newHeight * 0.5), angle, scale)

    rot_move = np.dot(rot_mat, np.array([(newWidth - width) * 0.5, (newHeight - height) * 0.5, 0]))

    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]

    return cv2.warpAffine(image, rot_mat, (int(math.ceil(newWidth)), int(math.ceil(newHeight))),
                      flags=cv2.INTER_LANCZOS4)


def rotateVideo(path):
    cap = cv2.VideoCapture(path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    file_name = ".".join(path.split(".")[:-1]) + ".avi"
    out = cv2.VideoWriter(file_name, fourcc, 20.0, (720,1280))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = rotateImage(frame, 270)
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

def dir_processing(path_to_dir):
    files = os.listdir(path_to_dir)
    videos = filter(lambda file: re.search(".mov", file), files)
    video_list = [os.path.join(path_to_dir, video) for video in videos]
    for video in video_list:
        rotateVideo(video)

if __name__ == '__main__':
    dir_processing("/home/tommyz/PycharmProjects/lip_reading/data")
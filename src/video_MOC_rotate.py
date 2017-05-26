import numpy as np
import cv2
import imutils
import os
import re

def rotateVideo(path):
    cap = cv2.VideoCapture(path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    file_name = ".".join(path.split(".")[:-1]) + ".avi"
    out = cv2.VideoWriter(file_name, fourcc, 20.0, (640,480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = imutils.rotate(frame, 270)
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

def dir_processing(path_to_dir):
    files = os.listdir(path_to_dir)
    videos = filter(lambda file: re.search(".MOV", file), files)
    video_list = [os.path.join(path_to_dir, video) for video in videos]
    for video in video_list:
        rotateVideo(video)

if __name__ == '__main__':
    dir_processing("test")
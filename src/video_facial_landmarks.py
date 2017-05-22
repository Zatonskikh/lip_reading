# coding=utf-8
# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1
# -*- coding: utf-8 -*-
# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import random
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vid = cv2.VideoCapture('bbal9a.mpg')
# TODO: обрезать ебало http://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
while(vid.isOpened()):
  ret, frame = vid.read()

  #gray = cv2.cvtColor
  #print (frame)
  if frame is None: break
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # detect faces in the grayscale frame
  rects = detector(gray, 0)

  # loop over the face detections
  for rect in rects:
	  # determine the facial landmarks for the face region, then
	  # convert the facial landmark (x, y)-coordinates to a NumPy
	  # array
	  shape = predictor(gray, rect)
	  shape = face_utils.shape_to_np(shape)

	  # loop over the (x, y)-coordinates for the facial landmarks
	  # and draw them on the image

	  lips = shape[48:]
	  for (x, y) in lips:
		  cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

  # show the frame
  cv2.imshow("Frame", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
# time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = cv2.medianBlur(frame,3)
	#frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		lips = shape[48:]
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in lips:
			cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)

		(x, y, w, h) = cv2.boundingRect(np.array([lips]))
		roi = frame[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, height=200)

		# show the particular face part
		cv2.imshow("ROI", roi)

	# visualize all facial landmarks with a transparent overlay
	# output = face_utils.visualize_facial_landmarks(frame, shape)
	# cv2.imshow("Image", output)

	# show the frame
	# try:
	# 	cv2.imshow("Frame", face)
	# except:
	# 	print "No face found"

	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
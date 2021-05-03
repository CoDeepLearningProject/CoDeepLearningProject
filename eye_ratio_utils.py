import numpy as np
import dlib
import cv2
import time
import timeit
import matplotlib.pyplot as plt

from scipy.spatial import distance
from imutils import face_utils
from threading import Thread
from threading import Timer
from imutils.video import VideoStream


def eye_aspect_ratio(eye) :
    P1P5 = distance.euclidean(eye[1], eye[5])
    P2P4 = distance.euclidean(eye[2], eye[4])
    P0P3 = distance.euclidean(eye[0], eye[3])

    return (P1P5 + P2P4) / (2.0 * P0P3)

#####################################################################################################################

#   eye aspect ratio function
#   EAR = ((p2 - p6) + (p3 - p5)) / 2 * (p1 - p4)
#   indexing start with 0

#####################################################################################################################

OPEN_EAR = 0  # For init_open_ear()
EAR_THRESH = 0  # Threashold value

print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("starting video stream thread...")
videoFile = './SampleVideo.mp4'
cap = cv2.VideoCapture(videoFile)

#time.sleep(1.0)
#####################################################################################################################
import imutils

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(gray, face)

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

            landmarks = face_utils.shape_to_np(landmarks)

            leftEye = landmarks[lStart:lEnd]
            rightEye = landmarks[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('VideoFrame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        print("Video finished...")
        break

cap.release()
cv2.destroyAllWindows()
import numpy as np
import dlib
import cv2
import time
import timeit

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

EAR_THRESH = 0  # Threashold value

print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("starting video stream thread...")
# videoFile = './SampleVideo.mp4'
videoFile = './VideoSample3.mp4'

cap = cv2.VideoCapture(videoFile)

# check EAR with Open Eyes, Closed Eyes with Python thread
def getEARTHRESHWithThread() :
    time.sleep(2)
    print("Init getAverageEar")

    ear_list = []
    # for _ in range(15):
    for _ in range(10):
        ear_list.append(average_EAR)
        time.sleep(1)

    global EAR_THRESH
    EAR_THRESH = (sum(ear_list) / len(ear_list)) - .01

    print("EAR_THRESH={:.4F}".format(EAR_THRESH))
    print("Finish getEARWithOpenEyes")


get_Average_ear = Thread(target=getEARTHRESHWithThread)
get_Average_ear.daemon = True
get_Average_ear.start()

#####################################################################################################################
import imutils


def light_removing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    #lab -> l 명도, a green의 보색, b blue의 보색
    L = lab[:, :, 0]
    med_L = cv2.medianBlur(L, 99)
    invert_L = cv2.bitwise_not(med_L)
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)
    return L, composed

DROWSY_FLAG = False
COUNTER = 0
COUNT_THRESH = 20

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=450)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        L, gray = light_removing(frame)
        original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('OriginalGray', original_gray)
        cv2.imshow('Gray', gray)
        #cv2.imshow('L', L)
        faces = detector(gray, 0)

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

            average_EAR = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # print(EAR_THRESH)
            # print(average_EAR)
            if average_EAR < EAR_THRESH:
                if not DROWSY_FLAG:
                    start_drowsy_detection = timeit.default_timer()
                    DROWSY_FLAG = True
                COUNTER += 1

                if COUNTER > COUNT_THRESH:
                    print("drowsy...")
                    # TODO
                    # Implement Alarm

            else:
                DROWSY_FLAG = False
                COUNTER = 0


            cv2.putText(frame, "EAR: {:.2f}".format(average_EAR), (300, 30),
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

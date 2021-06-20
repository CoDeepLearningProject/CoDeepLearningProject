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
videoFile = 0

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

# head pose detection

FACIAL_LANDMARK_INDEX = [33, 8, 36, 45, 48, 64]     # index of nose, chin, left eye, right eye, left mouth, right mouth

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner

                        ])

def return_face_pose(facial_landmarks, size):
    # setting the 2D image points

    image_points = np.array(facial_landmarks, dtype="double")

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

                    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return (p1,p2)                       

def getHEADPOSETHRESHWithThread():
    time.sleep(2)
    print("Init getAverageHeadPose")
    headPoseList = []
    for _ in range(20):
        headPoseList.append(head_direction_y)
        # print(head_pose_points[1][1])
        time.sleep(0.5)
    global headPoseThresh
    headPoseThresh = (sum(headPoseList) / len(headPoseList)) * 0.7
    print("Head Pose Thresh is {:.4F}".format(headPoseThresh))
    print("finish get headPoseThresh")

headPoseThresh = -1000

get_average_head = Thread(target = getHEADPOSETHRESHWithThread)
get_average_head.daemon = True
get_average_head.start()



import imutils


def light_removing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # lab -> l 명도, a green의 보색, b blue의 보색
    L = lab[:, :, 0]
    med_L = cv2.medianBlur(L, 99)
    invert_L = cv2.bitwise_not(med_L)
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)
    return L, composed

EAR_DROWSY_FLAG = False
EAR_COUNTER = 0
COUNT_THRESH = 20

HEADPOSE_DROWSY_FLAG = False
HAEDPOSE_COUNTER = 0


import pygame

def make_alarm_sound (path) :
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

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

            # EAR
            leftEye = landmarks[lStart:lEnd]
            rightEye = landmarks[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            average_EAR = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)



            # head pose

            facial_points = []

            for num in FACIAL_LANDMARK_INDEX:
                facial_points.append((landmarks[num][0], landmarks[num][1]))

            head_pose_points = return_face_pose(facial_points, frame.shape)
            cv2.line(frame, head_pose_points[0], head_pose_points[1], (0,0,255), 2)

            head_direction_y = head_pose_points[0][1] - head_pose_points[1][1]

            # check EAR_THRESH
            if average_EAR < EAR_THRESH:
                if not EAR_DROWSY_FLAG:
                    start_drowsy_detection = timeit.default_timer()
                    EAR_DROWSY_FLAG = True
                EAR_COUNTER += 1

                if EAR_COUNTER > COUNT_THRESH:
                    # print("drowsy...")
                    cv2.putText(frame, "EAR warning", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # TODO
                    # Implement Alarm
                    t = Thread(target=make_alarm_sound, args=("./short_alarm.mp3", ))
                    t.daemon = True
                    t.start()

            else:
                EAR_DROWSY_FLAG = False
                EAR_COUNTER = 0



            # check head pose
            if head_direction_y < headPoseThresh:
                if not HEADPOSE_DROWSY_FLAG:
                    HEADPOSE_DROWSY_FLAG = True
                HAEDPOSE_COUNTER += 1
                if HAEDPOSE_COUNTER > COUNT_THRESH:
                    cv2.putText(frame, "Head pose warning", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    t = Thread(target=make_alarm_sound, args=("./short_alarm.mp3", ))
                    t.daemon = True
                    t.start()
            else:
                HEADPOSE_DROWSY_FLAG = False
                HAEDPOSE_COUNTER = 0

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
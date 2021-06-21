import time
import timeit
from threading import Thread

import cv2

from alarm import ring_alarm
from ear import calculate_ear
from head_pose import calculate_head_pose
from preprocess import preprocess

EAR_DROWSY_START = 0
EAR_DROWSY_TIME_THRESH = 0.7
EAR = 0
EAR_DROWSY_FLAG = False
EAR_THRESH = 0

def getEARTHRESHWithThread() :
    global EAR

    time.sleep(2)
    print("Init getAverageEar")

    ear_list = []
    # for _ in range(15):
    for _ in range(10):
        ear_list.append(EAR)
        time.sleep(1)

    global EAR_THRESH
    EAR_THRESH = (sum(ear_list) / len(ear_list)) - .01

    print("EAR_THRESH={:.4F}".format(EAR_THRESH))
    print("Finish getEARWithOpenEyes")

# 프로그램 진입점
if __name__ == "__main__":
    # 카메라
    src = 0

    # 동영상 파일
    # src = "VideoSample3.mp4"

    cap = cv2.VideoCapture(0)

    # EAR_THRESH 계산 쓰레드
    get_Average_ear = Thread(target=getEARTHRESHWithThread)
    get_Average_ear.daemon = True
    get_Average_ear.start()

    prev = timeit.default_timer() - 1
    while(cap.isOpened()):
        now = timeit.default_timer()
        loop_time = now - prev
        fps = 1 / loop_time
        print(f"FPS: {fps}")
        prev = now

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 전처리
        L, gray, rgb, bgr = preprocess(frame)

        # head pose 계산
        headPose = calculate_head_pose(rgb, bgr)

        # EAR 계산
        EAR = calculate_ear(rgb, draw=bgr)
        if EAR is not None:
             # `ear`이 `EAR_THRESH`보다 `TIME_THRESH` 초 이상 낮으면 졸음으로 판단
            if EAR < EAR_THRESH:
                if not EAR_DROWSY_FLAG:
                    EAR_DROWSY_START = timeit.default_timer()
                    EAR_DROWSY_FLAG = True

                if (timeit.default_timer() - EAR_DROWSY_START) > EAR_DROWSY_TIME_THRESH:
                    cv2.putText(bgr, 'EAR_Drowsiness!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    #print('drowsiness...')
                    ring_alarm()
            else:
                EAR_DROWSY_FLAG = False

        # print(f"EAR = {ear}")
        # print(f"Head pose = {head_pose})

        # ESC로 종료
        cv2.imshow("Frame", bgr)
        if cv2.waitKey(100) & 0xFF == 27:
            break

    cap.release()

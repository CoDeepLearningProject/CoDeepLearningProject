import timeit
from threading import Thread

import cv2

from alarm import ring_alarm
from ear import calculate_ear
# from head_pose import calculate_head_pose
from preprocess import preprocess
import head_pose

EAR_DROWSY_FLAG = False
EAR_DROWSY_RATIO = 0.5
EAR_MAX = 0
EAR_MIN = 10
EAR_THRESH = 0
NOW = 0
START = 0
TIME_THRESH = 0.7

# 프로그램 진입점
if __name__ == "__main__":
    # 카메라
    src = 0

    # 동영상 파일
    # src = "VideoSample3.mp4"

    cap = cv2.VideoCapture(0)

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

        # EAR 계산
        ear = calculate_ear(rgb, draw=bgr)
        # EAR 최솟값과 최댓값 사이의 값을 임계값으로 지정
        if ear > EAR_MAX:
            EAR_MAX = ear
        if ear < EAR_MIN:
            EAR_MIN = ear
        EAR_THRESH = EAR_DROWSY_RATIO * EAR_MAX + (1 - EAR_DROWSY_RATIO) * EAR_MIN

        # head pose 계산
        headPose = head_pose.calculate_head_pose(rgb, bgr)

        # `ear`이 `EAR_THRESH`보다 `TIME_THRESH` 초 이상 낮으면 졸음으로 판단
        if ear < EAR_THRESH:
            if not EAR_DROWSY_FLAG:
                START = timeit.default_timer()
                EAR_DROWSY_FLAG = True

            NOW = timeit.default_timer()
            if (NOW - START) > TIME_THRESH:
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

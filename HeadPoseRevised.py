import cv2
import numpy as np
from whenet import WHENet
import os
import winsound as sd
from math import cos, sin
from YOLO.yolo_postprocess import YOLO
from PIL import Image

def beepsound():
    fr = 2000
    du = 1000
    sd.Beep(fr, du)

def light_removing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # lab -> l 명도, a green의 보색, b blue의 보색
    L = lab[:, :, 0]
    med_L = cv2.medianBlur(L, 99)
    invert_L = cv2.bitwise_not(med_L)
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)
    return L, composed

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    r = (0,0,255)
    g = (0,255,0)
    b = (255,0,0)

    if tdx == None or tdy == None:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X
    xAxis_x = size * (cos(yaw) * cos(roll)) + tdx
    xAxis_y = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y
    yAxis_x = size * (-cos(yaw) * sin(roll)) + tdx
    yAxis_y = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z
    zAxis_x = size * (sin(yaw)) + tdx
    zAxis_y = size * (-cos(yaw) * sin(pitch)) + tdy

    #위 값들: 성분벡터, 이를 통해 선 그리기
    cv2.line(img, (int(tdx), int(tdy)), (int(xAxis_x),int(xAxis_y)),r,2)
    cv2.line(img, (int(tdx), int(tdy)), (int(yAxis_x),int(yAxis_y)),g,2)
    cv2.line(img, (int(tdx), int(tdy)), (int(zAxis_x),int(zAxis_y)),b,2)

    print(int(xAxis_x) - int(xAxis_y))
    print(int(yAxis_x) - int(yAxis_y))
    print(int(zAxis_x) - int(zAxis_y))
    #print('zAxis_x: {} zAxis_y: {}'.format(zAxis_x, zAxis_y))
    print()

    if (int(zAxis_x) - int(zAxis_y)) < 15: #임계값
        cv2.putText(img, 'Drowsiness!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)

    return img

def process_detection(model, img, bbox):
    yMin, xMin, yMax, xMax = bbox

    yMin = max(0, yMin - abs(yMin - yMax) / 10)
    yMax = min(img.shape[0], yMax + abs(yMin - yMax) / 10)

    xMin = max(0, xMin - abs(xMin - xMax) / 5)
    xMax = min(img.shape[1], xMax + abs(xMin - xMax) / 5)
    xMax = min(xMax, img.shape[1])

    rgbImg = img[int(yMin):int(yMax), int(xMin):int(xMax)]
    rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
    rgbImg = cv2.resize(rgbImg, (224, 224))
    rgbImg = np.expand_dims(rgbImg, axis=0)

    cv2.rectangle(img, (int(xMin), int(yMin)), (int(xMax), int(yMax)), (0,0,0), 2)
    yaw, pitch, roll = model.get_angle(rgbImg)
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])

    draw_axis(img, yaw, pitch, roll, tdx=(xMin+xMax)/2, tdy=(yMin+yMax)/2, size = abs(xMax-xMin)//2)
    return img

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    whenet = WHENet(snapshot='WHENet.h5')
    yolo = YOLO()

    src = 0  # 웹캠이 없다면 파일 경로 넣기
    cap = cv2.VideoCapture(src)

    if src == 0:
        print('using web cam')
    else:
        print('using video, path: {}'.format(src))

    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('headPoseOutput.avi', fourcc, 30, (frame.shape[1], frame.shape[0]))

    while True:
        try:
            ret, frame = cap.read()
        except:
            break
        L, gray = light_removing(frame)

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgbFrame)
        bboxes, scores, classes = yolo.YOLO_DetectProcess(img_pil)

        #이미지 전처리 사용 시: frame = gray
        for bbox in bboxes:
            frame = process_detection(whenet, frame, bbox)

        cv2.imshow('output', frame)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
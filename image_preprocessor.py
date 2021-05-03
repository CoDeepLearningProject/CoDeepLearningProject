import cv2
import numpy as np
import time

########
#20210503
#이미지를 불러와서 조명 완화, 흑백 처리까지만
#영상 처리와 프레임 단위 리턴 구현 중
########
clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

img = cv2.imread('sample_human.png')

x,y,h,w = 550,250,400,300
# img = img[y:y+h, x:x+w]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayimg = gray

GLARE_MIN = np.array([0, 0, 50],np.uint8)
GLARE_MAX = np.array([0, 0, 225],np.uint8)

hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)
result = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA)

lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
lab_planes1 = cv2.split(lab1)
clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes1[0] = clahe1.apply(lab_planes1[0])
lab1 = cv2.merge(lab_planes1)
clahe_bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)


# fps = 1./(time.time()-t1)
# cv2.putText(clahe_bgr1    , "FPS: {:.2f}".format(fps), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255))

res = cv2.cvtColor(clahe_bgr1, cv2.COLOR_BGR2GRAY)

cv2.imshow("Originalimg", img)
cv2.imshow("res", res)
cv2.waitKey(0)
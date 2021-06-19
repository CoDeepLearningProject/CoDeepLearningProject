import cv2

def preprocess(frame):
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # lab -> l 명도, a green의 보색, b blue의 보색
    L = lab[:, :, 0]
    med_L = cv2.medianBlur(L, 99)
    invert_L = cv2.bitwise_not(med_L)
    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)

    return L, composed, rgb, frame
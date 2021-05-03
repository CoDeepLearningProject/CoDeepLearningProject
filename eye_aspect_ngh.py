# 남기훈
# Dependencies: `dlib`, `opencv-python`, `imutils`


from imutils import face_utils
import cv2
import dlib
import numpy as np


def preprocess(image):
    preprocessed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return preprocessed_image


def calculate_ear(els):
    """
    Parameters
    ----------
    els: list
        A list of eye landmarks
    """
    return (dist(els[1], els[5]) + dist(els[2], els[4])) / (2 * dist(els[0], els[3]))


def calculate_ears(fls):
    """
    Parameters
    ----------
    fls: list
        A list of face landmarks
    """
    return (calculate_ear(fls[36:42]), calculate_ear(fls[42:48]))


def dist(x, y):
    """
    Calculate distance between two points
    """
    return np.linalg.norm(x - y)


if __name__ == "__main__":
    print(f"dlib {dlib.__version__}")
    print(f"OpenCV {cv2.__version__}")

    # 카메라
    # cap = cv2.VideoCapture(0, cv2.CAP_ANY)

    # 비디오 파일
    cap = cv2.VideoCapture("./SampleVideo.mp4")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while cap.isOpened():
        ret, frame = cap.read()
        preprocessed_frame = preprocess(frame)

        rects = detector(preprocessed_frame, 0)
        left_ear, right_ear = (0, 0)

        for rect in rects:
            shape = predictor(preprocessed_frame, rect)
            shape = face_utils.shape_to_np(shape)

            left_ear, right_ear = calculate_ears(shape)

            for (i, (x, y)) in enumerate(shape):
                cv2.circle(preprocessed_frame, (x, y), 1, (0, 0, 255), -1)

        print(f"Left EAR: {left_ear}\nRight EAR: {right_ear}\n")
        cv2.imshow("preprocessed_frame", preprocessed_frame)

        if cv2.waitKey(100) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

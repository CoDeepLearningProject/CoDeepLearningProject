from imutils import face_utils
import cv2
import dlib
import numpy as np

def return_face_pose(facial_landmarks, size):
    # setting the 2D image points

    image_points = np.array(facial_landmarks, dtype="double")

    # setting the 3D model points
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner

                            ])
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
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return (p1,p2)



if __name__ == "__main__":
    
    # [NOSE_INDEX, NOSE_INDEX, LEFT_EYE_INDEX, RIGHT_EYE_INDEX, LEFT_MOUTH_INDEX, RIGHT_MOUSE_INDEX]
    FACIAL_LANDMARK_INDEX = [33, 8, 36, 45, 48, 64]

    # video read
    cap = cv2.VideoCapture("./SampleVideo3.mov")

    # dlib detector object 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)
            
            for face in faces:
                
                facial_points = []

                shape = predictor(gray, face)

                for num in FACIAL_LANDMARK_INDEX:
                    facial_points.append((shape.part(num).x, shape.part(num).y))

                head_pose_points = return_face_pose(facial_points, frame.shape)

                cv2.line(frame, head_pose_points[0], head_pose_points[1], (255,0,0), 2)
                cv2.imshow('VideoFrame', frame)


                if cv2.waitKey(100) == ord('q'):
                    break
        else:
            break

    cap.release()
    cv2.destroyAllwindows()
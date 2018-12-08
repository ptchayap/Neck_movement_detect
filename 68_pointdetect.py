from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\Image_Processing\Project\Code\Neck_movement_detect-master\Neck_movement_detect\shape_predictor_68_face_landmarks.dat')
capture = cv2.VideoCapture(0)
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])
 
 
# Camera internals
 

while True:
    ret,frame = capture.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = detector(gray,1)
    size = frame.shape
    try:
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            point = [36,30,45,48,54,8]
            for x in [30]:
                cv2.circle(frame, (shape[x,0], shape[x,1]), 1, (0, 0, 255), -1)

        image_points = np.array([
                                (shape[30,0], shape[30,1]),     # Nose tip
                                (shape[8,0], shape[8,1]),       # Chin
                                (shape[36,0], shape[36,1]),     # Left eye left cornerg
                                (shape[45,0], shape[45,1]),     # Right eye right corne
                                (shape[48,0], shape[48,1]),     # Left Mouth corner
                                (shape[54,0], shape[54,1])      # Right mouth corner
                        ], dtype="double")
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rotation_vector,_ = cv2.Rodrigues(rotation_vector)
        #K = cv2.decomposeProjectionMatrix()
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        cv2.circle(frame, (int(image_points[0,0]), int(image_points[0,1])), 3, (0,0,255), -1)
        projectoin_matrix = np.hstack((rotation_vector,translation_vector))
        _,_,_,_,_,_,A = cv2.decomposeProjectionMatrix(projectoin_matrix)
        print(A)
        pitch, yaw, roll = [math.radians(_) for _ in A]
        print(yaw)
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (255,0,0), 2)
    except:
        pass
        
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

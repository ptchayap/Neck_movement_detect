import numpy as np
import dlib
import imutils
import cv2
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'E:\Image_Processing\Project\Code\Neck_movement_detect-master\shape_predictor_5_face_landmarks.dat')

capture = cv2.VideoCapture(0)
while True:
	ret,image = capture.read()
	size = image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	for rect in rects:
	# compute the bounding box of the face and draw it on the
		# frame
		(bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
 
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw each of them
		for (i, (x, y)) in enumerate(shape):
			if i ==0:
				righteye = [x,y]
			elif i==2:
				lefteye = [x,y]
			elif i==4:
				nose =[x,y]
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	center_x = int((righteye[0]+lefteye[0])/2)
	center_y = int((righteye[1]+lefteye[1])/2)

#2D image points. If you change the image, you need to change vector
	image_points = np.array([
    	                        (nose[0], nose[1]),
        	                    (lefteye[0],lefteye[1]),     # Left eye left corner
            	                (righteye[0], righteye[1]),     # Right eye right corner
								(center_x,center_y)
                    	    ], dtype="double")

# # 3D model points.
	model_points = np.array([
    	                        (0,0,0),
        	                    (-225.0,170.0, -135.0),     # Left eye left corner
            	                (225.0,170.0, -135.0),      # Right eye right corne 
								(0.0, 170.0, -100.0) 		# Nose tip
								            			                     
                    	    ])


	focal_length = size[1]
	center = (size[1]/2, size[0]/2)
	camera_matrix = np.array(
    	                     [[focal_length, 0, center[0]],
        	                 [0, focal_length, center[1]],
            	             [0, 0, 1]], dtype = "double"
                	         )

	
 
# print "Camera Matrix :\n {0}".format(camera_matrix)
 
	dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=0)
 
# print "Rotation Vector:\n {0}".format(rotation_vector)
# print "Translation Vector:\n {0}".format(translation_vector)
 
 
# # Project a 3D point (0, 0, 1000.0) onto the image plane.
# # We use this to draw a line sticking out of the nose
 
 
	(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
 
	for p in image_points:
		cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
 
 
	p1 = ( int(image_points[0][0]), int(image_points[0][1]))
	p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
	cv2.line(image, p1, p2, (255,0,0), 2)


# Display image
	cv2.imshow("Output", image)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

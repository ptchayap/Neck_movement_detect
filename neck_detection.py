import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage  
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
import cv2
import dlib
import math
from imutils import face_utils
import imutils
import time
import numpy as np
import argparse
from collections import deque

class MyApp(QDialog):
    def __init__(self):
        super(MyApp,self).__init__()
        loadUi(r'E:\Image_Processing\Project\Code\Neck_movement_detect-master\Neck_movement_detect\neck_movement.ui',self)
        self.image = None
        self.nose_init_x = 0
        self.raiselow.setChecked(True)
        self.tilt.setChecked(True)
        self.rotation.setChecked(True)
        self.start.clicked.connect(self.start_webcam)
        self.cap.clicked.connect(self.stop_webcam)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r'E:\Image_Processing\Project\Code\Neck_movement_detect-master\Neck_movement_detect\shape_predictor_68_face_landmarks.dat')
        

    def start_webcam(self):
        self.capture = cv2.VideoCapture(1)
        self.count = 0
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,300)
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,400)
        self.timer = QTimer(self)
        
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
        self.deg = 0
        
        
    
    def update_frame(self):
        _,self.image = self.capture.read()
        self.image = cv2.flip(self.image,1)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.rects = self.detector(self.gray, 0) #detect face in the current frame (var = image)
        for rect in self.rects:
            self.shape = self.predictor(self.gray, rect)
            self.shape = face_utils.shape_to_np(self.shape)
            point = [36,27,45,30,48,54,8] # L eye L corner, center of 2 eyes, R eye R corner, nose, L mouth angle, R mouth angle, chin
            self.key = np.zeros(14,dtype='int').reshape(-1,2)
            for i, x in enumerate(point):
                if i==1:
                    self.key[i,0],self.key[i,1] = (self.shape[39,0]+self.shape[42,0])/2 , (self.shape[39,1]+self.shape[42,1])/2
                else:
                    self.key[i,0] = self.shape[x,0]
                    self.key[i,1] = self.shape[x,1]

        if self.raiselow.isChecked() == True :
            self.cal_raiselow()
        if self.rotation.isChecked() == True:
            self.cal_rotation()
        if self.tilt.isChecked() == True:
            self.cal_tilt()
        self.displayImage(self.image,1)
        


    def stop_webcam(self):
        self.timer.stop()
    
    def displayImage(self,img,window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3: #[0]=rows [1]=col [2]=channel
            if img.shape[2] ==4:
                qformat=QImage.Format_RGBA888
            else:
                qformat=QImage.Format_RGB888
        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage = outImage.rgbSwapped()
        if window ==1:
            self.imglabel.setPixmap(QPixmap.fromImage(outImage))
            self.imglabel.setScaledContents(True)

    def cal_rotation(self):  
        #3D model points
        image_points = np.array([(self.key[3,0], self.key[3,1]),     # Nose tip
                                (self.key[6,0], self.key[6,1]),       # Chin
                                (self.key[0,0], self.key[0,1]),     # Left eye left cornerg
                                (self.key[2,0], self.key[2,1]),     # Right eye right corne
                                (self.key[4,0], self.key[4,1]),     # Left Mouth corner
                                (self.key[5,0], self.key[5,1])      # Right mouth corner
                                    ], dtype="double")
        model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # camera internals
        size = self.image.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
    	                     [[focal_length, 0, center[0]],
        	                 [0, focal_length, center[1]],
            	             [0, 0, 1]], dtype = "double"
                	         )
	    #print("Camera Matrix :\n {0}".format(camera_matrix))
        dist_coeffs = np.zeros((4,1)) #Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=0)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        rotation_matrix,_ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix,translation_vector))
        _,_,_,_,_,_,A = cv2.decomposeProjectionMatrix(projection_matrix)
        _,yaw,_ = A
        cv2.circle(self.image, (int(image_points[0][0]), int(image_points[0][1])), 3, (0,0,255), -1)
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(self.image, p1, p2, (255,0,0), 2)
        self.deg = yaw
        self.lcd.display(self.deg)

    def cal_tilt(self):
        
        cv2.line(self.image,(self.key[1,0],self.key[1,1]),(self.key[3,0],self.key[3,1]),(0,255,0),2)
        dis_x = self.key[1,0]-self.key[3,0]
        cen_to_nose = math.sqrt(pow(self.key[1,0]-self.key[3,0],2)+pow(self.key[1,1]-self.key[3,1],2))
        try:
            rad = math.asin(dis_x/cen_to_nose)
            self.deg = int(math.degrees(rad))
        except:
            pass
       
        self.lcd.display(self.deg)
        # return(degree)
    def cal_raiselow(self):
        lower_pink = np.array([160, 100, 150])
        upper_pink = np.array([180, 255, 255])
        lower_red = np.array([150, 150, 50])
        upper_red = np.array([180, 255, 150])
        lower_green = np.array([40, 100, 50])
        upper_green = np.array([80, 255, 255])
        lower_blue = np.array([80, 60, 50])
        upper_blue = np.array([150, 255, 150])

        ref_rad = None
        new_rad = None
        count = 0
        angle = None

        ap = argparse.ArgumentParser()
        ap.add_argument("-b", "--buffer", type=int, default=64,
                        help="max buffer size")
        args = vars(ap.parse_args())
        pts = deque(maxlen=args["buffer"])

        threshold = 100  # TODO Adapt to your needs.
        kernel = np.ones((5, 5), np.uint8)
        blurred = cv2.GaussianBlur(self.image, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask_red = cv2.inRange(hsv, lower_pink, upper_pink)
        mask_red = cv2.erode(mask_red, kernel, iterations=2)
        mask_red = cv2.dilate(mask_red, kernel, iterations=2)

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_green = cv2.erode(mask_green, kernel, iterations=2)
        mask_green = cv2.dilate(mask_green, kernel, iterations=2)

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_blue = cv2.erode(mask_blue, kernel, iterations=2)
        mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)

        result = cv2.bitwise_and(blurred, blurred, mask=mask_red+mask_green+mask_blue)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts_r = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts_r = cnts_r[0] if imutils.is_cv2() else cnts_r[1]
        center_r = None

        cnts_g = cv2.findContours(mask_green.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts_g = cnts_g[0] if imutils.is_cv2() else cnts_g[1]
        center_g = None

        cnts_b = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cnts_b = cnts_b[0] if imutils.is_cv2() else cnts_b[1]
        center_b = None
        # only proceed if at least one contour was found
        if len(cnts_r) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c_r = max(cnts_r, key=cv2.contourArea)
            ((x_r, y_r), radius_r) = cv2.minEnclosingCircle(c_r)
            M_r = cv2.moments(c_r)
            center_r = (int(M_r["m10"] / M_r["m00"]), int(M_r["m01"] / M_r["m00"]))

            # only proceed if the radius meets a minimum size
            if radius_r > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(self.image, (int(x_r), int(y_r)), int(radius_r),
                           (0, 255, 255), 2)
                cv2.circle(self.image, center_r, 5, (0, 0, 255), -1)


            # line
                # update the points queue
            pts.appendleft(center_r)
            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                cv2.line(self.image, pts[i - 1], pts[i], (0, 0, 255), thickness)


        if len(cnts_g) > 0:
            c_g = max(cnts_g, key=cv2.contourArea)
            ((x_g, y_g), radius_g) = cv2.minEnclosingCircle(c_g)
            M_g = cv2.moments(c_g)
            center_g = (int(M_g["m10"] / M_g["m00"]), int(M_g["m01"] / M_g["m00"]))

            if radius_g > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(self.image, (int(x_g), int(y_g)), int(radius_g),
                           (0, 255, 255), 2)
                cv2.circle(self.image, center_g, 5, (0, 0, 255), -1)

        if len(cnts_b) > 0:
            c_b = max(cnts_b, key=cv2.contourArea)
            ((x_b, y_b), radius_b) = cv2.minEnclosingCircle(c_b)
            M_b = cv2.moments(c_b)
            center_b = (int(M_b["m10"] / M_b["m00"]), int(M_b["m01"] / M_b["m00"]))

            if radius_b > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(self.image, (int(x_b), int(y_b)), int(radius_b),
                           (0, 255, 255), 2)
                cv2.circle(self.image, center_b, 5, (0, 0, 255), -1)
        # make triangle
        print(self.count)
        if len(cnts_r and cnts_g and cnts_b) > 0:
            if self.count == 0:
                ref_c7_to_aom = math.sqrt(((center_g[0] - center_b[0]) ** 2) + ((center_g[1] - center_b[1]) ** 2))
                ref_aom_to_chin = math.sqrt(((center_b[0] - center_r[0]) ** 2) + ((center_b[1] - center_r[1]) ** 2))
                ref_c7_to_chin = math.sqrt(((center_g[0] - center_r[0]) ** 2) + ((center_g[1] - center_r[1]) ** 2))
                self.ref_rad = math.acos((ref_aom_to_chin**2+ref_c7_to_aom**2-ref_c7_to_chin**2)/(2*ref_c7_to_aom*ref_aom_to_chin))
                self.count = self.count + 1
                self.new_rad = 0
                
            else:
                c7_to_aom = math.sqrt(((center_g[0] - center_b[0]) ** 2) + ((center_g[1] - center_b[1]) ** 2))
                aom_to_chin = math.sqrt(((center_b[0] - center_r[0]) ** 2) + ((center_b[1] - center_r[1]) ** 2))
                c7_to_chin = math.sqrt(((center_g[0] - center_r[0]) ** 2) + ((center_g[1] - center_r[1]) ** 2))
                self.new_rad = math.acos((aom_to_chin ** 2 + c7_to_aom ** 2 - c7_to_chin ** 2) / (2 * c7_to_aom * aom_to_chin))
        # calculate different angle
        try:
            self.deg = math.degrees(self.new_rad)-math.degrees(self.ref_rad)
            self.deg = int(self.deg)
            print(self.deg)
        except:
            pass
        self.lcd.display(self.deg)
        
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.setWindowTitle('Neck GUI')
    window.show()
    sys.exit(app.exec_())

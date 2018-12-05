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

class MyApp(QDialog):
    def __init__(self):
        super(MyApp,self).__init__()
        loadUi(r'.\neck_movement.ui',self)
        self.image = None
        self.nose_init_x = 0
        self.raiselow.setChecked(True)
        self.tilt.setChecked(True)
        self.rotation.setChecked(True)
        self.start.clicked.connect(self.start_webcam)
        self.cap.clicked.connect(self.stop_webcam)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r'.\shape_predictor_5_face_landmarks.dat')
    
    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,250)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,400)
        self.timer = QTimer(self)
        
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
        self.deg =0
        
    
    def update_frame(self):
        ret,self.image = self.capture.read()
        self.image = cv2.flip(self.image,1)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.rects = self.detector(self.gray, 0) #detect face in the current frame (var = image)
        for rect in self.rects:
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            self.shape = self.predictor(self.gray, rect)
            self.shape = face_utils.shape_to_np(self.shape)
            for (i, (x, y)) in enumerate(self.shape):
                if i == 1:
                    self.righteye = [x,y]
                elif i == 3:
                    self.lefteye = [x,y]
                elif i == 4:
                    self.nose = [x,y]
                cv2.circle(self.image, (x, y), 1, (0, 0, 255), -1)
       
        if self.raiselow.isChecked() == True :
            pass
        if self.rotation.isChecked() == True:
            print("detect rotation")
            self.cal_rotation()
        if self.tilt.isChecked() == True:
            print("detect tilt")
            self.cal_tilt()


        self.displayImage(self.image,1)
        # self.lcd.display(self.deg)


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
        center_x = int((self.righteye[0]+self.lefteye[0])/2)
        center_y = int((self.righteye[1]+self.lefteye[1])/2)
        #2D image points
        image_points = np.array([
    	                        (self.nose[0], self.nose[1]),
        	                    (self.lefteye[0],self.lefteye[1]),     # Left eye left corner
            	                (self.righteye[0], self.righteye[1]),     # Right eye right corner
								(center_x,center_y)
                    	        ], dtype="double")
        y_dis_pixel = abs(self.nose[1]-center_y)
        y_dis_mm = 170

        #3D model points
        model_points = np.array([
    	                            (0,0,0),                    # nose tip
        	                        (-225.0,170.0, -35.0),     # Left eye left corner
            	                    (225.0,170.0, -35.0),      # Right eye right corne 
								    (0.0, 170.0, 0.0) 		    # center of eyes
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
        for p in image_points:
            cv2.circle(self.image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(self.image, p1, p2, (255,0,0), 2)
        try:
        # Angle calculation
            x_dis = abs(p1[0]-p2[0])
            z_dis = 1000*(y_dis_pixel/y_dis_mm)
            radian = math.asin(x_dis/z_dis)
            angle = math.degrees(radian)
            self.deg = int(angle)
        except:
            pass
        self.lcd.display(self.deg)

    def cal_tilt(self):
        
        center_x = int((self.righteye[0]+self.lefteye[0])/2)
        center_y = int((self.righteye[1]+self.lefteye[1])/2)
		
        nose_x = self.nose[0]
        nose_y = self.nose[1]
       
        cv2.circle(self.image,(center_x,center_y),1,(0,0,255),-1)
        cv2.line(self.image,(center_x,center_y),(nose_x,nose_y),(0,255,0),2)

        dist_x = nose_x - center_x
         
        cen_to_nose = math.sqrt(pow(center_x-nose_x,2)+pow(center_y-nose_y,2))
        try:
            rad = math.asin(dist_x/cen_to_nose)
            self.deg = int(math.degrees(rad))
        except:
            pass
       
        self.lcd.display(self.deg)
        # return(degree)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.setWindowTitle('Neck GUI')
    window.show()
    sys.exit(app.exec_())

    #yyyyyyyy
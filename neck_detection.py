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
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,250)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,400)
        self.timer = QTimer(self)
        
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
        self.deg = 0
        
    
    def update_frame(self):
        ret,self.image = self.capture.read()
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
            pass
        if self.rotation.isChecked() == True:
            self.cal_rotation()
        if self.tilt.isChecked() == True:
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
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.setWindowTitle('Neck GUI')
    window.show()
    sys.exit(app.exec_())

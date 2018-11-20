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

class MyApp(QDialog):
    def __init__(self):
        super(MyApp,self).__init__()
        loadUi('neck_movement.ui',self)
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
            pass
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
        pass

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
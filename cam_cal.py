import numpy as np
import cv2
import glob
     
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
     
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
count = 0
images = glob.glob(r'E:\Image_Processing\Project\Code\*.jpg')
print(images)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
    
    
        # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
    
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
    
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,7), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img = cv2.imread(r'E:\Image_Processing\Project\Code\new.jpg')
h,w = img.shape[:2]
print(h,w)
#dist = np.array([-0.13615181, 0.53005398, 0, 0, 0])
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort 1
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# undistort 2
#mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# crop the image
x,y,w,h = roi
print(newcameramtx)
print(roi)
dst = dst[y:y+h, x:x+w]
while True :
    cv2.imshow('img',dst)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
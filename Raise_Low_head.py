import cv2
import numpy as np
import imutils
from collections import deque
import argparse
import math

def main():
    # hsv hue sat value
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
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
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
                cv2.circle(frame, (int(x_r), int(y_r)), int(radius_r),
                           (0, 255, 255), 2)
                cv2.circle(frame, center_r, 5, (0, 0, 255), -1)


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
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)


        if len(cnts_g) > 0:
            c_g = max(cnts_g, key=cv2.contourArea)
            ((x_g, y_g), radius_g) = cv2.minEnclosingCircle(c_g)
            M_g = cv2.moments(c_g)
            center_g = (int(M_g["m10"] / M_g["m00"]), int(M_g["m01"] / M_g["m00"]))

            if radius_g > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x_g), int(y_g)), int(radius_g),
                           (0, 255, 255), 2)
                cv2.circle(frame, center_g, 5, (0, 0, 255), -1)

        if len(cnts_b) > 0:
            c_b = max(cnts_b, key=cv2.contourArea)
            ((x_b, y_b), radius_b) = cv2.minEnclosingCircle(c_b)
            M_b = cv2.moments(c_b)
            center_b = (int(M_b["m10"] / M_b["m00"]), int(M_b["m01"] / M_b["m00"]))

            if radius_b > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x_b), int(y_b)), int(radius_b),
                           (0, 255, 255), 2)
                cv2.circle(frame, center_b, 5, (0, 0, 255), -1)
        # make triangle
        if len(cnts_r and cnts_g and cnts_b) > 0:
            if count == 0:
                ref_c7_to_aom = math.sqrt(((center_g[0] - center_b[0]) ** 2) + ((center_g[1] - center_b[1]) ** 2))
                ref_aom_to_chin = math.sqrt(((center_b[0] - center_r[0]) ** 2) + ((center_b[1] - center_r[1]) ** 2))
                ref_c7_to_chin = math.sqrt(((center_g[0] - center_r[0]) ** 2) + ((center_g[1] - center_r[1]) ** 2))
                ref_rad = math.acos((ref_aom_to_chin**2+ref_c7_to_aom**2-ref_c7_to_chin**2)/(2*ref_c7_to_aom*ref_aom_to_chin))
                count = count+1
            else:
                c7_to_aom = math.sqrt(((center_g[0] - center_b[0]) ** 2) + ((center_g[1] - center_b[1]) ** 2))
                aom_to_chin = math.sqrt(((center_b[0] - center_r[0]) ** 2) + ((center_b[1] - center_r[1]) ** 2))
                c7_to_chin = math.sqrt(((center_g[0] - center_r[0]) ** 2) + ((center_g[1] - center_r[1]) ** 2))
                new_rad = math.acos((aom_to_chin ** 2 + c7_to_aom ** 2 - c7_to_chin ** 2) / (2 * c7_to_aom * aom_to_chin))
        # calculate different angle
        try:
            rad = new_rad-ref_rad
            angle = math.degrees(rad)
        except:
            pass

        # show the frame to our screen
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask_red+mask_blue+mask_green)
        cv2.imshow('result', result)
        # if cv2.countNonZero(mask) > threshold:
        #     print('FOUND')
        #     break4
        # print(cv2.countNonZero(mask_red+mask_blue+mask_green))
        print(angle)
        #
        # Wait for escape key.
        #
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
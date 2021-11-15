#! /usr/bin/env python
from __future__ import print_function
from geometry_msgs.msg import Twist
import roslib
import cv2
import math
import roslaunch
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import time
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import imutils

rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
pub2 = rospy.Publisher('/license_plate', String, queue_size=1)
time.sleep(1)


class ControlTimer:

    def __init__(self):
        self.start_command = str('alexsean,cheese1,0,0000')
        self.end_command = str('alexsean,cheese1,-1,1111')

    def startTimer(self):
        pub2.publish(self.start_command)

    def endTimer(self):
        pub2.publish(self.end_command)

class LicensePlateDetector:
    def __init__(self):
        pass
    def isRectangle(self, c):
        peri = cv2.arcLength(c, True)
        # could just make it 10?
        approx = cv2.approxPolyDP(c, 0.04*peri, True)
        if len(approx)==4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if ar <= 1.05:
                return approx, True
        return None, False
    def isMyRectangle(self, points):
        # if 0<dst[1][0][0]-dst[0][0][0] <=35 and 190< dst[1][0][1]-dst[0][0][1] < 310:
        #         if 0 < dst[2][0][0] - dst[3][0][0] <= 35 and 190 < dst[2][0][1] - dst[3][0][1] < 310:
        topLefty = points[0][0][0]
        topLeftx = points[0][0][1]
        topRighty = points[1][0][0]
        topRightx = points[1][0][1]
        botLefty = points[3][0][0]
        botLeftx = points[3][0][1]
        botRighty = points[2][0][0]
        botRightx = points[2][0][1]
        topWidth = topRightx - topLeftx
        botWidth = botRightx - botLeftx
        leftHeight = topLefty - botRighty
        rightHeight = topRighty - botLefty

        if 0.75< topWidth/botWidth <1.25 and 0.75< rightHeight/leftHeight <1.25:
            return True



class ProcessImage:

    def __init__(self):
        self.bridge = CvBridge()
        self.timeout = 0

    def __callback(self, data):
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        grayframe = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = grayframe.shape
        cv_image_crop = cv_image[int(height/3):height, 30:width/2]

        img = cv2.imread( '/home/alexkneifel/Downloads/ThreshPlate.png', cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("image", img)
        blurred_feed = cv2.medianBlur(cv_image_crop, 5)

# Convert BGR to HSV
        hsv = cv2.cvtColor(blurred_feed, cv2.COLOR_BGR2HSV)
        homography = None

        uh = 0 #157
        us = 2#8
        uv = 215#168
        lh = 0#0
        ls = 0#0
        lv = 100#87
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

#homography

        sift = cv2.xfeatures2d.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(img, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        kp_grayframe, desc_grayframe = sift.detectAndCompute(mask, None)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        img3 = cv2.drawMatches(img, kp_image, mask, kp_grayframe, good_points, mask)

        cv2.imshow("matches of keypoints", img3)
        cv2.waitKey(3)


        if len(good_points) > 10:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            matrix, mask2 = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask2.ravel().tolist()

            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)


            lpd = LicensePlateDetector()
            if lpd.isMyRectangle(dst):
                homography = cv2.polylines(cv_image_crop, [np.int32(dst)], True, (255, 0, 0), 3)
                cv2.imshow("homography", homography)
                cv2.waitKey(3)
                license_plate_crop = cv_image_crop[int(dst[0][0][1]): int(dst[1][0][1]), int(dst[0][0][0]):int(dst[3][0][0])]
                cv2.imshow("license plate", license_plate_crop)
                cv2.waitKey(3)


    def process_image(self):
        listen = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.__callback)
        rospy.spin()

        return True



timer = ControlTimer()
timer.startTimer()
process_image = ProcessImage()
process_image.process_image()
#need to find a way to stop timer

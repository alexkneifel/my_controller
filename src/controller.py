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


bridge = CvBridge()

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
            # (x, y, w, h) = cv2.boundingRect(approx)
            # ar = w / float(h)
            # if ar <= 1.05:
            return approx, True
        return None, False


class ProcessImage:

    def __init__(self):
        self.bridge = CvBridge()
        self.timeout = 0

    def __callback(self, data):
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        grayframe = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("raw image", cv_image)
        # cv2.waitKey(3)
        height, width = grayframe.shape
        cv_image_crop = cv_image[int(height/3):height, 30:width/2]
        # cv2.imshow("Image crop", cv_image_crop)
        # cv2.waitKey(3)


        img = cv2.imread( '/home/alexkneifel/Downloads/ThreshPlate.png', cv2.IMREAD_GRAYSCALE)
#         cv2.imshow("image", img)
        blurred_feed = cv2.medianBlur(cv_image_crop, 5)
#
#         # Convert BGR to HSV
        hsv = cv2.cvtColor(blurred_feed, cv2.COLOR_BGR2HSV)
        homography = None
#
        uh = 0 #157
        us = 2#8
        uv = 215#168
        lh = 0#0
        ls = 0#0
        lv = 100#87
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        # Threshold the HSV image to get only blue colors
        # could use HSV with the key feature detection
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
#         # window_name = "HSV Calibrator"
#         # cv2.namedWindow(window_name)
#         #
#         # def nothing(x):
#         #     print("Trackbar value: " + str(x))
#         #     pass
#         #
#         # # create trackbars for Upper HSV
#         # cv2.createTrackbar('UpperH', window_name, 0, 255, nothing)
#         # cv2.setTrackbarPos('UpperH', window_name, uh)
#         #
#         # cv2.createTrackbar('UpperS', window_name, 0, 255, nothing)
#         # cv2.setTrackbarPos('UpperS', window_name, us)
#         #
#         # cv2.createTrackbar('UpperV', window_name, 0, 255, nothing)
#         # cv2.setTrackbarPos('UpperV', window_name, uv)
#         #
#         # # create trackbars for Lower HSV
#         # cv2.createTrackbar('LowerH', window_name, 0, 255, nothing)
#         # cv2.setTrackbarPos('LowerH', window_name, lh)
#         #
#         # cv2.createTrackbar('LowerS', window_name, 0, 255, nothing)
#         # cv2.setTrackbarPos('LowerS', window_name, ls)
#         #
#         # cv2.createTrackbar('LowerV', window_name, 0, 255, nothing)
#         # cv2.setTrackbarPos('LowerV', window_name, lv)
#         #
#         # font = cv2.FONT_HERSHEY_SIMPLEX
#         #
#         # print("Loaded images")
#         #
#         # while (1):
#         #     # Threshold the HSV image to get only blue colors
#         #     mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
#         #     cv2.putText(mask, 'Lower HSV: [' + str(lh) + ',' + str(ls) + ',' + str(lv) + ']', (10, 30), font, 0.5,
#         #                (200, 255, 155), 1, cv2.LINE_AA)
#         #     cv2.putText(mask, 'Upper HSV: [' + str(uh) + ',' + str(us) + ',' + str(uv) + ']', (10, 60), font, 0.5,
#         #                (200, 255, 155), 1, cv2.LINE_AA)
#         #
#         #     cv2.imshow(window_name, mask)
#         #
#         #     k = cv2.waitKey(1) & 0xFF
#         #     if k == 27:
#         #         break
#         #     # get current positions of Upper HSV trackbars
#         #     uh = cv2.getTrackbarPos('UpperH', window_name)
#         #     us = cv2.getTrackbarPos('UpperS', window_name)
#         #     uv = cv2.getTrackbarPos('UpperV', window_name)
#         #     upper_blue = np.array([uh, us, uv])
#         #     # get current positions of Lower HSCV trackbars
#         #     lh = cv2.getTrackbarPos('LowerH', window_name)
#         #     ls = cv2.getTrackbarPos('LowerS', window_name)
#         #     lv = cv2.getTrackbarPos('LowerV', window_name)
#         #     upper_hsv = np.array([uh, us, uv])
#         #     lower_hsv = np.array([lh, ls, lv])
#         #
#         #     time.sleep(.1)
#         #
#         # cv2.destroyAllWindows()
#
#
# # rectangle
# #         contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #         contours = imutils.grab_contours(contours)
# #         contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
# #         lpd = LicensePlateDetector()
# #         location = None
# #         for contour in contours:
# #             approx, rectangle = lpd.isRectangle(contour)
# #             if rectangle:
# #                 location = approx
# #                 break
# #         print(location)
#
#         # cv2.imshow("hsv window", mask)
#         # cv2.waitKey(3)
#
#
# #homography

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

            homography = cv2.polylines(cv_image_crop, [np.int32(dst)], True, (255, 0, 0), 3)

        if homography is not None:
            cv2.imshow("homography", homography)
            cv2.waitKey(3)



    def process_image(self):
        listen = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.__callback)
        rospy.spin()
        # use the following if you want to filter image in some way, probably should go in callback
        # threshold = 150
        # data = None
        # while data is None:
        #     try:
        #         data = rospy.wait_for_message('/R1/pi_camera/image_raw', Image,
        #                                       timeout=5)
        #     except:
        #         pass
        # try:
        #     cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # except CvBridgeError as e:
        #     print(e)
        #
        # # cv2.imshow("raw", cv_image)
        # # could play with threshold worked for lab 3
        #
        # gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        # _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
        #
        # cv2.imshow("gray image", gray_image)
        # cv2.waitKey(3)
        # cv2.imshow("binary image", binary_image)
        # cv2.waitKey(3)
        #
        # rospy.spin()
        # where should I put rospy.spin() ?

        return True




timer = ControlTimer()
timer.startTimer()
process_image = ProcessImage()
process_image.process_image()

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


class ProcessImage:

    def __init__(self):
        self.bridge = CvBridge()
        self.timeout = 0

    def __callback(self, data):
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv2.imshow("raw image", cv_image)
        cv2.waitKey(3)

        img = cv2.imread( '/home/alexkneifel/Downloads/CroppedPlate.png', cv2.IMREAD_GRAYSCALE)
        sift = cv2.xfeatures2d.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(img, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)


        grayframe = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)

        cv2.imshow("matches of keypoints", img3)


        if len(good_points) > 10:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(cv_image, [np.int32(dst)], True, (255, 0, 0), 3)

        cv2.imshow("homography", homography)



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

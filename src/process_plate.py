#! /usr/bin/env python
from __future__ import print_function
from geometry_msgs.msg import Twist
import roslib
import cv2 as cv
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


class ProcessPlate:

    def __init__(self):
        self.s = None
        self.v = None
        self.setVal = False
        self.max_s = 0
        self.max_v = 0

# check ratio of width to height, check area, check colors within area
    def isLicensePlate(self, crop, left, right, top , bottom):
        small_width = right - left
        small_height = bottom - top
        if 2.8 < float(small_width)/small_height < 3.9:
            if 30 < small_width < 70 and 10 < small_height <= 20:
                return True
        else:
            return False

#
# # perhaps could
    def __normalize_img(self, img):
        max_area = 0
        foundPlate = False
        max_c = None

        cv.imshow("original image ", img)
        cv.waitKey(1)

        rx = 400.0 / img.shape[1]
        dim = (400, int(img.shape[0] * rx))
        # perform the actual resizing of the image
        resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        height, width, channels = resized_img.shape
        ry = height / float(img.shape[0])
        # crop this 25 pixels lower, would be easier if
        cv_image_crop = resized_img[int(height/1.7):height, 0:width]

        hsv = cv.cvtColor(cv_image_crop, cv.COLOR_BGR2HSV)  # Convert to hsv color system
        h, s, v = cv.split(hsv)
        result_v = cv.equalizeHist(v)
        result_s = cv.equalizeHist(s)
        #
        hsv = cv.merge((h, result_s, result_v))
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        #
        # # #
        # could also try a different blur
        # cv2.GaussianBlur(gray, (5, 5), 0)
        blur = cv.medianBlur(rgb,5)
        # cv.imshow("normalized s and v", blur)
        # cv.waitKey(1)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        # homography = None

        uh = 23 #157
        us = 255#8
        uv = 255#168
        lh = 1#0
        ls = 0#0
        lv = 0#87
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        #kernel2 = np.ones((2, 2), np.uint8)
        kernel1 = np.ones((3,3), np.uint8)
        # could also do if hsv is above certain threshold then dilate

        # could increase # iterations or kernel size. could do two iterations of each
        #img_erosion = cv.erode(mask, kernel2, iterations=1)
        img_dilation = cv.dilate(mask, kernel1, iterations=2)
        # if dilation is above certain threshold then do the rest of this
        cv.imshow("Dilation", img_dilation)
        cv.waitKey(1)
        print(sum(sum(img_dilation)))


        contours = cv.findContours(img_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        # area_thresh = 0
        result = resized_img.copy()
        for c in contours:
            area = cv.contourArea(c)
            #print("contour" + str(c))
            if area > max_area:
                max_c = c
                max_area = area
                #print("max area"+ str(max_area))
                # need to get the points from the contour and check if it is a rectangle w my function
                # need to make sure I don't get the blob up top, could check if perimeter is closed

# local variable max_c reference before assignment
        if max_c is not None:
            leftmost = tuple(max_c[max_c[:, :, 0].argmin()][0])
            rightmost = tuple(max_c[max_c[:, :, 0].argmax()][0])
            top = tuple(max_c[max_c[:, :, 1].argmin()][0])
            bot = tuple(max_c[max_c[:, :, 1].argmax()][0])
            # i want width in a certain ratio to height
            small_leftp = int(leftmost[0])
            small_rightp = int(rightmost[0])
            small_topp = int(top[1])
            small_botp = int(bot[1])

            test_crop = cv_image_crop[small_topp:small_botp, small_leftp:small_rightp]

            cv.imshow("small plate", test_crop)
            cv.waitKey(1)

            if self.isLicensePlate(test_crop, small_leftp, small_rightp, small_topp, small_botp):
                leftp = int(small_leftp/rx)
                rightp = int(small_rightp / rx)
                botp = int(small_botp / ry) + int(img.shape[0]/1.7)
                topp = int(small_topp / ry) + int(img.shape[0]/1.7)
                license_plate_crop = img[topp:botp,leftp:rightp]
                cv.imshow("license plate", license_plate_crop)
                cv.waitKey(1)
                foundPlate = True


        # should stop the car from moving if it detects a rectangle of right area
        #license_plate_crop = cv_image_crop[int(top[1]): int(bot[1]), int(leftmost[0]):int(rightmost[0])]



        #cv.drawContours(result, [max_c], -1, (0, 0, 255), 1)



    def proccessPlate(self, cv_image):
        self.__normalize_img(cv_image)

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
import os



class ProcessPlate:

    def __init__(self):
        self.plate_search = True

    def isLicensePlate(self, left, right, top , bottom):
        small_width = right - left
        small_height = bottom - top
        if 2.8 < float(small_width)/small_height < 3.9:
            if 30 < small_width < 70 and 10 < small_height <= 20:
                return True
        else:
            return False

    def __normalize_img(self, img, string):
        max_area = 0
        max_c = None

        rx = 400.0 / img.shape[1]
        dim = (400, int(img.shape[0] * rx))
        resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        height, width, channels = resized_img.shape
        ry = height / float(img.shape[0])

        cv_image_crop = resized_img[int(height/1.55):height, 0:width/2]

        hsv = cv.cvtColor(cv_image_crop, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        result_v = cv.equalizeHist(v)
        result_s = cv.equalizeHist(s)

        hsv = cv.merge((h, result_s, result_v))
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        #could try a different blur
        blur = cv.medianBlur(rgb,5)

        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

        uh = 23
        us = 255
        uv = 255
        lh = 1
        ls = 0
        lv = 0
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        kernel1 = np.ones((3,3), np.uint8)
        #kernel2 = np.ones((1, 1), np.uint8)
        # could also do if hsv is above certain threshold then dilate to reduce processing

        # could increase # iterations or kernel size. could do two iterations of each
        #img_erosion = cv.erode(mask, kernel2, iterations=1)
        img_dilation = cv.dilate(mask, kernel1, iterations=1)
        cv.imshow("Dilation", img_dilation)
        cv.waitKey(1)
        dilation_sum = sum(sum(img_dilation))
        print(dilation_sum)

# reset to search for plates once we are past the parked vehicle
        if dilation_sum == 0:
            self.plate_search = True

        # if dilation is above certain threshold then do the rest of this
        if self.plate_search == True:
            if dilation_sum > 11500:
                contours = cv.findContours(img_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                for c in contours:
                    area = cv.contourArea(c)
                    if area > max_area:
                        max_c = c
                        max_area = area

                if max_c is not None:
                    leftmost = tuple(max_c[max_c[:, :, 0].argmin()][0])
                    rightmost = tuple(max_c[max_c[:, :, 0].argmax()][0])
                    top = tuple(max_c[max_c[:, :, 1].argmin()][0])
                    bot = tuple(max_c[max_c[:, :, 1].argmax()][0])
                    small_leftp = int(leftmost[0])
                    small_rightp = int(rightmost[0])
                    small_topp = int(top[1])
                    small_botp = int(bot[1])

                    test_crop = cv_image_crop[small_topp:small_botp, small_leftp:small_rightp]

                    cv.imshow("small plate", test_crop)
                    cv.waitKey(1)

                    if self.isLicensePlate(small_leftp, small_rightp, small_topp, small_botp):
                        leftp = int(small_leftp/rx)
                        rightp = int(small_rightp / rx)
                        botp = int(small_botp / ry) + int(img.shape[0]/1.55)
                        topp = int(small_topp / ry) + int(img.shape[0]/1.55)
                        license_plate_crop = img[topp:botp,leftp:rightp]
                        cv.imshow(""+string, license_plate_crop)
                        cv.waitKey(1)
                        self.plate_search = False
                        return license_plate_crop, True
                    else:
                        # this would be if the man is in the middle of the road. no plate, but it was above threshold
                        # if we have dilation sum above, but it is not a plate. then it is the man.
                        # dont return a plate, but return False that we cant move forward
                        return None, False
            # if we dont have dilation sum above a certain value, then return no plate, but true that we can move forward
            # not sure if this is necessarily true
        return None, True

    def proccessPlate(self, cv_image, string):
        plate, canMove = self.__normalize_img(cv_image, string)
        return plate, canMove

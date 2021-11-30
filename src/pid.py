#!/usr/bin/env python

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

# let states be, 1 be a corner turn, 0 be straight

class PidCtrl:
    def __init__(self):
        self.lastState = 0
        self.desiredVal = 29600
        self.lastTurnSpeed = 0

    def __countFrontZeros(self, array):
        front_zeros=0
        for val in array:
                if val != 0:
                    break
                elif val == 0:
                    front_zeros += 1
        return np.zeros(front_zeros)


    def __countBackZeros(self, w_zeros, no_leading_zeros, front_zero):
        return int(w_zeros - front_zero - no_leading_zeros)

    def __computeAverage(self,section1,section2,section3):
        return (section1+section2+section3)/3

#TODO
# could recalculate value for rhs being where I want it
# could process smaller image to make PID less intensive
# could make right hand turns a part of original control instead of its own control
# could not use there being 0's for turning and just use the P
    def nextMove(self, cv_image):
        fwdSpeed = 0.2
        turnSpeed = 0.1
        min_index = 5
        grayframe = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)

        ten_images = []
        count = [0] * 10
        index = 0

        img = cv_image
        img = cv.medianBlur(img, 5)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        uh = 100
        us = 255
        uv = 255
        lh = 0
        ls = 0
        lv = 239
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        mask = cv.inRange(hsv, lower_hsv, upper_hsv)

        height, width = grayframe.shape
        slicewidth = width // 10

        for i in range(10):
            ten_images.append(mask[:, i * slicewidth: (i + 1) * slicewidth])

        for image in ten_images:
            count[index] = sum(sum(image))
            index += 1

        sizeCount = len(count)
        short_count = np.trim_zeros(count)
        frontZeros = len(self.__countFrontZeros(count))
        if len(short_count) > 0 :
            min_index = np.argmin(short_count) + frontZeros
        backZeros = self.__countBackZeros(sizeCount,len(short_count),frontZeros)

        # corner turn, if RHS of screen is black turn, or if last state was turning turn,
        if frontZeros ==0 and backZeros > 1 or self.lastState ==1:
            if min_index == 4 or min_index == 5:
                fwdSpeed = 0.15
                turnSpeed = 0
                self.lastState = 0
            else:
                fwdSpeed = 0.1
                turnSpeed = 3
                self.lastState = 1

# if not turning, do normal PID for straight road adjustments
        else:
            # maybe could not take abs then wouldn't have to use backZeros!! to check if left or right turn
            difference = abs(self.desiredVal - self.__computeAverage(count[6],count[7],count[8]))

            # for crosswalk to not be swerving
            if count[6] and count[7] < 26000:
                left_p = 0.00005
                right_p = 0.00005

            else:
                left_p = 0.0003
                right_p = 0.0003
            fwdSpeed = 0.15
            if backZeros > 0:
                turnSpeed = difference*left_p
                if turnSpeed > 3:
                    turnSpeed = 3
            else:
                turnSpeed = -difference*right_p
                if abs(turnSpeed) > 3:
                    turnSpeed = -3
            self.lastState = 0

        self.lastTurnSpeed = turnSpeed

        return fwdSpeed,turnSpeed, self.lastState

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

# let 1 be a corner turn, 0 be straight

class PidCtrl:
    def __init__(self):
        self.lastState = 0
        self.desiredVal = 29600
        self.lastTurnSpeed = 0
        pass

    # need to finish these two functions and make sure they work
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

        # Convert BGR to HSV
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


        # cv.imshow("hsv", mask)
        # cv.waitKey(1)

        # max_count = np.amax(count)
        # print("COUNT " + str(count))
        # max_index = np.argmax(count)
        # print("MAX INDEX " + str(max_index))
        # num_nonzero = np.count_nonzero(count)
        # print("NUM NONZERO" + str(num_nonzero))

        sizeCount = len(count)
        short_count = np.trim_zeros(count)
        frontZeros = len(self.__countFrontZeros(count))
        if len(short_count) > 0 :
            min_index = np.argmin(short_count) + frontZeros
        backZeros = self.__countBackZeros(sizeCount,len(short_count),frontZeros)
        #print("count " + str(count))

        if frontZeros ==0 and backZeros > 1 or self.lastState ==1:
            #could be while still have 0's turn but hard if this isnt sending command
            if min_index == 4 or min_index == 5:
                fwdSpeed = 0.3
                turnSpeed = -0.05
                self.lastState = 0
            else:
                fwdSpeed = 0.1
                turnSpeed = 3
                self.lastState = 1


        else:
            difference = abs(self.desiredVal - self.__computeAverage(count[6],count[7],count[8]))
            if count[6] and count[7] < 26000:
                left_p = 0.00005
                right_p = 0.00005
            else:
                left_p = 0.0003
                right_p = 0.0003
            fwdSpeed = 0.33
            # this forces it to be awfully on the line on the right , maybe say if most right hand one is below a certain val?
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
        # print("forward: " + str(fwdSpeed))
        # print("turn: " + str(turnSpeed))
        return fwdSpeed,turnSpeed

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
        threshold = 100
        min_index = 5
        grayframe = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)


        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False
        ten_images = []
        count = [0] * 10
        index = 0
        # print("Width", width)

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

        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)

        height, width = grayframe.shape
        slicewidth = width // 10

        # #size of 800
        for i in range(10):
            ten_images.append(mask[:, i * slicewidth: (i + 1) * slicewidth])
        #
        # # scroll through ten images
        # #print("ten images length", len(ten_images))
        for image in ten_images:
            count[index] = sum(sum(image))
            index += 1


        cv.imshow("hsv", mask)
        cv.waitKey(1)

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
        # print("Index of min: " + str(min_index))
        # print("num of back zeros" + str(backZeros))
        print("count " + str(count))

# go straight unless frontZeros=0 and backZeros>0, then turn until minIndex is 4 or 5
        #print("sum of count: " + str(sum(count)))
        if frontZeros ==0 and backZeros > 1 or self.lastState ==1:
            if min_index == 4 or min_index == 5:
                fwdSpeed = 0.3
                turnSpeed = 0
                self.lastState = 0
            else:
                fwdSpeed = 0
                turnSpeed = 3
                self.lastState = 1

# or maybe middle val
#         elif count[4] > 30000 and count[5] > 30000:
#             fwdSpeed = 0.3
#             turnSpeed = 0

        else:
            scaledAmt = 0.0007
            fwdSpeed = 0.3
            self.lastState = 0
            difference = abs(self.desiredVal - self.__computeAverage(count[6],count[7],count[8]))
            # this forces it to be awfully on the line on the right , maybe say if most right hand one is below a certain val?
            if backZeros > 0:
                turnSpeed = difference*scaledAmt
                if turnSpeed > 5:
                    turnSpeed = 5
            else:
                turnSpeed = -difference*scaledAmt
                if abs(turnSpeed) > 5:
                    turnSpeed = -5
            #if backZeros>0 then turn left scaled amt by difference
            # otherwise turn right a scaled amount



        print("forward: " + str(fwdSpeed))
        print("turn: " + str(turnSpeed))
        return fwdSpeed,turnSpeed
        # need to deal with count having all 0's
        # need to deal with all values being non zero (could be straight, but could be turn)
        # need to deal with hav multiple 0's in the middle

        # for the turns can define a minimum amt of white to count, otherwise go straight
        # if even distribution of white above certain threshold turn
        # if even distribution of a higher threshold go straight
        # find first minimum value between first and last nonzeroVal
        # could cut out the left side a bit to not have black






#
#
# listen = rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback)
#
# rospy.spin()

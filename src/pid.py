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
        if frontZeros ==0 and backZeros > 0 or self.lastState ==1:
            if min_index == 4 or min_index == 5:
                fwdSpeed = 0.3
                turnSpeed = 0
                self.lastState = 0
            else:
                fwdSpeed = 0
                turnSpeed = 3
                self.lastState = 1
        else:
            fwdSpeed = 0.3
            turnSpeed = 0
            self.lastState = 0

        # slowTurnFwd = 0
        # sharpTurn = 3
        # mediumTurn = 2
        # slowTurn = 1
        # turn = 2
        # #check zeros first
        # if np.count_nonzero(count) == 0:
        #     fwdSpeed = 0
        #     turnSpeed = 6
        #
        # elif backZeros > 0:
        #     fwdSpeed = slowTurnFwd
        #     turnSpeed = turn * backZeros
        # elif frontZeros > 0:
        #     if count[4] <20000 or count[5] < 20000:
        #         fwdSpeed = 0.3
        #         turnSpeed = 0
        #     else:
        #         fwdSpeed = slowTurnFwd
        #         turnSpeed = -turn * frontZeros
        #
        # # then check indicese
        # # elif count[min_index] > 11000 :
        # #     fwdSpeed = 0.3
        # #     turnSpeed = 0
        # elif 2 < min_index <= 6:
        #     turnSpeed = 0
        #     if min_index == 3:
        #         fwdSpeed = 0.3
        #     elif min_index == 4 or min_index ==5:
        #         fwdSpeed = 0.45
        #     elif min_index == 6:
        #         fwdSpeed = 0.3
        # elif 6 < min_index:
        #     if min_index == 7:
        #         fwdSpeed = slowTurnFwd
        #         turnSpeed = sharpTurn
        #     elif min_index == 8:
        #         fwdSpeed = slowTurnFwd
        #         turnSpeed = mediumTurn
        #     elif min_index == 9:
        #         fwdSpeed = slowTurnFwd
        #         turnSpeed = slowTurn
        # else:
        #     if count[4] <25000 or count[5] < 25000:
        #         fwdSpeed = 0.3
        #         turnSpeed = 0
        #     else:
        #         if min_index == 0:
        #                 fwdSpeed = 0.2
        #                 turnSpeed = -slowTurn*0.5
        #         elif min_index == 1:
        #                 fwdSpeed = 0.2
        #                 turnSpeed = -slowTurn*0.5
        #         elif min_index == 2:
        #                 fwdSpeed = 0.2
        #                 turnSpeed = -slowTurn*0.5





        # if 0 <= max_index <6:
        #     if num_nonzero == 6:
        #         fwdSpeed = 0.3
        #         turnSpeed = 0
        #     elif num_nonzero == 4:
        #         fwdSpeed = 0.1
        #         turnSpeed = 3
        #     else:
        #         fwdSpeed = 0.2
        #         turnSpeed = 1
        # else:
        #     fwdSpeed = 0.1
        #     turnSpeed = -0.3

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

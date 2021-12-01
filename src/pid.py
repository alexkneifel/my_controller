#!/usr/bin/env python

from __future__ import print_function
import cv2 as cv
import numpy as np
import time


# let states be, 1 be a corner turn, 0 be straight

class PidCtrl:
    def __init__(self):
        self.lastState = 0
        self.desiredVal = 29600
        self.stopped = False
        self.already_stopped = False
        self.no_ped_count = 0

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
        fwdSpeed = 0
        turnSpeed = 0

        # if at crosswalk and waiting
        if self.stopped is True:
            height,width,channels = cv_image.shape
            cv_image_crop = cv_image[int(height /1.8):height, width/3:width/2]
            blur = cv.medianBlur(cv_image_crop, 5)
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

            kernel1 = np.ones((4, 4), np.uint8)
            img_dilation = cv.dilate(mask, kernel1, iterations=1)
            dilation_sum = sum(sum(img_dilation))
            print("Person Sum: " + str(dilation_sum))
            fwdSpeed = 0
            turnSpeed = 0
            self.lastState = 0
            if self.already_stopped and self.no_ped_count > 75:
                self.stopped = False
                print("go")
                self.no_ped_count = 0
            elif self.already_stopped and dilation_sum < 2000:
                print("waiting")
                self.no_ped_count +=1
                print("no ped " + str(self.no_ped_count))
            else:
                time.sleep(0.2)
                self.stopped = False
                print("go")
                self.no_ped_count = 0

        # if not waiting at crosswalk
        if self.stopped is False:
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

            if frontZeros ==0 and backZeros > 1 or self.lastState ==1:
                if min_index == 4 or min_index == 5:
                    fwdSpeed = 0.05
                    turnSpeed = 0
                    self.lastState = 0
                else:
                    fwdSpeed = 0.01
                    turnSpeed = 1
                    self.lastState = 1

            else:
                difference = abs(self.desiredVal - self.__computeAverage(count[6],count[7],count[8]))

#TODO maybe raise 24500 is stops at crosswalk randomly again, did it again, i think lower it?
                if count[6] and count[7] < 24000 and self.already_stopped is False:
                    left_p = 0
                    right_p= 0
                    fwdSpeed =0
                    self.already_stopped = True
                    self.stopped = True
                    # was 26000
                elif count[6] and count[7] < 27000 and self.already_stopped:
                    left_p = 0.00005
                    right_p = 0.00005
                    fwdSpeed = 0.15

                else:
                    left_p = 0.00025
                    right_p = 0.00025
                    fwdSpeed = 0.15
                    self.already_stopped = False
                if backZeros > 0:
                    turnSpeed = difference*left_p
                    if turnSpeed > 3:
                        turnSpeed = 3
                else:
                    turnSpeed = -difference*right_p
                    if abs(turnSpeed) > 3:
                        turnSpeed = -3
                self.lastState = 0

        return fwdSpeed,turnSpeed, self.lastState

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


class ProcessPlate:

    def __init__(self):
       pass
    def proccessPlate(self, plate):
       # rectangle
     # need to find all rectangles in the image and see what it gives me
        #print(location)


        # corners = cv2.goodFeaturesToTrack(plate, 27, 0.01, 10)
        # corners = np.int0(corners)

        # we iterate through each corner,
        # making a circle at each point that we think is a corner.
        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(img, (x, y), 3, 255, -1)
        #
       #ANPR tutorial
        cv2.imshow("plate", plate)
        bfilter = cv2.bilateralFilter(plate, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None

        mask = np.zeros(plate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("filtered plate", new_image)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        while True:
            k = cv2.waitKey(0) & 0xFF
            print(k)
            #space bar
            if k == 32:
                cv2.destroyAllWindows()
                break




plateProcessor = ProcessPlate()
img = cv2.imread( '/home/alexkneifel/Pictures/LicensePlate_from_homog.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("plate", img)
plateProcessor.proccessPlate(img)
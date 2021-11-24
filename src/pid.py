#!/usr/bin/env python

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


class PidCtrl:
    def __init__(self):
        pass
    def nextMove(self, cv_image):
        print("succesfully called nextMove")
        # threshold = 150
        #
        # state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # done = False
        # ten_images = []
        # count = [0] * 10
        # index = 0
        # # print("Width", width)
        #
        # gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        # _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
        #
        # height, width = gray_image.shape
        # slicewidth = width // 10
        #
        # # #size of 800
        # for i in range(10):
        #     ten_images.append(binary_image[:, i * slicewidth: (i + 1) * slicewidth])
        # #
        # # # scroll through ten images
        # # #print("ten images length", len(ten_images))
        # for image in ten_images:
        #     count[index] = sum(sum(image))
        #     index += 1
        #
        # cv2.imshow("gray image", gray_image)
        # cv2.waitKey(3)
        # cv2.imshow("binary image", binary_image)
        # cv2.waitKey(3)
        #
        # # max_count = np.amax(count)
        # # print(count)
        # # if max_count == 0:
        # #     self.timeout += 1
        # #     if self.timeout > 30:
        # #         done = True
        # # else:
        # #     max_index = np.argmax(count)
        # #     print(max_index)
        # #     state[max_index] = 1
        # #     self.timeout = 0


#
#
# listen = rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback)
#
# rospy.spin()

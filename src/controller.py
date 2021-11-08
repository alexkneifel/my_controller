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

    def __callback(self,data):
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv2.imshow("raw image", cv_image)
        cv2.waitKey(3)

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



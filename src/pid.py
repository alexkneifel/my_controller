#!/usr/bin/env python

from __future__ import print_function
import rospy
from geometry_msgs.msg import Twist
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import colorsys
import math

bridge = CvBridge()

#rospy.init_node('topic_publisher')

rospy.init_node('topic_subscriber')

def callback(data, lastTurn=None):
	cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
	firstValLoc = 0
	lastValLoc = 0
	desiredMiddle = 400
	firstValFound = False
	lastValFound = False
	cv2.imshow("camera_output", cv_image)
	print(cv_image)
	# gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
	# croppedframe = gray[799:800, 0:800]


listen = rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback)

rospy.spin()

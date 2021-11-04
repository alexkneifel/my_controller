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

rospy.init_node('topic_publisher')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

def middleFinder(subframe,len):
  	b=0
  	for x in range(0,len):
  		if x==0:
  			start = False
  			a=0
  			b=0
		if (subframe[0][x]<100 and start == False) or (subframe[0][x]<100 and x == 0):
			start=True
			a = x
    	if subframe[0][x]>100 and start == True:
      		start=False
      		b = x
  	return a,b

bridge = CvBridge()
dim = 800
ideal = 800/2
thresh = 100

def callback(data):
	# rate = rospy.Rate(2)
	cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
	gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
	subframe = gray[799:dim, 0:dim]
	a,b = middleFinder(subframe,dim)
	if a is not None and b is not None:
		# middle = a+(b-a)/2
		middle = a
	print(str(a) + " " + str(b))
	move = Twist()
	if middle==0:
		move.angular.z = 2
	else:
		move.angular.z = -(middle-ideal)/40
		move.linear.x=0.5

	# move.linear.x = 0.5
	pub.publish(move)
	# rate.sleep()

listen = rospy.Subscriber('/rrbot/camera1/image_raw', Image, callback)

rospy.spin()

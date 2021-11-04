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
from cv_bridge import CvBridge
bridge = CvBridge()

start_command = str('alexsean,cheese1,0,0000')
end_command = str('alexsean,cheese1,-1,1111')

rospy.init_node('publisher', anonymous=True)
pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
pub2 = rospy.Publisher('/license_plate', String, queue_size=1)
time.sleep(1)
#how to have two init nodes in the same place


def startTimer():
    pub2.publish(start_command)

def moveCommand():
    #move
    for i in range(1000):
        move = Twist()
        move.linear.x = i
        pub.publish(move)

def endTimer():
    #end timer
    pub2.publish(end_command)
    print("cheese hole")

startTimer()
moveCommand()
endTimer()

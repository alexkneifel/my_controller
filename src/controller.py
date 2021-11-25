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
import pid, process_plate

bridge = CvBridge()

rospy.init_node('publisher', anonymous=True)
pub2 = rospy.Publisher('/license_plate', String, queue_size=1)
pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
time.sleep(1)

#left all files that need to contact with Nodes in here

class ControlTimer:

    def __init__(self):
        self.start_command = str('alexsean,cheese1,0,0000')
        self.end_command = str('alexsean,cheese1,-1,1111')

    def startTimer(self):
        pub2.publish(self.start_command)

    def endTimer(self):
        pub2.publish(self.end_command)


class gazeboClock:
    def __init__(self):
        self.clock =rospy.Subscriber('/clock', String)
    def getTime(self):
        return rospy.get_time()

class moveBot:
    def __init__(self):
        self.move = Twist()
    def moveForward(self,forwdVal,turnVal):
        self.move.linear.x = forwdVal
        self.move.angular.z = turnVal
        pub.publish(self.move)

class ControlLoop:

    def __init__(self):
        self.bridge = CvBridge()
        self.timeout = 0
        self.clock = gazeboClock()
        self.timer = ControlTimer()
        self.timer.startTimer()
        self.startTime = self.clock.getTime()
        self.stopped = False
        self.moveBot = moveBot()
        self.pid = pid.PidCtrl()
        self.processPlate = process_plate.ProcessPlate()

    def start_control(self):
        listen = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.__callback)
        rospy.spin()

        return True

    def __callback(self, data):
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        currentTime = self.clock.getTime()
        if currentTime - self.startTime > 10 and self.stopped is not True:
            self.moveBot.moveForward(0,0)
            self.timer.endTimer()
            self.stopped = True
        else:
            if currentTime - self.startTime < 1:
                print("hi")
                #self.moveBot.moveForward(0.3, 0.7)
                #self.pid.nextMove(cv_image)
                #self.processPlate.proccessPlate(cv_image)
                # it running to this once self stopped is true
            else:
                #self.firstMove.moveForward(0.2, 0.0)
                #self.processPlate.proccessPlate(cv_image)
                fwdVal, turnVal = self.pid.nextMove(cv_image)
                self.moveBot.moveForward(fwdVal, turnVal)


control_loop = ControlLoop()
control_loop.start_control()

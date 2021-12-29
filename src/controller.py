#! /usr/bin/env python
from __future__ import print_function
from geometry_msgs.msg import Twist
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import time
from cv_bridge import CvBridge, CvBridgeError

import innerloop
import pid, process_plate, neuralnet

bridge = CvBridge()

rospy.init_node('my_publisher', anonymous=True)
pub2 = rospy.Publisher('/license_plate', String, queue_size=1)
pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
time.sleep(1)

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
        pass
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
        self.clock = gazeboClock()
        self.timer = ControlTimer()
        self.startTime = self.clock.getTime()
        self.stopped = False
        self.pid = pid.PidCtrl()
        self.processPlate = process_plate.ProcessPlate()
        self.neuralnet = neuralnet.NeuralNet()
        self.innerLoop = innerloop.InnerLoop()
        #TODO set back to 0
        self.count = 0
        self.currentTime = 0
        self.timer.startTimer()
        self.lastState = 0
        self.moveBot = moveBot()
        self.controlbot = False
        self.fixedTime = 0
        self.fixedStarted=False
        self.secondLoop = False

    def start_control(self):
        listen = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.__callback)
        rospy.spin()

    def __callback(self, data):
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.currentTime = self.clock.getTime()

        # first move is all messed up it thinks there is a pedestrian there
        if self.currentTime - self.startTime > 230 and self.stopped is not True or self.count == 8:
            self.moveBot.moveForward(0,0)
            self.timer.endTimer()
            self.stopped = True
        elif self.count >=6:
            if self.fixedStarted is False:
                self.fixedTime = self.currentTime
                self.fixedStarted = True
            elif self.secondLoop is False and self.currentTime - self.fixedTime < 5.5:
                fwdVal, turnVal, self.lastState = self.pid.nextMove(cv_image)
                self.moveBot.moveForward(fwdVal, turnVal)
            elif self.secondLoop is False and 7.75 >self.currentTime - self.fixedTime > 6:
                self.moveBot.moveForward(0, 1)
            elif self.secondLoop is False and 8 < self.currentTime - self.fixedTime:
                self.secondLoop = True
                self.moveBot.moveForward(0, 0)
            if self.secondLoop is True:
                if 8 < self.currentTime - self.fixedTime < 10.5:
                    self.moveBot.moveForward(0.25, 0)
                elif 10.5 < self.currentTime - self.fixedTime < 12:
                    self.moveBot.moveForward(0, 1)
                else:
                # what is the error on viewWorld?
                    #self.innerLoop.viewWorld(cv_image)
                    plate1, canMove = self.processPlate.proccessPlate(cv_image,self.secondLoop)
                    if plate1 is not None:
                        self.moveBot.moveForward(0, 0)
                        print("plate to NN")
                        self.count += 1
                        message = self.neuralnet.licencePlateToString(plate1)
                        pub2.publish(message)
                    else:
                        fwdVal,turnVal = self.innerLoop.viewWorld(cv_image)
                        #print(str(fwdVal) + " 0,0,0,0, " + str(turnVal))
                        self.moveBot.moveForward(fwdVal, turnVal)
        elif self.stopped is True:
            self.moveBot.moveForward(0, 0)
        else:
            # need to fix this beginning move
            if self.currentTime - self.startTime < 9.5:
                if self.currentTime - self.startTime < 8:
                    self.moveBot.moveForward(0.15, 0)
                else:
                    self.moveBot.moveForward(0, 1)
                self.lastState = 1
                print("hi")
            else:
                if self.lastState ==0:
                    plate1, canMove = self.processPlate.proccessPlate(cv_image,self.secondLoop)
                    if plate1 is not None:
                        self.moveBot.moveForward(0,0)
                        time.sleep(2)
                        print("plate to NN")
                        self.count += 1
                        message = self.neuralnet.licencePlateToString(plate1)
                        pub2.publish(message)

                    elif canMove is True:
                        fwdVal, turnVal, self.lastState = self.pid.nextMove(cv_image)
                        self.moveBot.moveForward(fwdVal, turnVal)
                else:
                    fwdVal, turnVal, self.lastState = self.pid.nextMove(cv_image)
                    self.moveBot.moveForward(fwdVal, turnVal)


control_loop = ControlLoop()
control_loop.start_control()

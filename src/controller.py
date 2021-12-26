#! /usr/bin/env python
from __future__ import print_function
from geometry_msgs.msg import Twist
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import time
from cv_bridge import CvBridge, CvBridgeError
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
        self.count = 0
        self.currentTime = 0
        self.timer.startTimer()
        self.lastState = 0
        self.moveBot = moveBot()

    def start_control(self):
        listen = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.__callback)
        rospy.spin()

    def __callback(self, data):
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.currentTime = self.clock.getTime()

        # first move is all messed up it thinks there is a pedestrian there
        if self.currentTime - self.startTime > 230 and self.stopped is not True:
            self.moveBot.moveForward(0,0)
            self.timer.endTimer()
            self.stopped = True
        elif self.count >6 and self.stopped is not True:
            self.moveBot.moveForward(0, 0)
            self.timer.endTimer()
            self.stopped = True
        elif self.stopped is True:
            self.moveBot.moveForward(0, 0)
        else:
            if self.currentTime - self.startTime < 6:
                if self.currentTime - self.startTime < 5:
                    self.moveBot.moveForward(0.15, 0)
                else:
                    self.moveBot.moveForward(0, 1)
                self.lastState = 1
                print("hi")
            else:
                if self.lastState ==0:
                    plate1, canMove = self.processPlate.proccessPlate(cv_image)
                    if plate1 is not None:
                        self.moveBot.moveForward(0,0)
                        time.sleep(2)
                        print("plate 1 to NN")
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

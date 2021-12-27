import cv2 as cv
import numpy as np
from geometry_msgs.msg import Twist
import rospy
import time


class InnerLoop:


    def __init__(self):
        #rospy.init_node('my_subscriber', anonymous=True)
        sub = rospy.Subscriber('/R1/cmd_vel', Twist, queue_size=1)
        self.count = 0
        self.filename= "/home/alexkneifel/imitation_imgs_n_labels"
        self.content =[None]*2
        self.match = False
        time.sleep(1)

    def __callback(self, cmd_mesg):
        mesg = str(cmd_mesg).replace("linear: \n  x: ", "")
        mesg = mesg.replace("\n  y:", "")
        mesg = mesg.replace("\n  z:", "")
        mesg = mesg.replace("\nangular: \n  x:", "")
        mesg = mesg.replace("\n  z:", "")
        mesg = str(mesg).split(" ")
        self.content[1] = mesg
        print mesg
        print("callback")
        self.match = True


    def viewWorld(self, img):
        listen = rospy.Subscriber('/R1/cmd_vel', Twist,  self.__callback)
        if self.match == True:
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
            rx = 500.0 / mask.shape[1]
            dim = (500, int(mask.shape[0] * rx))
            resize_img = cv.resize(mask, dim, interpolation=cv.INTER_AREA)
            # i need to resize this image,
            cv.imshow("Mask: ", resize_img)
            cv.waitKey(1)
            self.content[0] = resize_img
            print("mask")
            self.match = False
            self.save_to_file()

    def save_to_file(self):
        # why did it only write one?, maybe it is overwriting
        # if want to overwrite file do 'w'
        with open(self.filename + str(self.count), 'a') as file:
            file.write(str(self.content))




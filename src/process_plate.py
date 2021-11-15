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
    def proccesPlate(self, plate):
       # rectangle
     # need to find all rectangles in the image and see what it gives me
        print(location)


        # corners = cv2.goodFeaturesToTrack(plate, 27, 0.01, 10)
        # corners = np.int0(corners)

        # we iterate through each corner,
        # making a circle at each point that we think is a corner.
        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(img, (x, y), 3, 255, -1)
        #
        # plt.imshow(img), plt.show()
        while True:
            k = cv2.waitKey(0) & 0xFF
            print(k)
            #esc key
            if k == 27:
                cv2.destroyAllWindows()
                break




plateProcessor = ProcessPlate()
img = cv2.imread( '/home/alexkneifel/Pictures/LicensePlate_from_homog.png', cv2.IMREAD_GRAYSCALE)
plateProcessor.proccesPlate(img)
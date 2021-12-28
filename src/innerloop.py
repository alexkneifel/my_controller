import cv2 as cv
import numpy as np
from geometry_msgs.msg import Twist
import rospy
import time
from uuid import uuid4

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


# sess1 = tf.Session()
# graph1 = tf.get_default_graph()
# set_session(sess1)

class InnerLoop:


    def __init__(self):
        #rospy.init_node('my_subscriber', anonymous=True)
        sub = rospy.Subscriber('/R1/cmd_vel', Twist, queue_size=1)
        self.count = 0
        self.filename= "/home/alexkneifel/imitation_imgs_n_labels"
        self.content =[None]*2
        self.match = False
        self.path = "/home/alexkneifel/im_learn_img/"
        #self.vel_model = tf.keras.models.load_model('/home/alexkneifel/Downloads/vel_nn')
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
            # could perhaps reduce sizes even more
            # should resize to quite a bit smaller, and then get lots of data
            rx = 200.0 / mask.shape[1]
            dim = (200, int(mask.shape[0] * rx))
            print(dim)
            resize_img = cv.resize(mask, dim, interpolation=cv.INTER_AREA)
            # i need to resize this image,
            cv.imshow("Mask: ", resize_img)
            cv.waitKey(1)
            self.content[0] = resize_img
            #print("mask")
            self.match = False
            self.save_to_file()
            #return self.find_vel_cmd(resize_img)

    def find_vel_cmd(self, road_img):
        vel1 = 0
        vel6 = 0
        road_img = np.asarray(road_img)
        road_img = road_img.reshape(-1, 500, 281, 1)
        # global sess1
        # global graph1
        # with graph1.as_default():
        #     set_session(sess1)
        #     predicted= self.vel_model.predict(road_img)
        # if predicted == [1,0,0,0,0,0]:
        #     return vel1,vel6
        # elif predicted == [0, 1, 0, 0, 0, 0]:
        #     vel1 = 0.295245
        #     return vel1,vel6
        # elif predicted == [0, 0, 1, 0, 0, 0]:
        #     vel6 = 1.5944049
        #     return vel1, vel6
        # elif predicted == [0, 0, 0, 1, 0, 0]:
        #     vel6 = -1.5944049
        #     return vel1, vel6
        # elif predicted == [0, 0, 0, 0, 1, 0]:
        #     vel1 = 0.295245
        #     vel6 = 1.5944049
        #     return vel1, vel6
        # elif predicted == [0, 0, 0, 0, 0, 1]:
        #     vel1 = 0.295245
        #     vel6 = -1.5944049
        #     return vel1, vel6


    def save_to_file(self):
        # associated labels for each folder
        # ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0'] stationary -> [1,0,0,0,0,0]
        # ['0.295245', '0.0', '0.0', '0.0', '0.0', '0.0'] forward ->[0,1,0,0,0,0]
        # ['0.0', '0.0', '0.0', '0.0', '0.0', '1.5944049'] left -> [0,0,1,0,0,0]
        # ['0.0', '0.0', '0.0', '0.0', '0.0', '-1.5944049'] right-> [0,0,0,1,0,0]
        # ['0.295245', '0.0', '0.0', '0.0', '0.0', '1.5944049'] forward left ->[0,0,0,0,1,0]
        # ['0.295245', '0.0', '0.0', '0.0', '0.0', '-1.5944049'] forward right-> [0,0,0,0,0,1]
        if self.content[1] == ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0']:
            # save to stationary file
            cv.imwrite(self.path+'stationary/'+str(uuid4())+'.jpg', self.content[0])
        elif self.content[1] == ['0.0', '0.0', '0.0', '0.0', '0.0', '1.5944049']:
            # save to left file
            cv.imwrite(self.path + 'left/' + str(uuid4()) + '.jpg', self.content[0])
        elif self.content[1] == ['0.0', '0.0', '0.0', '0.0', '0.0', '-1.5944049']:
            # save to right file
            cv.imwrite(self.path + 'right/' + str(uuid4()) + '.jpg', self.content[0])
        elif self.content[1] == ['0.295245', '0.0', '0.0', '0.0', '0.0', '0.0']:
            # save to forward file
            cv.imwrite(self.path + 'forward/' + str(uuid4()) + '.jpg', self.content[0])
        elif self.content[1] == ['0.295245', '0.0', '0.0', '0.0', '0.0', '1.5944049']:
            # save to forward-left file
            cv.imwrite(self.path + 'left_forward/' + str(uuid4()) + '.jpg', self.content[0])
        elif self.content[1] == ['0.295245', '0.0', '0.0', '0.0', '0.0', '-1.5944049']:
            # save to forward-right file
            cv.imwrite(self.path + 'right_forward/' + str(uuid4()) + '.jpg', self.content[0])
        #print(self.content[1])





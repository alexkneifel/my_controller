#! /usr/bin/env python
from __future__ import print_function
from geometry_msgs.msg import Twist
import roslib
import cv2 as cv
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
        self.s = None
        self.v = None
        self.setVal = False
        self.max_s = 0
        self.max_v = 0

    def isMyRectangle(self, points):
        topLefty = points[0][0][0]
        topLeftx = points[0][0][1]
        topRighty = points[1][0][0]
        topRightx = points[1][0][1]
        botLefty = points[3][0][0]
        botLeftx = points[3][0][1]
        botRighty = points[2][0][0]
        botRightx = points[2][0][1]
        topWidth = topRightx - topLeftx
        botWidth = botRightx - botLeftx
        leftHeight = topLefty - botRighty
        rightHeight = topRighty - botLefty

        if 0.75< topWidth/botWidth <1.25 and 0.75< rightHeight/leftHeight <1.25:
            return True
#
# # perhaps could
    def __normalize_img(self, img):
        max_area = 0

        cv.imshow("original image ", img)
        cv.waitKey(1)

        rx = 400.0 / img.shape[1]
        dim = (400, int(img.shape[0] * rx))
        # perform the actual resizing of the image
        resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        height, width, channels = resized_img.shape
        print("height of small " +str(height))
        print("original height " + str(img.shape[0]))
        ry = height / float(img.shape[0])
        cv_image_crop = resized_img[int(height/2):height, 0:width]

        hsv = cv.cvtColor(cv_image_crop, cv.COLOR_BGR2HSV)  # Convert to hsv color system
        h, s, v = cv.split(hsv)
        result_v = cv.equalizeHist(v)
        result_s = cv.equalizeHist(s)
        #
        hsv = cv.merge((h, result_s, result_v))
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        #
        # # #
        # could also try a different blur
        # cv2.GaussianBlur(gray, (5, 5), 0)
        blur = cv.medianBlur(rgb,5)
        # cv.imshow("normalized s and v", blur)
        # cv.waitKey(1)
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        # homography = None

        uh = 23 #157
        us = 255#8
        uv = 255#168
        lh = 1#0
        ls = 0#0
        lv = 0#87
        lower_hsv = np.array([lh, ls, lv])
        upper_hsv = np.array([uh, us, uv])

        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        #kernel2 = np.ones((2, 2), np.uint8)
        kernel1 = np.ones((3,3), np.uint8)
        # could increase # iterations or kernel size. could do two iterations of each
        #img_erosion = cv.erode(mask, kernel2, iterations=1)
        img_dilation = cv.dilate(mask, kernel1, iterations=2)

        #
        contours = cv.findContours(img_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        # area_thresh = 0
        result = resized_img.copy()
        for c in contours:
            area = cv.contourArea(c)
            #print("contour" + str(c))
            if area > max_area:
                max_c = c
                max_area = area
                #print("max area"+ str(max_area))
                # need to get the points from the contour and check if it is a rectangle w my function
                # need to make sure I don't get the blob up top, could check if perimeter is closed

        leftmost = tuple(max_c[max_c[:, :, 0].argmin()][0])
        rightmost = tuple(max_c[max_c[:, :, 0].argmax()][0])
        top = tuple(max_c[max_c[:, :, 1].argmin()][0])
        bot = tuple(max_c[max_c[:, :, 1].argmax()][0])
        small_leftp = int(leftmost[0])
        small_rightp = int(rightmost[0])
        small_topp = int(top[1])
        small_botp = int(bot[1])

        leftp =  int(small_leftp/rx)
        rightp = int(small_rightp / rx)
        botp = int(small_botp / ry) + int(img.shape[0]/2)
        topp = int(small_topp / ry) + int(img.shape[0]/2)
        license_plate_crop = img[topp:botp,leftp:rightp]


        # should stop the car from moving if it detects a rectangle of right area
        #license_plate_crop = cv_image_crop[int(top[1]): int(bot[1]), int(leftmost[0]):int(rightmost[0])]
        cv.imshow("license plate", license_plate_crop)
        cv.waitKey(1)


        cv.drawContours(result, [max_c], -1, (0, 0, 255), 1)



    def proccessPlate(self, cv_image):
        self.__normalize_img(cv_image)


    #     grayframe = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #     height, width = grayframe.shape
    #     cv_image_crop = cv_image[int(height/3):height, 30:width/2]
    #
    #     img = cv2.imread( '/home/alexkneifel/Downloads/ThreshPlate.png', cv2.IMREAD_GRAYSCALE)
    #         # cv2.imshow("image", img)
    #     blurred_feed = cv2.medianBlur(cv_image_crop, 5)
    #
    # # Convert BGR to HSV
    #     hsv = cv2.cvtColor(blurred_feed, cv2.COLOR_BGR2HSV)
    #     homography = None
    #
    #     uh = 0 #157
    #     us = 2#8
    #     uv = 215#168
    #     lh = 0#0
    #     ls = 0#0
    #     lv = 100#87
    #     lower_hsv = np.array([lh, ls, lv])
    #     upper_hsv = np.array([uh, us, uv])
    #
    #     mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    #
    # #homography
    #
    #     sift = cv2.xfeatures2d.SIFT_create()
    #     kp_image, desc_image = sift.detectAndCompute(img, None)
    #
    #     index_params = dict(algorithm=0, trees=5)
    #     search_params = dict()
    #     flann = cv2.FlannBasedMatcher(index_params, search_params)
    #
    #     kp_grayframe, desc_grayframe = sift.detectAndCompute(mask, None)
    #     matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    #
    #     good_points = []
    #     for m, n in matches:
    #         if m.distance < 0.6 * n.distance:
    #             good_points.append(m)
    #
    #     img3 = cv2.drawMatches(img, kp_image, mask, kp_grayframe, good_points, mask)
    #
    #     cv2.imshow("matches of keypoints", img3)
    #     cv2.waitKey(1)
    #
    #
    #     if len(good_points) > 10:
    #         query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    #         train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    #
    #         matrix, mask2 = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    #         matches_mask = mask2.ravel().tolist()
    #
    #         h, w = img.shape
    #         pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    #         dst = cv2.perspectiveTransform(pts, matrix)
    #
    #         if self.isMyRectangle(dst):
    #             homography = cv2.polylines(cv_image_crop, [np.int32(dst)], True, (255, 0, 0), 3)
    #             cv2.imshow("homography", homography)
    #             cv2.waitKey(1)
    #             license_plate_crop = cv_image_crop[int(dst[0][0][1]): int(dst[1][0][1]), int(dst[0][0][0]):int(dst[3][0][0])]
    #             cv2.imshow("license plate", license_plate_crop)
    #             cv2.waitKey(1)



       # rectangle
     # need to find all rectangles in the image and see what it gives me
        #print(location)


        # corners = cv2.goodFeaturesToTrack(plate, 27, 0.01, 10)
        # corners = np.int0(corners)

        # we iterate through each corner,
        # making a circle at each point that we think is a corner.
        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(img, (x, y), 3, 255, -1)
        #
       #ANPR tutorial
#         cv2.imshow("plate", plate)
#         bfilter = cv2.bilateralFilter(plate, 5, 50, 50)  # Noise reduction
#         cv2.imshow("bilat Filter", bfilter)
#         edged = cv2.Canny(bfilter, 0, 10)
#         cv2.imshow("Canny", edged)
#         keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         contours = imutils.grab_contours(keypoints)
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
#         location = None
#
#         mask = np.zeros(plate.shape, np.uint8)
#         cv2.imshow("mask", mask)
#         #new_image = cv2.drawContours(mask, [location], 0, 255, -1)
#         new_image = cv2.bitwise_and(img, img, mask=mask)
#         cv2.imshow("filtered plate", new_image)
#         for contour in contours:
#             approx = cv2.approxPolyDP(contour, 10, True)
#             if len(approx) == 4:
#                 location = approx
#                 break
#         while True:
#             k = cv2.waitKey(0) & 0xFF
#             print(k)
#             #space bar
#             if k == 32:
#                 cv2.destroyAllWindows()
#                 break
#
#
# plateProcessor = ProcessPlate()
# img = cv2.imread( '/home/alexkneifel/Pictures/LicensePlate_from_homog.png', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("plate", img)
# plateProcessor.proccessPlate(img)

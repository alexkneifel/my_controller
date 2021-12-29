import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt


from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess1 = tf.Session()
graph1 = tf.get_default_graph()
set_session(sess1)


class NeuralNet:
    def __init__(self):
        self.plate_count = 1
        self.num_model = models.load_model('/home/alexkneifel/Downloads/num_nn')
        self.char_model = models.load_model('/home/alexkneifel/Downloads/char_nn')

    def find_chars(self, img_dilation, img_gray):
        count = 0
        order_of_chars = {}
        crops = {}
        flipd_img = cv.bitwise_not(img_dilation)
        width = flipd_img.shape[1]
        max_area = 0
        contours = cv.findContours(flipd_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        #cnt = sorted(contours, key=cv.contourArea, reverse=True)
        for max_c in contours:
            leftmost = tuple(max_c[max_c[:, :, 0].argmin()][0])
            rightmost = tuple(max_c[max_c[:, :, 0].argmax()][0])
            top = tuple(max_c[max_c[:, :, 1].argmin()][0])
            bot = tuple(max_c[max_c[:, :, 1].argmax()][0])
            if leftmost[0] - 5 > 0 and top[1] - 5 > 0:
                small_leftp = int(leftmost[0] - 5)
                small_rightp = int(rightmost[0] + 5)
                small_topp = int(top[1] - 5)
                small_botp = int(bot[1] + 5)
                # making minimum size of bounding box a square
                if small_rightp - small_leftp < small_botp - small_topp:
                    diff = (small_botp - small_topp) - (small_rightp - small_leftp)
                    small_leftp = small_leftp - self.ceildiv(diff, 2)
                    small_rightp = small_rightp + (diff // 2)
                # if including middle thing, cut it in half
                crop_width = small_rightp - small_leftp
                if 0.42 < small_leftp / width < 0.455 and crop_width < 60:
                    small_leftp = small_leftp + int((crop_width) / 2)
                crop = img_gray[small_topp:small_botp, small_leftp:small_rightp]
                if crop.shape[1] / float(crop.shape[0]) >= 1:
                    order_of_chars.update({count: small_leftp})
                    crops.update({count: crop})
                    count = count + 1


        for count in list(crops.keys()):
            if crops[count].shape[0] < 18:
                crops.pop(count)

        four_points = []

        chars = OrderedDict(sorted(crops.items(), key=lambda kv: kv[1].size, reverse=True)[:4])


        for count1, char in chars.items():
            for count2, point in order_of_chars.items():
                if count1 == count2:
                    four_points.append(point)


        zip_iterator = zip(four_points, chars.values())
        chars_dict = OrderedDict(zip_iterator)
        organized_chars_map = OrderedDict(sorted(chars_dict.items(), key=lambda kv: kv[0]))
        organized_chars = list(organized_chars_map.values())

        return organized_chars, flipd_img

    def ceildiv(self, a, b):
        return -(a // -b)

    def four_photo_return(self, plate):
        two_split = False


        gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 107, 22)
        kernel1 = np.ones((2, 2), np.uint8)
        img_erosion = cv.erode(thresh, kernel1, iterations=1)
        dilate = cv.dilate(img_erosion, kernel1, iterations=1)

        chars, flipd_img = self.find_chars(dilate, gray)
        plate_chars = []

        if sum(sum(flipd_img)) > 26000:
            for i in range(4):
                if chars[i].shape[1] / float(chars[i].shape[0]) > 1.455:
                    kernel1 = np.ones((2, 2), np.uint8)
                    two_dilate = cv.dilate(dilate, kernel1, iterations=1)
                    chars, flipd_img = self.find_chars(two_dilate, gray)
                    # print("extra dilation")
                    break

        for i in range(len(chars)):
            height = chars[i].shape[0]
            width = chars[i].shape[1]
            if width / float(height) > 2.65:
                # print("pre-3 split")
                plate_chars.append(chars[i][:height, int((0.333) * width):int((0.666) * width)])
                plate_chars.append(chars[i][:height, int((0.666) * width):width])
            elif width / float(height) > 1.55:
                # print("pre-2 split")
                two_split = True
                plate_chars.append(chars[i][:height, 0:int((0.5) * width)])
                plate_chars.append(chars[i][:height, int((0.5) * width):width])
            else:
                plate_chars.append(chars[i])

        if len(plate_chars) == 5 and two_split:
            min_val = plate_chars[4].size
            min_index = 4
            for i in range(4):
                if plate_chars[i].size < min_val:
                    min_val = plate_chars[i].size
                    min_index = i
            plate_chars.pop(min_index)

        first_char = cv.resize(plate_chars[0], (26, 22), interpolation=cv.INTER_AREA)
        second_char = cv.resize(plate_chars[1], (26, 22), interpolation=cv.INTER_AREA)
        first_num = cv.resize(plate_chars[2], (26, 22), interpolation=cv.INTER_AREA)
        second_num = cv.resize(plate_chars[3], (26, 22), interpolation=cv.INTER_AREA)
        cv.imshow("First Char", first_char)
        cv.waitKey(1)
        cv.imshow("2nd Char", second_char)
        cv.waitKey(1)
        cv.imshow("1st Num", first_num)
        cv.waitKey(1)
        cv.imshow("2nd Num", second_num)
        cv.waitKey(1)

        # order of my chars is messed up, but it is getting all the chars which is good

        return first_char, second_char, first_num, second_num

    def neuralnetwork(self, plate):
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        nums = "0123456789"
        # right now these char images are no good

        # step 2 this function is good so need to check into these methods, what this is returning
        first_char_img, second_char_img, first_num_img, second_num_img = self.four_photo_return(plate)

        # this is saying it is missing fourth dimension
        first_char_img = np.asarray(first_char_img)
        first_char_img = first_char_img.reshape(-1, 26, 22, 1)
        global sess1
        global graph1
        with graph1.as_default():
            set_session(sess1)
            predicted_one = self.char_model.predict(first_char_img)
        first_char = [characters[np.argmax(i)] for i in predicted_one]

        second_char_img = np.asarray(second_char_img)
        second_char_img = second_char_img.reshape(-1, 26, 22, 1)
        with graph1.as_default():
            set_session(sess1)
            predicted_two = self.char_model.predict(second_char_img)
        second_char = [characters[np.argmax(i)] for i in predicted_two]

        first_num_img = np.asarray(first_num_img)
        first_num_img = first_num_img.reshape(-1, 26, 22, 1)
        with graph1.as_default():
            set_session(sess1)
            predicted_three = self.num_model.predict(first_num_img)
        first_num = [nums[np.argmax(i)] for i in predicted_three]

        second_num_img = np.asarray(second_num_img)
        second_num_img = second_num_img.reshape(-1, 26, 22, 1)
        with graph1.as_default():
            set_session(sess1)
            predicted_four = self.num_model.predict(second_num_img)
        second_num = [nums[np.argmax(i)] for i in predicted_four]

        return first_char, second_char, first_num, second_num

    def __getParkingNumber(self):
        self.plate_count += 1
        if self.plate_count > 6:
            self.plate_count = 1
        return self.plate_count

    def licencePlateToString(self, img):

        first_char, second_char, first_num, second_num = self.neuralnetwork(img)

        parking_num = self.__getParkingNumber()

        char_list = first_char + second_char + first_num + second_num

        char_list = ''.join(char_list)
        command = str('alexsean,cheese1,') + str(parking_num) + ',' + str(char_list)
        return command

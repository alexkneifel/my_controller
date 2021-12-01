import numpy as np
import cv2

from matplotlib import pyplot as plt
import tensorflow as tf
# from tensforflow import keras


class NeuralNet:
    def __init__(self):
        self.plate_count = 1

    def image_cropper(self,img):

        img_bin = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if (img[i, j] > 60):
                    img_bin[i, j] = 255
                else:
                    img_bin[i, j] = 0

        col_sum = np.zeros((img_bin.shape[1], 1))
        for i in range(0, img_bin.shape[1]):
            col = 0
            for j in range(0, img_bin.shape[0]):
                col += img_bin[j, i]
            col_sum[i] = col

        i = 0
        a = 0
        while col_sum[i] < 500:
            i += 1
            if col_sum[i] > 500:
                a = i
        crop_img_bin = img_bin[0:img_bin.shape[0], a:img_bin.shape[1]]
        return crop_img_bin

    def bound_finder(self,img):
        initial_guess1 = 5
        thresh = 50
        margin = 5

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        img_bin = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if (img[i, j] > 60):
                    img_bin[i, j] = 0
                else:
                    img_bin[i, j] = 255

        col_sum = np.zeros((img_bin.shape[1], 1))
        for i in range(0, img_bin.shape[1]):
            col = 0
            for j in range(0, img_bin.shape[0]):
                col += img_bin[j, i]
            col_sum[i] = col
            # print(str(i) + " " + str(col_sum[i]))

        a = initial_guess1
        while col_sum[a] < thresh:
            a += 1
        a -= margin
        b = a + margin + 1
        while col_sum[b] > thresh:
            b += 1
        b += margin

        c = b
        while col_sum[c] < thresh:
            c += 1
        c -= margin
        d = c + margin + 1
        while col_sum[d] > thresh:
            d += 1
        d += margin

        imgshape = int(img.shape[1])
        e = d + int(imgshape * 0.18)
        while col_sum[e] < thresh:
            e += 1
        e -= margin
        f = e + margin + 1
        while col_sum[f] > thresh:
            f += 1
        f += margin

        g = f
        while col_sum[g] < thresh:
            g += 1
        g -= margin
        h = g + margin + 1
        while col_sum[h] > thresh:
            h += 1
        h += margin

        return (a, b), (c, d), (e, f), (g, h)
    def colsum(self,cropped_img):
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        col_sum = np.zeros((cropped_img.shape[1], 1))
        for i in range(0, cropped_img.shape[1]):
            col = 0
            for j in range(0, cropped_img.shape[0]):
                col += cropped_img[j, i]
            col_sum[i] = col

    def plotter(self,cropped_img):

        (a,b), (c,d), (e,f), (g,h) = self.bound_finder(cropped_img)

        char1=cropped_img[0:60,a:b]
        char2=cropped_img[0:60,c:d]
        char3=cropped_img[0:60,e:f]
        char4=cropped_img[0:60,g:h]

        return char1,char2,char3,char4

    def neuralnetwork(self, char_img):
        model = tf.keras.models.load_model('/home/alexkneifel/ros_ws/src/my_controller/src/competition-image-to-char-2.h5')
        #rx = 35.0 / char_img.shape[1]
        dim = (35, 45)
        resized_img = cv2.resize(char_img, dim, interpolation=cv2.INTER_AREA)
        char_img_aug = np.expand_dims(resized_img, axis=0)
        char_predict = model.predict(char_img_aug)[0]
        return char_predict
    def toCharacter(self,arg):
        if arg<10:
            return str(arg)
        else:
            return chr(arg+55)
    def __getParkingNumber(self):
        self.plate_count +=1
        if self.plate_count > 6:
            self.plate_count =1
        return self.plate_count

    def licencePlateToString(self,img):

        parking_num = self.__getParkingNumber()
        # char1, char2, char3, char4 = self.plotter(img)
        #
        # char1 = self.toCharacter(np.argmax(self.neuralnetwork(char1)))
        # char2 = self.toCharacter(np.argmax(self.neuralnetwork(char2)))
        # char3 = self.toCharacter(np.argmax(self.neuralnetwork(char3)))
        # char4 = self.toCharacter(np.argmax(self.neuralnetwork(char4)))
        # char_list = char1+char2+char3+char4
        command = str('alexsean,cheese1,')+str(parking_num)+','+'0000'
        return command


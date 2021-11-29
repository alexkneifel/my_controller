import numpy as np
import cv2

class NeuralNet:
	def image_cropper(img):

	    img_bin = np.zeros((img.shape[0],img.shape[1]))
	    for i in range(0,img.shape[0]):
	        for j in range(0,img.shape[1]):
	            if(img[i,j]>60):
	                img_bin[i,j]=255
	            else:
	                img_bin[i,j]=0
	                
	    col_sum = np.zeros((img_bin.shape[1],1))
	    for i in range(0,img_bin.shape[1]):
	        col=0
	        for j in range(0,img_bin.shape[0]):
	            col+=img_bin[j,i]
	        col_sum[i] = col
	    
	    i = 0
	    a=0
	    while col_sum[i] < 500:
	        i+=1
	        if col_sum[i]>500:
	            a=i
	    crop_img_bin = img_bin[0:img_bin.shape[0],a:img_bin.shape[1]]
    return crop_img_bin
    def bound_finder(img):
	    initial_guess1 = 18
	    initial_guess2 = 150
	    thresh = 200
	    
	    img_bin = np.zeros((img.shape[0],img.shape[1]))
	    for i in range(0,img.shape[0]):
	        for j in range(0,img.shape[1]):
	            if(img[i,j]>60):
	                img_bin[i,j]=0
	            else:
	                img_bin[i,j]=255
	    
	    col_sum = np.zeros((img_bin.shape[1],1))
	    for i in range(0,img_bin.shape[1]):
	        col=0
	        for j in range(0,img_bin.shape[0]):
	            col+=img_bin[j,i]
	        col_sum[i] = col
	        # print(str(i) + " " + str(col_sum[i]))

	    a = initial_guess1
	    b = initial_guess1+44
	    while col_sum[b-5] > thresh:
	        b+=1
	        if col_sum[b-5] <thresh:
	            a=b-44
	    c = b+1
	    d = c+44

	    e = initial_guess2
	    f = initial_guess2+44
	    while col_sum[f-5] >thresh:
	        f+=1
	        if col_sum[f-5] < thresh:
	            e=f-44
	    g=f+1
	    h = g+44
	    return (a,b), (c,d), (e,f), (g,h)
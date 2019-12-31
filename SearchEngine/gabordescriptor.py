# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 19:45:23 2019

@author: Dell
"""

import cv2
import numpy as np

class GaborDescriptor:
    
    def build_filters(self):
        filters = []
        ksize = 9
        #define the range for theta and nu
        for theta in np.arange(0, np.pi, np.pi / 8):
            for nu in np.arange(0, 6*np.pi/4 , np.pi / 4):
                kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, nu, 0.5, 0, ktype=cv2.CV_32F)
                kern /= 1.5*kern.sum()
                filters.append(kern)
        return filters
    
    def process(self,img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum
    
    def extract_features(self,img):
        
        filters = self.build_filters()
        f = np.asarray(filters)
        #imgg = cv2.imread(img,0)
        feat = []
    	#calculating the local energy for each convolved image
        for j in range(40):
            res = self.process(img, f[j])
            temp = 0
            for p in range(128):
                for q in range(128):
                    temp = temp + res[p][q]*res[p][q]
            feat.append(temp)
    	#calculating the mean amplitude for each convolved image	
        for j in range(40):
            res = self.process(img, f[j])
            temp = 0
            for p in range(128):
                for q in range(128):
                    temp = temp + abs(res[p][q])
            feat.append(temp)
        #feat matrix is the feature vector for the image
        myarray = np.asarray(feat).flatten()
        myarray =[(float(i)-min(myarray))/(max(myarray)-min(myarray)) for i in myarray]
        return myarray
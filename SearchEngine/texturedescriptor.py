# -*- coding: utf-8 -*-

import mahotas as mt
import cv2

class TextureDescriptor:
    
    def extract_features(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(gray)
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        norm=[(float(i)-min(ht_mean))/(max(ht_mean)-min(ht_mean)) for i in ht_mean]
        return norm
        



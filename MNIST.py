#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:45:32 2020

@author: alfredocu
"""

# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA # Modelo no supervisado.
import pickle

img = mpimg.imread("ocho-original.jpeg")
plt.imshow(img)

img = mpimg.imread("ocho.png")
plt.imshow(img, cmap=plt.cm.gray)
img = img.reshape((1,-1))

model = pickle.load(open("Mnist_classifier.sav", "rb"))
print("Prediction: ", model.predict(img)[0])
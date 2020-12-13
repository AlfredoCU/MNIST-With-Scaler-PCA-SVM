#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:12:07 2020

@author: alfredocu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # Modelo no supervisado.
from sklearn.pipeline import Pipeline

data = pd.read_csv("mnist_784.csv")
# print(data)

n_samples = 70000 # 500

x = np.asanyarray(data.drop(columns=["class"]))[:n_samples,:]
y = np.asanyarray(data[["class"]])[:n_samples].ravel()

# sample = np.random.randint(n_samples)
# plt.imshow(x[sample].reshape((28, 28)), cmap=plt.cm.gray)
# plt.title("Class: %i" % y[sample])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=50)),
    ("svm", svm.SVC(gamma=0.0001))])

model.fit(xtrain, ytrain)
print("Train: ", model.score(xtrain, ytrain))
print("Test: ", model.score(xtest, ytest))

ypred = model.predict(xtest)
print("Classification report: \n", metrics.classification_report(ytest, ypred))
print("Confusion matrix: \n", metrics.confusion_matrix(ytest, ypred))

sample = np.random.randint(xtest.shape[0])
plt.imshow(xtest[sample].reshape((28, 28)), cmap=plt.cm.gray)
plt.title("Prediction: %i" % ypred[sample])
# plt.savefig("Predict.eps", format="eps")
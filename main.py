# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 22:10:35 2021

@author: Camilo Martínez
"""
from utils.functions import load_variable_from_file, visualize_segmentation
from feature_descriptor import multiscale_statistics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

gt = load_variable_from_file("ground_truth_with_originals", "saved_variables")[:, 0]
imgs = (
    load_variable_from_file("ground_truth_with_originals", "saved_variables")[:, 1]
    / 255.0
)

X = []
y = []
for i, img in enumerate(imgs):
    print(f"Loading image {i}... ", end="")
    features = multiscale_statistics(img, 3)
    X.append(features.reshape(-1, features.shape[-1]))
    y.append(gt[i].ravel().astype(int))
    print("Done")

X = np.concatenate(X)
y = np.concatenate(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)

# %%
i = 30

features = multiscale_statistics(imgs[i], 3)
predicted = gnb.predict(features.reshape(-1, features.shape[-1]))

visualize_segmentation(
    imgs[i], ["pearlite", "proeutectoid ferrite"], predicted.reshape(imgs[50].shape)
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 12:24:04 2024
Objective:
    - Calculate the standard deviation and mean values 
     of the dataset. That include all the categories Training, Test and Val
    - Collect them under one list
@author: dagi
"""
import cv2
import os 
import numpy as np

# In[] Set Route path
root_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Ensemble/Model_1/"

# In[] Create list from root_path
dataset_mri = []
for dirPath, dirNames, fileNames in os.walk(root_path):
    for file in fileNames:
        if file.endswith('.jpg' ) or file.endswith('.tif'):
            full_path = dirPath + "/" + file
            dataset_mri.append(full_path)

# In[] Initialise the parameters 
mean_sum = 0
sum_img = None # accummulate the sum of pixel values of the entire dataset
std_sum = 0 
std_value = 0
height = 256 
width = 256

# In[] Iterate over dataset_mri, read each image and collect the commulative sum value of each pixels
for img_name in dataset_mri:
    # Read the image in grayscale
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    # Resize the image to 256x256 for consistency
    img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
    # accumulate teh sum of pixel values of each individual pixels
    if sum_img is None:
        sum_img = img / 255
    else:
        sum_img += img/255
        
# In[] calculating the mean
mean_img = sum_img / len(dataset_mri)

# In[] Calculate  the mean value of pixels for each channel
mean_pixel_value = np.mean(mean_img, axis=(0, 1))

# mean_pixel_value = [0.18206]

# In[] For standard deviation
sum_squared_img = None
squared_diff = 0

# In[] Calculating STD
# Go through the same dataset of training mri images
for img_path in dataset_mri:
    # Read the image as Grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to 256x256
    img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)

    # Accumulate the squared differences from the mean image
    squared_diff = (img/255 - mean_img) ** 2
    if sum_squared_img is None:
        sum_squared_img = squared_diff
    else:
        sum_squared_img += squared_diff
    
# In[] Calculating the variance
variance = sum_squared_img / len(dataset_mri)

# Standard Deviation
std = np.sqrt(np.mean(variance, axis = (0, 1)))

# std = [0.16517]

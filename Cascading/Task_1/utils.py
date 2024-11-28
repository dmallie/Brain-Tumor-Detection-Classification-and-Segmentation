#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:08:42 2024
Objective:
    - Do the following jobs
        - Calculate the accuracy of the model's output
@author: dagi
"""

import torch 
import cv2 
import numpy as np
import os 

# In[] Calculats the precision of the model 
def precision(preds, labels):
    preds = preds.round()  # Rounding sigmoid outputs to get binary predictions
    TP = (preds * labels).sum().item()  # True positives
    FP = ((preds == 1) & (labels == 0)).sum().item()  # False positives
    # Avoid division by zero
    precision_value = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return precision_value

# In[] Setup the accuracy calculator
def accuracy_calculator(y_pred = None, y_true = None):
    # find the labels of predicted values from y_pred
    pred_labels = y_pred.round()
    # compare the predicted labels with true labels & count correctly predicted labels
    # correct_predictions = torch.eq(y_true, pred_labels).sum().item()
    correct_predictions = 0
    wrong_predictions = 0
    for index in range(len(pred_labels)):
        if pred_labels[index] == y_true[index]:
            correct_predictions += 1
        else:
            wrong_predictions += 1
    # calculate the incorrect ones
    return correct_predictions

# In[] 
def model_accuracy(pred_list = None, true_label_list = None):
    # check first if both lists are of the same length
    if len(pred_list) != len(true_label_list):
        return "Size of the two lists don't match"
    
    # accumulte the value of right predictions
    right_preds = 0
    
    for index in range(len(pred_list)):
        if pred_list[index] == true_label_list[index]:
            right_preds += 1
    return right_preds

# In[] Calculate the mean and standard deviation for test dataset
def calculate_mean_std(path):
    # create list of all images with their full_path
    dataset_mri = []
    for dirPath, dirNames, fileNames in os.walk(path):
        for file in fileNames:
            if file.endswith('.jpg' ) or file.endswith('.tif'):
                full_path = dirPath + "/" + file
                dataset_mri.append(full_path)

    # set parameters based on which to process mean
    sum_img = None # accummulate the sum of pixel values of the entire dataset
    height = 256 
    width = 256

    # Calculate the mean
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
        
    #  calculating the mean
    mean_img = sum_img / len(dataset_mri)

    # Calculate  the mean value of pixels for each channel
    mean_pixel_value = np.mean(mean_img, axis=(0, 1))

    # set parameters for standard deviation
    sum_squared_img = None
    squared_diff = 0

    # calculate the standard deviation
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
    # Calculating the variance
    variance = sum_squared_img / len(dataset_mri)

    # Standard Deviation
    std = np.sqrt(np.mean(variance, axis = (0, 1)))

    # return the mean and standard deviation of the dataset
    return mean_pixel_value, std
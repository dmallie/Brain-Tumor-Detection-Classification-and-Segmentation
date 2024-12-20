#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:22:06 2024
Objective:
    - This script contains utility function for brain tumor classification
        - accuracy_calculator: for each batch of prediction it calculates how many of 
            it is correct or otherwise
        - model_accuracy: this returns the number of right prediction during testing
        - calculate_mean_std: this calculates the mean and standard_deviation for each dataset
@author: dagi
"""
import torch 
import cv2
import numpy as np 
import os 
from sklearn.metrics import confusion_matrix

# In[] Setup the accuracy calculator
# def accuracy_calculator(y_pred=None, y_true=None):
#     """
#     Calculate the number of correct predictions.

#     Args:
#         y_pred (torch.Tensor): Model predictions (logits or probabilities), shape [batch_size, num_classes].
#         y_true (torch.Tensor): True labels, shape [batch_size].

#     Returns:
#         int: Number of correct predictions.
#     """
#     # Get the predicted labels from the model's output
#     pred_labels = torch.argmax(y_pred, dim=1)

#     # Count correctly predicted labels using vectorized operations
#     correct_predictions = (pred_labels == y_true).sum().item()

#     return correct_predictions

def accuracy_calculator(y_pred = None, y_true = None):
    # find the labels of predicted values from y_pred
    # pred_labels = torch.argmax(y_pred, dim=1)
    # print(f"y_pred: {y_pred.shape}")
    # print(f"y_true: {y_true.shape}")
    # pred_labels = [row.index(max(row)) for row in y_pred]
    # compare the predicted labels with true labels & count correctly predicted labels
    # correct_predictions = torch.eq(y_true, pred_labels).sum().item()
    correct_predictions = 0
    wrong_predictions = 0
    for index in range(len(y_true)):
        if y_pred[index] == y_true[index]:
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
# %% Implementation of Precision
def precision(y_pred, y_true):
    class_labels = ["No Tumor", "Pituitary", "Glioma", "Meningioma"]
    # Ensure both inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels = np.arange(len(class_labels)))
    # Precision for each class
    precision_per_class = {}
    for class_id in range(len(class_labels)):
        true_positives = cm[class_id, class_id]
        false_positives = cm[:, class_id].sum() - true_positives
 
        if true_positives + false_positives == 0:
            precision = 0.0  # Avoid division by zero
        else:
            precision = true_positives / (true_positives + false_positives)
 
        precision_per_class[class_id] = precision
 
    # Macro-averaged precision
    precision = np.mean(list(precision_per_class.values()))
    return precision

# %% Implementation of Recall
def recall(y_pred, y_true):
    class_labels = ["No Tumor", "Pituitary", "Glioma", "Meningioma"]
    # Ensure both inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels = np.arange(len(class_labels)))
    # Precision for each class
    recall_per_class = {}
    for class_id in range(len(class_labels)):
        true_positives = cm[class_id, class_id]
        false_negatives = cm[class_id, :].sum() - true_positives
        if true_positives + false_negatives == 0:
            recall = 0.0  # Avoid division by zero
        else:
            recall = true_positives / (true_positives + false_negatives)
 
        recall_per_class[class_id] = recall
 
    # Macro-averaged precision
    recall = np.mean(list(recall_per_class.values()))
    return recall

# %% Implementation of F1-Score
def f1_score(y_pred, y_true):
    # calculate the average precision of the dataset
    precision_value = precision(y_pred, y_true)
    # calculate the average recall value of the prediction
    recall_value = recall(y_pred, y_true)
    # to avoide potential division by zero 
    if precision_value + recall_value == 0:
        return 0.0
    # Calculate the F1-Score 
    f1 = 2 * (precision_value * recall_value)/(precision_value + recall_value)
    
    return f1 
    
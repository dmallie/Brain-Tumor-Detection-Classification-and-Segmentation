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


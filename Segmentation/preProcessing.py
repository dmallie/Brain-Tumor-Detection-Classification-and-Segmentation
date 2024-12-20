#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:49:15 2024
Objective:
    - This script a new dataset in the specified directories
    - The new dataset first using in built cv2 histogram based equalization we enhance the image
    - The enhanced image will be subjected to skullstripping threshold algorithm
    - The output is saved in the preProcessed directory
@author: dagi
"""
import os 
from utils import simple_skull_strip
import cv2 
from tqdm import tqdm 
import numpy as np 

# %% set the source directories
root_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Segmentation/"
test_path = root_path + "Test/Images/"
train_path = root_path + "Train/Images/"
val_path = root_path + "Val/Images/"

src_path = [test_path, train_path, val_path]

#%% create list of image files from the directories
test_lst = os.listdir(test_path)
train_lst = os.listdir(train_path)
val_lst = os.listdir(val_path)

src_lst = [test_lst, train_lst, val_lst]

# %% Set the destination pathes
test_preProcessed = root_path + "Test/preProcessed/"
train_preProcessed = root_path + "Train/preProcessed/"
val_preProcessed = root_path + "Val/preProcessed/"

dest_path = [test_preProcessed, train_preProcessed, val_preProcessed]

# %% iterate through the lists and do the preprocessing then save them in the right destination
description = ["pre-processing the test set", "pre-processing the train set", "pre-processing the val set"]
for index, current_lst in enumerate(src_lst):
    # tierate through each files
    for img_files in tqdm(current_lst, desc = description[index], leave=False):
        # set the full path of the file
        full_path = src_path[index] + img_files
        # load the image in gray scale format
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        img_1 = simple_skull_strip(img)
        # convert binary_img uint8
        binary_img = (img_1 *  255).astype(np.uint8)

        # apply histogram equalization on the image
        img_processed = cv2.equalizeHist(binary_img)
        # apply the simple threshold algorithm to filter out some of the unwanted features from teh image
        threshold = simple_skull_strip(img_processed)
        # convert the threshold image to numpy unsigned int datatype
        preprocessed_img = (threshold *  255).astype(np.uint8)
        # set the destination path
        save_path = dest_path[index] + img_files
        # save the file
        cv2.imwrite(save_path, preprocessed_img)
    
        
        

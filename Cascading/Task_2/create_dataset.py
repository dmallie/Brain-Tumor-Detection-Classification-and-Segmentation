#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 07:04:43 2024
Objective:
    - Create dataset for no_tumor category
    - Data set is composed from 2 different sources
        - Source : https://www.kaggle.com/datasets/malicks111/brain-tumor-detection
        - Data is organized as Training and Testing.
            - Training : organized as glioma_tumor (6613), meningioma_tumor (6708), no_tumor (2842)
            and pituitary_tumor (6189)
            - Testin: glioma_tumor(620), meningioma_tumor(620), no_tumor (620) and pituitary_tumor (620)
    - The combined size of the dataset with the three types of tumors is 21370 thus 
        I didn't need to look for another dataset    
@author: dagi
"""
import os
import cv2 
import shutil 
import numpy as np 
import random 

# In[] Classify all mri scans either as tumor or no_tumor based on their corresponding mask file
glioma = []
meningioma = []
pituitary = []

dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Ensemble/Model_2/"
# In[] Set path and create list from source
src_path = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/"
# Traverse through the directory 
for root, dirs, files in os.walk(src_path):
    for folder in dirs:
        if "_tumor"  in folder:
            folder_path = root + "/" + folder
            mri_scans = os.listdir(folder_path)
            for each_scan in mri_scans:
                full_path = folder_path + "/" + each_scan
                if folder == "no_tumor":
                    continue
                elif folder == "glioma_tumor":
                    glioma.append(full_path)
                elif folder == "meningioma_tumor":
                    meningioma.append(full_path)
                else:
                    pituitary.append(full_path)
                    
# In[] Shuffle and split the datasets to Test 10%, Train 70% and Val 20%
random.shuffle(glioma)
random.shuffle(meningioma)
random.shuffle(pituitary)

# In[] Populat test train and val directories 
# calculate the last index from the whole dataset that corresponds to the size of training set
train_cutoff_glioma = int(0.7 * len(glioma)) # all scans till train_cutoff_glioma go to train dataset
train_cutoff_meningioma = int(0.7 *  len(meningioma)) # all scans till train_cutoff_meningioma go to train dataset
train_cutoff_pituitary = int(0.7 *  len(pituitary)) # all scans till train_cutoff_pituitary go to train dataset

# calculate the last index from the whole dataset that corresponds to the size of validation set 
val_cutoff_glioma = int(0.2 *  len(glioma)) + train_cutoff_glioma
val_cutoff_meningioma = int(0.2 * len(meningioma)) + train_cutoff_meningioma 
val_cutoff_pituitary = int(0.2 * len(pituitary)) + train_cutoff_pituitary

dest_path_list = [dest_path+"Train/glioma/", dest_path+"Train/meningioma/", dest_path+"Train/pituitary/", 
                  dest_path+"Val/glioma/", dest_path+"Val/meningioma/", dest_path+"Val/pituitary/",
                  dest_path+"Test/glioma/", dest_path+"Test/meningioma/", dest_path+"Test/pituitary/"]
# In[] split glioma dataset into training, testing and val
for index, path in enumerate(glioma): 
    # decide to which dataset group the item should go
    if index <= train_cutoff_glioma:
        # move the image to the training directory
        shutil.copy(path, dest_path_list[0])
    elif index <= val_cutoff_glioma:
        # copy and paste the image to the Val directory
        shutil.copy(path, dest_path_list[3])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, dest_path_list[6])
    
# In[] split meningioma dataset into training, testing and val
for index, path in enumerate(meningioma): 
    # decide to which dataset group the item should go
    if index <= train_cutoff_meningioma:
        # move the image to the training directory
        shutil.copy(path, dest_path_list[1])
    elif index <= val_cutoff_meningioma:
        # copy and paste the image to the Val directory
        shutil.copy(path, dest_path_list[4])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, dest_path_list[7])
    
# In[] split pituitary dataset into training, testing and val
for index, path in enumerate(pituitary): 
    # decide to which dataset group the item should go
    if index <= train_cutoff_pituitary:
        # move the image to the training directory
        shutil.copy(path, dest_path_list[2])
    elif index <= val_cutoff_pituitary:
        # copy and paste the image to the Val directory
        shutil.copy(path, dest_path_list[5])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, dest_path_list[8])
    








#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:43:37 2024
Objective:
    - Create dataset for no_tumor category
    - Data set is composed from 2 different sources
        - Source 1 : https://www.kaggle.com/datasets/deepaa28/brain-tumor-classification/data
        - Data comes from 110 different subjects
        - Data is organized for segementation purpose, thus for each MRI there is a corresponding mask file
        - By examining the mask file, if no tumor is identified in the mask, I assume that no tumor is present in 
        the corresponding slice of the MRI scan and I included that slice into no_tumor category
        - Source 2: https://www.kaggle.com/datasets/malicks111/brain-tumor-detection
        - Data is organized as Training and Testing.
            - Training : organized as glioma_tumor (6613), meningioma_tumor (6708), no_tumor (2842)
            and pituitary_tumor (6189)
            - Testing: glioma_tumor(620), meningioma_tumor(620), no_tumor (620) and pituitary_tumor (620)
        
@author: dagi
"""
import os
import cv2 
import shutil 
import numpy as np 
import random 

# In[] Set path for the source 1 and create list from the directory
src_1_path = "/media/Linux/Downloads/Brain Tumor/Brain_Tumor kaggle/kaggle_3m/"
dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Ensemble/Model_1/"

# In[] Classify all mri scans either as tumor or no_tumor based on their corresponding mask file
tumor = []
no_tumor = []
# Walk through the directory structure
for root, dirs, files in os.walk(src_1_path):
    for folder in dirs:
        path = root + folder + "/"
        mri_scans = os.listdir(path)
        for each_scan in mri_scans:
            # if each_scan is not the mask file then skip to the next
            if "_mask" not in each_scan:
                continue
            # set the full path of the file
            mask_full_path = root + folder + "/" + each_scan
            # read the mask file 
            mask_img = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
            # check if there is any tumor present in the mask file 
            has_tumor = np.any(mask_img == 255)
            # if mask file is pitch black then categorize the full path of the corresponding scan to no_tumor
            scan_full_path = mask_full_path.replace("_mask", "")
            if has_tumor:
                tumor.append(scan_full_path)
            else:
                # otherwise categoize the scan to tumor
                no_tumor.append(scan_full_path)

# In[] Set path and create list from source 2
src_2_path = "/media/Linux/Downloads/Brain Tumor/Brain Tumor Detection by MALICKS111/"
# Traverse through the directory 
for root, dirs, files in os.walk(src_2_path):
    for folder in dirs:
        if "_tumor"  in folder:
            folder_path = root + "/" + folder
            mri_scans = os.listdir(folder_path)
            for each_scan in mri_scans:
                scan_full_path = folder_path + "/" + each_scan
                if folder == "no_tumor":
                    no_tumor.append(scan_full_path)
                else:
                    tumor.append(scan_full_path)

# In[] Shuffle and divide the datasets to Test 10%, Train 70% and Val 20%

random.shuffle(tumor)
random.shuffle(no_tumor)

# In[] Populat test train and val directories 

train_size = int(0.7 * len(tumor))
val_limit = int(0.2 *  len(tumor)) + train_size

tumor_dest_path = [dest_path+"Train/tumor/", dest_path+"Val/tumor/", dest_path+"Test/tumor/"]
# split tumor dataset into training, testing and val
for index, path in enumerate(tumor): 
    # decide to which dataset group the item should go
    if index <= train_size:
        # move the image to the training directory
        shutil.copy(path, tumor_dest_path[0])
    elif index <= val_limit:
        # copy and paste the image to the Val directory
        shutil.copy(path, tumor_dest_path[1])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, tumor_dest_path[2])
        
# In[] Now split the no_tumor dataset to train, val and test 

tumor_dest_path = [dest_path+"Train/no_tumor/", dest_path+"Val/no_tumor/", dest_path+"Test/no_tumor/"]
train_size = int(0.7 * len(no_tumor))
val_limit = int(0.2 *  len(no_tumor)) + train_size

# split tumor dataset into training, testing and val
for index, path in enumerate(no_tumor): 
    # decide to which dataset group the item should go
    if index <= train_size:
        # move the image to the training directory
        shutil.copy(path, tumor_dest_path[0])
    elif index <= val_limit:
        # copy and paste the image to the Val directory
        shutil.copy(path, tumor_dest_path[1])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, tumor_dest_path[2])

     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        













#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:11:49 2024
Objective:
    - Create dataset for Independent (non-cascading) classification project
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
        - Source 3: https://www.kaggle.com/datasets/zakariaolamine/brain-tumor-dataset
            - Data set contains 3064 weighted contrast-enhanced mri scans from 233 patients
            - Data comes in matlab .mat file, then mri scan 
            - Each .mat file labels coordinates for mask file, mri scan file tumor segmentation coordinates are extracted
            - Dataset is structured in glioma(1426), meningioma(708) and pituitary(930)
    - The purpose of this script is read the downloaded directory, shuffle the data, organize the dataset 
     to training, testing and val categories.
     
@author: dagi
"""
import os
import cv2 
import shutil 
import numpy as np 
import random 

# In[] Set path for the source 1 and create list from the directory
src_1_path = "/media/Linux/Downloads/Brain Tumor/Brain_Tumor kaggle/kaggle_3m/"
dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Independent/"

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
glioma = []
meningioma = []
pituitary = []
# In[]]}
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
                elif folder == "glioma_tumor":
                    glioma.append(scan_full_path)
                elif folder == "meningioma_tumor":
                    meningioma.append(scan_full_path)
                elif folder == "pituitary_tumor":
                    pituitary.append(scan_full_path)
                    
# In[] collect the mri scans from Source 3
glioma_3_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/glioma/mri/"
meningioma_3_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/meningioma/mri/"
pituitary_3_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Brain_tumor_dataset_by_cheng/pituitary/mri/"

src_path = [glioma_3_path, meningioma_3_path, pituitary_3_path]
# create list from directories
glioma_3 = os.listdir(glioma_3_path)
meningioma_3 = os.listdir(meningioma_3_path)
pituitary_3 = os.listdir(pituitary_3_path)

source_3 = [glioma_3, meningioma_3, pituitary_3]
# merge the data with source_2
for index, category in enumerate(source_3):
    for scan in category:
        # get the full path
        full_path = src_path[index] + scan
        # determine to which types of tumor the scan belongs
        if index == 0:
            glioma.append(full_path)
        elif index == 1: 
            meningioma.append(full_path)
        else: 
            pituitary.append(full_path)
            
# In[] Shuffle the dataset 
random.shuffle(no_tumor)
random.shuffle(glioma)
random.shuffle(meningioma)
random.shuffle(pituitary)

# In[] Split the glioma dataset to Test 10%, Train 70% and Val 20% directories 

train_end_index = int(0.7 * len(glioma))
val_end_index = int(0.2 *  len(glioma)) + train_end_index

dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Independent/"
train_path = [dest_path+"Train/glioma/", dest_path + "Val/glioma/", dest_path + "Test/glioma/"]

# split tumor dataset into training, testing and val
for index, path in enumerate(glioma): 
    # decide to which dataset group the item should go
    if index <= train_end_index:
        # move the image to the training directory
        shutil.copy(path, train_path[0])
    elif index <= val_end_index:
        # copy and paste the image to the Val directory
        shutil.copy(path, train_path[1])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, train_path[2])

# In[] Split the meningioma dataset to Test 10%, Train 70% and Val 20% directories 

train_end_index = int(0.7 * len(meningioma))
val_end_index = int(0.2 *  len(meningioma)) + train_end_index

dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Independent/"
train_path = [dest_path+"Train/meningioma/", dest_path + "Val/meningioma/", dest_path + "Test/meningioma/"]

# split tumor dataset into training, testing and val
for index, path in enumerate(meningioma): 
    # decide to which dataset group the item should go
    if index <= train_end_index:
        # move the image to the training directory
        shutil.copy(path, train_path[0])
    elif index <= val_end_index:
        # copy and paste the image to the Val directory
        shutil.copy(path, train_path[1])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, train_path[2])

# In[] Split the no_tumor dataset to Test 10%, Train 70% and Val 20% directories 

train_end_index = int(0.7 * len(no_tumor))
val_end_index = int(0.2 *  len(no_tumor)) + train_end_index

dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Independent/"
train_path = [dest_path+"Train/no_tumor/", dest_path + "Val/no_tumor/", dest_path + "Test/no_tumor/"]

# split tumor dataset into training, testing and val
for index, path in enumerate(no_tumor): 
    # decide to which dataset group the item should go
    if index <= train_end_index:
        # move the image to the training directory
        shutil.copy(path, train_path[0])
    elif index <= val_end_index:
        # copy and paste the image to the Val directory
        shutil.copy(path, train_path[1])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, train_path[2])

# In[] Split the pituitary dataset to Test 10%, Train 70% and Val 20% directories 

train_end_index = int(0.7 * len(pituitary))
val_end_index = int(0.2 *  len(pituitary)) + train_end_index

dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Independent/"
train_path = [dest_path+"Train/pituitary/", dest_path + "Val/pituitary/", dest_path + "Test/pituitary/"]

# split tumor dataset into training, testing and val
for index, path in enumerate(pituitary): 
    # decide to which dataset group the item should go
    if index <= train_end_index:
        # move the image to the training directory
        shutil.copy(path, train_path[0])
    elif index <= val_end_index:
        # copy and paste the image to the Val directory
        shutil.copy(path, train_path[1])
    else:
        # the remaining scans will go to test directory        
        shutil.copy(path, train_path[2])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 06:48:17 2024

@author: dagi
"""

import os 
import cv2
from tqdm import tqdm

# %% set route and create list for mask file
root_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Segmentation/"
train_dir_mask = root_path + "Train/Masks/"
val_dir_mask = root_path + "Val/Masks/"
test_dir_mask = root_path + "Test/Masks/"

paths_mask = [train_dir_mask, val_dir_mask, test_dir_mask]

train_dir = root_path + "Train/Images/"
val_dir = root_path + "Val/Images/"
test_dir = root_path + "Test/Images/"

paths = [train_dir, val_dir, test_dir]

# %% create list from the directories
train_lst_mask = os.listdir(train_dir_mask)
val_lst_mask = os.listdir(val_dir_mask)
test_lst_mask = os.listdir(test_dir_mask)

mask_lst_mask = [train_lst_mask, val_lst_mask, test_lst_mask]

train_lst = os.listdir(train_dir)
val_lst = os.listdir(val_dir)
test_lst = os.listdir(test_dir)

mask_lst = [train_lst, val_lst, test_lst]
# %% Set the dimension for the new size
IMG_HEIGHT = 256 
IMG_WIDTH = 256 

new_dim = (IMG_HEIGHT, IMG_WIDTH)
current_dataset = ["Cleaning the test set", "Cleaning the training set", "Cleaning the val set" ]
# %% Iterate through train_lst and resize mask file which happen to differ from 256x256
for index, current_lst in enumerate(mask_lst_mask):
    current_path = paths_mask[index]
    for imgs in tqdm(current_lst, desc=current_dataset[index], leave=False):
        # set the full path of the img
        full_path = current_path + imgs
        # load the img
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        # check the dimension of the img if it's the desired size no need to re-size it
        if img.shape != new_dim:
            # resize the loaded imgs to desired size
            img_resized = cv2.resize(img, dsize = new_dim, interpolation = cv2.INTER_CUBIC)
            # rewrite the resized image on the same directory under the same name
            cv2.imwrite(full_path, img_resized)
            break 

# %% Iterate through train_lst and resize mri file which happen to differ from 256x256
for index, current_lst in enumerate(mask_lst):
    current_path = paths[index]
    for imgs in tqdm(current_lst, desc=current_dataset[index], leave=False):
        # set the full path of the img
        full_path = current_path + imgs
        # load the img
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        # check the dimension of the img if it's the desired size no need to re-size it
        if img.shape != new_dim:
            # resize the loaded imgs to desired size
            img_resized = cv2.resize(img, dsize = new_dim, interpolation = cv2.INTER_CUBIC)
            # rewrite the resized image on the same directory under the same name
            cv2.imwrite(full_path, img_resized)

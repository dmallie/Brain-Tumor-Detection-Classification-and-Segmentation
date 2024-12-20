#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:16:57 2024

@author: dagi
"""

import os 
import cv2 
from tqdm import tqdm 

# %% Set path to the source and destination
src_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Segmentation/"
dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Contrast_enhanced/"

# %% Walk through the directory structure and store the pathes in the list
src = []
# Walk through the directory structure
for root, dirs, files in os.walk(src_path):
    for folder in dirs:
        path = root + folder + "/" + "images/"
        # we collect only mri scans not masks as they don't need to be enhanced
        if "/masks/" in path:
            continue
        mri_scans = os.listdir(path)
        for each_scan in tqdm(mri_scans, desc="equalization", leave=False):
            # set the full path of the file
            full_path = path + each_scan
            # read the file 
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            # apply histogram equalization
            equalized_img = cv2.equalizeHist(img)
            # set destination path. Path is same other than those two folder names
            dest_path = full_path.replace("Segmentation", "Contrast_enhanced").replace("images", "Images") 
            # save image
            cv2.imwrite(dest_path, equalized_img)

            break                 
# %% create list from directory
src_lst = os.listdir(src_path)

#%% go through the list one by one and perform contrast enhancement
for img_file in src_lst:
    # get the full path
    full_path = src_path + img_file 
    # load teh image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # do histogram equalization
    img_hist = cv2.equalizeHist(img)
    # save the enhanced image at  the right destination
    save_img = dest_path + img_file
    cv2.imwrite(save_img, img_hist)
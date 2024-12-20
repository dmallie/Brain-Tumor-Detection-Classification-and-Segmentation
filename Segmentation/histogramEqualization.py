#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:50:21 2024

@author: dagi
"""
import os 
import cv2 
# %%
root = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Refined/Train/"
img_path = root + "images/"
save_path = root + "image_enhanced/"

# %%
path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Refined/"
val_path = path + "Val/images/"
test_path = path + "Test/images/"

#%% create list 
img_lst = os.listdir(img_path)
val_lst = os.listdir(val_path)

save_val = path + "Val/image_enhanced/"
# %%
for img_file in img_lst:
    # full path
    full_path = img_path + img_file 
    #load the image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # apply histogram equalization
    equalized_img = cv2.equalizeHist(img)
    # set destination path
    dest_path = save_path + img_file 
    # save image
    cv2.imwrite(dest_path, equalized_img)

# %%
for img_file in val_lst:
    # full path
    full_path = val_path + img_file 
    #load the image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # apply histogram equalization
    equalized_img = cv2.equalizeHist(img)
    # set destination path
    dest_path = save_val + img_file 
    # save image
    cv2.imwrite(dest_path, equalized_img)


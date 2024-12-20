#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 12:18:12 2024

@author: dagi
"""
import numpy as np
import cv2
import os 
from utils import (average_difference, precision_score_segmentation,
                    dice_loss, intersection_over_union)
from tqdm import tqdm 

# In[] Set Route path to data directories
root_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Segmentation/Val/"
unet_path = root_path + "unet_mask/"
unet_plus_path = root_path + "unet_plus_plus/"
unet_attention_path = root_path + "unet_attention/"
ensembled_path = root_path + "ensembled/"
mask_path = root_path + "Masks/"

# %% Create list from the given directories
mask_lst = os.listdir(mask_path)

# %% name the output text file
output = "model_performance_2.txt"

# In[] Instantiation of list object to store the values of mask comparison
# Dice Coffecient
dice_unet = []
dice_unet_plus = []
dice_unet_attention = []
dice_ensembled = []
# Intersection Over Union
iou_unet = []
iou_unet_plus = []
iou_unet_attention = []
iou_ensembled = []
# Precision metrics
precision_unet = []
precision_unet_plus = []
precision_unet_attention = []
precision_ensembled = []

# Average Difference
ad_unet = []
ad_unet_plus = []
ad_unet_attention = []
ad_ensembled = []


detected_unet = 0
detected_unet_plus = 0
detected_unet_attention = 0
detected_ensembled = 0

# In[] Calculate the evaluation matrices
for mask_file in tqdm(mask_lst, desc="Evaluating the quality of output", leave=False):
    # set full path for each directories
    full_path_mask = mask_path + mask_file
    full_path_unet = unet_path + mask_file 
    full_path_unet_plus = unet_plus_path + mask_file 
    full_path_unet_attention = unet_attention_path + mask_file
    full_path_ensembled = ensembled_path + mask_file 
    
    # load the images
    mask = cv2.imread(full_path_mask, cv2.IMREAD_GRAYSCALE)
    unet = cv2.imread(full_path_unet, cv2.IMREAD_GRAYSCALE)
    unet_plus = cv2.imread(full_path_unet_plus, cv2.IMREAD_GRAYSCALE)
    unet_attention = cv2.imread(full_path_unet_attention, cv2.IMREAD_GRAYSCALE)
    ensembled = cv2.imread(full_path_ensembled, cv2.IMREAD_GRAYSCALE)
    
    # Calculate the dice loss
    dice_unet.append(1- dice_loss( unet, mask ))
    dice_unet_plus.append(1 - dice_loss( unet_plus, mask ))
    dice_unet_attention.append(1 - dice_loss(unet_attention, mask))
    dice_ensembled.append(1 - dice_loss( ensembled, mask ))
    
    # Calculate the loss on intersection over union
    iou_unet.append(intersection_over_union(unet, mask ))
    iou_unet_plus.append(intersection_over_union(unet_plus, mask ))
    iou_unet_attention.append(intersection_over_union(unet_attention, mask))
    iou_ensembled.append(intersection_over_union(ensembled, mask ))
    
    # Calculate the precision between teh generated mask and mask
    precision_unet.append(precision_score_segmentation(mask, unet))
    precision_unet_plus.append(precision_score_segmentation(mask, unet_plus))
    precision_unet_attention.append(precision_score_segmentation(mask, unet_attention))
    precision_ensembled.append(precision_score_segmentation(mask, ensembled))
    
    # calculate the averae difference between the mask and prediction
    ad_unet.append(average_difference(unet, mask))
    ad_unet_plus.append(average_difference(unet_plus, mask))
    ad_unet_attention.append(average_difference(unet_attention, mask))
    ad_ensembled.append(average_difference(ensembled, mask))
    # count the files in which the models detect tumor
    if np.any(unet):
        detected_unet += 1
        
    if np.any(unet_plus):
        detected_unet_plus += 1
    
    if np.any(unet_attention):
        detected_unet_attention += 1
    
    if np.any(ensembled):
        detected_ensembled += 1

# In[] Save the out put of yolo8n
with open(output, "w") as f:
    # summary of the model output
    f.write(f'Average Dice Coefficient: U-Net {np.mean(dice_unet):.3f}\t U-Net++ {np.mean(dice_unet_plus):.3f}\t U-Net_attention {np.mean(dice_unet_attention):.3f}\t Ensembled {np.mean(dice_ensembled):.3f}\n')
    f.write(f'Average IoU: U-Net {np.mean(iou_unet):.3f}\t U-Net++ {np.mean(iou_unet_plus):.3f}\t U-Net_attention {np.mean(iou_unet_attention):.3f}\t Ensembled {np.mean(iou_ensembled):.3f}\n')
    f.write(f'Average Precision: U-Net {np.mean(precision_unet):.3f}\t U-Net++ {np.mean(precision_unet_plus):.3f}\t  U-Net_attention {np.mean(precision_unet_attention):.3f}\t Ensembled {np.mean(precision_ensembled):.3f}\n')
    f.write(f'Average of Average Difference (AD): U-Net {np.mean(ad_unet):.3f}%\t U-Net++ {np.mean(ad_unet_plus):.3f}%\t U-Net_attention {np.mean(ad_unet_attention):.3f}%\t Ensembled {np.mean(ad_ensembled):.3f}%\n')
    f.write(f'Tumors detected by: U-Net is {detected_unet}/{len(mask_lst)}\t U-Net++ is {detected_unet_plus}/{len(mask_lst)}\t U-Net_attention is {detected_unet_attention}/{len(mask_lst)}\t Ensembled is {detected_ensembled}/{len(mask_lst)}\n')
    # Output of Each mask file comparison
    f.write('###########################################\n')

f.close()     

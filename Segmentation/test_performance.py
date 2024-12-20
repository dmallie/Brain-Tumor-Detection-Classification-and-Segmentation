#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 08:49:48 2024
Objective:
    - Using the test set we evaluate the performance of the models
    - we load the unet model
    - we load teh unet_plus_plus model
    - each mri scan in the test dataset is passes through both models
    - The output of unet_model will be saved in unet_mask
    - the output of unet_plus_plus will be saved in unet_plus_plus directory
    - tthe intersection of the two model output will be saved in ensemble directory
@author: dagi
"""
import sys
# Add the directory containing UNET_architecture.py to the Python path
# sys.path.append('/home/dagi/Documents/PyTorch/MIP/final_project_3/Task_4')
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import DataLoader
from UNET_architecture import UNet
import cv2
from custom_dataset import UNetDataset
from tqdm import tqdm
import numpy as np 
from UNet_plus_plus import UNetPlusPlus
from utils import calculate_mean_std
from timeit import default_timer as timer
from Attention_unet_architecture import AttentionUNet
from scipy.ndimage import label
from torchsummary import summary

# In[] Routing path to the Source directory
src_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Segmentation/Val/Images/"
mask_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Segmentation/Val/Masks/"
src_list = os.listdir(src_path)

# In[] Destination folder
root_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Thesis Project/Segmentation/Val/"
dest_unet = root_path + "unet_mask/"
dest_unet_plus_plus = root_path + "unet_plus_plus/"
dest_unet_attention = root_path + "unet_attention/"
dest_ensembled = root_path + "ensembled/"

# In[] Setting Hyperparameters
WIDTH = 256 
HEIGHT = 256 
OUTPUT_SHAPE = 1
BATCH  = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% Calculate the mean and standard deviation of the dataset
mean, std = calculate_mean_std(src_path, HEIGHT, WIDTH)
print(f"mean of test dataset: {mean}\t std of test dataset: {std}\n")

# In[] Set Transform Functions
transform_fn = A.Compose([
                        A.Resize(height=HEIGHT, width=WIDTH),
                        # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                        A.Normalize(
                            mean = [mean],
                            std = [std],
                            max_pixel_value = 1.0),
                        ToTensorV2(),
                        ])
                        
# In[] Setting the dataset and dataloader
dataset = UNetDataset(src_path, mask_path, transform_fn)
data_loader = DataLoader(dataset = dataset,
                              batch_size = BATCH,
                              shuffle = True,
                              num_workers = 4,
                              pin_memory = True)

# In[] Load the unet model 
model_path_1 = "unet.pth"
model_unet = UNet(in_channels= 1, num_classes=OUTPUT_SHAPE)

# %% load the saved dict
saved_state_dict = torch.load(model_path_1, weights_only=True)
# load teh state_dict into the model
model_unet.load_state_dict(saved_state_dict)

# In[] Load the unet++ model 
model_path_2 = "unet++.pth"
model_unet_plus_plus = UNetPlusPlus(in_channels= 1, num_classes=OUTPUT_SHAPE)

# %% load the saved dict
saved_state_dict = torch.load(model_path_2, weights_only=True)
# load teh state_dict into the model
model_unet_plus_plus.load_state_dict(saved_state_dict)

# In[] Load the unet_attention model 
model_path_3 = "unet_attention.pth"
model_attention = AttentionUNet(in_channels= 1, num_classes=OUTPUT_SHAPE)

# %% load the saved dict
saved_state_dict = torch.load(model_path_3, weights_only=True)
# load teh state_dict into the model
model_attention.load_state_dict(saved_state_dict)

# %% move the models to cuda device
model_unet = model_unet.to(DEVICE)
model_unet_plus_plus = model_unet_plus_plus.to(DEVICE)
model_attention = model_attention.to(DEVICE)

# %% set the models to evaluation mode
model_unet.eval()
model_unet_plus_plus.eval()
model_attention.eval()

# %% iterate through each image file and perform segmentation
for img_name in tqdm(src_list, desc="Testing", leave=False):
    # set the full path of the image
    full_path = src_path + img_name
    # set the destination path
    if img_name.endswith(".jpg"):
        dest_path_unet = dest_unet + img_name.replace(".jpg", ".png")
        dest_path_unet_plus = dest_unet_plus_plus + img_name.replace(".jpg", ".png")
        dest_path_unet_attention = dest_unet_attention + img_name.replace(".jpg", ".png")
        dest_path_ensembled = dest_ensembled + img_name.replace(".jpg", ".png")
    else:
        dest_path_unet = dest_unet + img_name.replace(".tif", "_mask.tif")
        dest_path_unet_plus = dest_unet_plus_plus + img_name.replace(".tif", "_mask.tif")
        dest_path_unet_attention = dest_unet_attention + img_name.replace(".tif", "_mask.tif")
        dest_path_ensembled = dest_ensembled + img_name.replace(".tif", "_mask.tif")
        
    ##########################################################
    # load the image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # convet img to numpy array and standardize the values
    img_standardized = np.array(img) / 255.0
    # Transform the image
    img_transformed = transform_fn(image = img_standardized)["image"].unsqueeze(0).to(DEVICE) # add batch dimension
    
    ##############################################################
    # Perform the evaluation task
    with torch.no_grad():
        # Forward pass
        unet_output = model_unet(img_transformed)
        prediction_unet = torch.sigmoid(unet_output)
        # convert probabilites to 0 or 1
        binary_unet = (prediction_unet > 0.5).float()
        # Forward pass
        unet_plus_output = model_unet_plus_plus(img_transformed)
        prediction_unet_plus_plus = torch.sigmoid(unet_plus_output)
        # convert probabilites to 0 or 1
        binary_unet_plus_plus = (prediction_unet_plus_plus > 0.5).float()
        # Forward pass
        attention_output = model_attention(img_transformed)
        prediction_unet_attention = torch.sigmoid(attention_output)
        # convert probabilites to 0 or 1
        binary_unet_attention = (prediction_unet_attention > 0.5).float()
    # ensemble learning
    # ensemble_probabilities = (prediction_unet + prediction_unet_plus_plus + prediction_unet_attention)/3
    ensemble_probabilities = 0.338 *  prediction_unet + 0.321 * prediction_unet_plus_plus + 0.341* prediction_unet_attention
    # ensemble_probabilities = 0 *  prediction_unet + 0 * prediction_unet_plus_plus + 0 * prediction_unet_attention
    # prediction_ensemble = torch.sigmoid(ensemble_output)
    # convert probabilites to 0 or 1
    binary_ensemble = (ensemble_probabilities > 0.5).float()
    ##############################################################
    # move the binary predictions to cpu and numpy
    mask_unet = binary_unet.squeeze(0).squeeze(0).cpu().detach().numpy()
    mask_unet_plus_plus = binary_unet_plus_plus.squeeze(0).squeeze(0).cpu().detach().numpy()
    mask_unet_attention = binary_unet_attention.squeeze(0).squeeze(0).cpu().detach().numpy()
    mask_ensemble = binary_ensemble.squeeze(0).squeeze(0).cpu().detach().numpy()
    # convert mask to uint8 and values between 0 and 255
    mask_unet = (mask_unet * 255).astype(np.uint8)
    mask_unet_plus_plus = (mask_unet_plus_plus *  255).astype(np.uint8)
    mask_unet_attention = (mask_unet_attention * 255).astype(np.uint8)
    mask_ensemble = (mask_ensemble *  255).astype(np.uint8)
    
    # mask_ensemble = np.clip(mask_ensemble, 0, 255).astype(np.uint8)
    ####### REFINE THE OUTPUT #############################################
    # If ensemble didn't detect any tumor then copy the one that detects any
    if not np.any(mask_ensemble): # True means one is detected
        mask_fusion = mask_unet + mask_unet_plus_plus + mask_unet_attention
        mask_ensemble = np.clip(mask_fusion, 0, 255) # trim values >= 255 to 255 and <= 0 to 0
    # Check whether there are two or more disconnected balls in the ensemble png
    # Perform connected components labeling using scipy
    labeled_image, num_balls = label(mask_ensemble)
    if num_balls > 1:
        # get the centroid coordinate of the balls from ensemble and all others
        sizes = []
        centroid = []
        for i in range(1, num_balls + 1):
            coords = np.argwhere(labeled_image == i)
            centroid = coords.mean(axis=0)  # Calculate centroid
            size = (labeled_image == i).sum()
            sizes.append(size)
        # find the largest component
        largest_component = np.argmax(sizes) + 1
        # redraw the ensemble mask with only the largest component
        mask_ensemble = (255*(labeled_image == largest_component)).astype(np.uint8)
        
            # print(f"Two balls detected in {img_name}")

    # save the mask file
    cv2.imwrite(dest_path_unet, mask_unet)
    cv2.imwrite(dest_path_unet_plus, mask_unet_plus_plus)
    cv2.imwrite(dest_path_unet_attention, mask_unet_attention)
    cv2.imwrite(dest_path_ensembled, mask_ensemble)
    
# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
# %%
print(f"Number of parameters in U-Net model is: {count_parameters(model_unet):,}")

print(f"Number of parameters in U-Net++ model is: {count_parameters(model_unet_plus_plus):,}")
print(f"Number of parameters in Attention U-Net model is: {count_parameters(model_attention):,}")


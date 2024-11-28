#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 07:34:13 2024
Objective:
    - Train ResNet50 to differentiate MRI with tumor with that of no-tumor
@author: dagi
"""
import os 
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
import torch
from torch import nn 
from torchinfo import summary
from torchvision import models 
from timeit import default_timer as timer
from loops import main_loop
import matplotlib.pyplot as plt
import itertools
import torch.optim as optim 
from utils import calculate_mean_std

# In[] Function/method counts the total number of files in the provided directory
def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)  # Add the number of files in each directory
    return file_count

# In[] Set route path for the data
rootPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Independent/"
trainingPath = rootPath + "Train/"
valPath = rootPath + "Val/"

no_training_data = count_files(trainingPath)
no_val_data = count_files(valPath) 

# In[] Set Hyperparameter
WIDTH = 256
HEIGHT = 256
BATCH = 2**6
OUTPUT_SHAPE = 4 #  0: glioma, 1: meningioma, 2: no_tumor and 3: pituitary
EPOCH = 100

# In[] Calculate the mean and standard deviation for each dataset
mean_train, std_train = calculate_mean_std(trainingPath)
mean_val, std_val = calculate_mean_std(valPath)

# In[] Set transform function
transform_train = transforms.Compose([
                        # Resize the image to fit the model
                        transforms.Resize(size=(HEIGHT, WIDTH)),  
                        transforms.Grayscale(num_output_channels=1),
                        transforms.RandomHorizontalFlip(p = 0.5),# Randomly flip some images horizontally with probability of 50%
                        transforms.RandomVerticalFlip(p=0.2),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                        # Convert image to tensor object
                        transforms.ToTensor(),
                        # Normalize the 3-chanelled image
                        transforms.Normalize(mean=[mean_train], std = [std_train]),
                ])
transform_val = transforms.Compose([
                        # Resize the image to fit the model
                        transforms.Resize(size=(HEIGHT, WIDTH)),
                        # Convert image to grayscale 
                        transforms.Grayscale(num_output_channels=1),
                        # Convert image to tensor object
                        transforms.ToTensor(),
                        # Normalize the 3-channeled tensor object
                        transforms.Normalize(mean=[mean_val], std = [std_val])
                ])

# In[] Setup the dataset
train_dataset = datasets.ImageFolder(
                    root = trainingPath,
                    transform = transform_train)
val_dataset = datasets.ImageFolder(
                    root = valPath,
                    transform = transform_val)

# In[] Setup the DataLoader
train_dataloader = DataLoader(
                        dataset = train_dataset,
                        batch_size = BATCH,
                        num_workers = 4,
                        shuffle = True,
                        pin_memory = True)
val_dataloader = DataLoader(
                        dataset = val_dataset,
                        batch_size = BATCH,
                        num_workers = 4,
                        shuffle = False,
                        pin_memory = True)

# In[] Step 5: import and instantiate ResNet50
weights = models.ResNet50_Weights.DEFAULT 
model = models.resnet50(weights = weights) 

# In[] Modify the first layer to receive grayscale images
# Get the pretrained model's first layer
original_conv1 = model.conv1

# Create a new Conv2d layer with 1 input channel instead of 3
model.conv1 = nn.Conv2d(
    in_channels=1,               # Set to 1 for grayscale images
    out_channels=original_conv1.out_channels,
    kernel_size=original_conv1.kernel_size,
    stride=original_conv1.stride,
    padding=original_conv1.padding,
    bias=original_conv1.bias
)

# In[] To inspect Model Info
summary(model = model,
        input_size = (BATCH, 1, HEIGHT, WIDTH),
        col_names = ["input_size", "output_size", "trainable"],
        col_width = 20,
        row_settings = ["var_names"])

# In[10] Modifying the model to meet input and output criteria
# 1. Freezing the trainablility of base model
for params in model.parameters():
    params.requires_grad = False
    
# In layer4 Bottleneck[2] set the parameter to True
for params in model.layer4.parameters():
    params.requires_grad = True 

# modify the output shape and connection layer of the model
model.fc = nn.Sequential(
    nn.Dropout(p = 0.2, inplace = True),
    nn.Linear(in_features = 2048,
              out_features = 512,              
              bias = True),
    nn.ReLU(),
    nn.Linear(in_features=512,
              out_features=OUTPUT_SHAPE)
    )
# In[9] Model Info after configuration
summary(model = model,
        input_size = (BATCH, 1, HEIGHT, WIDTH),
        col_names = ["input_size", "output_size", "trainable"],
        col_width = 20,
        row_settings = ["var_names"])

# In[] Step 6: Setup the loss function and Optimizer function & class imbalance handling
device = "cuda" if torch.cuda.is_available() else "cpu"

optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode = 'min',
                                                 factor = 0.1,
                                                 patience = 10)
loss_fn = nn.CrossEntropyLoss()
# In[] Step 7: Start the training loop
start_time = timer()

# Setup training and save the results
accuracy_list_training, accuracy_list_val, loss_list_training, loss_list_val = main_loop( model,
                                         train_dataloader,
                                         val_dataloader,
                                         optimizer,
                                         criterion = loss_fn,
                                         epochs = EPOCH,
                                         scheduler = scheduler,
                                         save_path = "resnet_50.pth")

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# In[]
accuracy_list_training = [0.84, 0.92, 0.95, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99,
                          0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
                          0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.00]
accuracy_list_val = [0.91, 0.93, 0.93, 0.95, 0.96, 0.97, 0.97, 0.97, 0.97, 0.96, 0.98, 0.98, 0.98, 0.98, 0.97,
                     0.98, 0.98, 0.97, 0.98, 0.98, 0.99, 0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 0.98, 0.98, 0.98,
                     0.98, 0.98, 0.98, 0.98, 0.99, 0.98]
loss_list_training = [0.4376, 0.2196, 0.1487, 0.1220, 0.0983, 0.0784, 0.0734, 0.0604, 0.0582, 0.0552, 0.0484,
                      0.0496, 0.0423, 0.0372, 0.0386, 0.0391, 0.0316, 0.0314, 0.0342, 0.0264, 0.0284, 0.0267,
                      0.0273, 0.0217, 0.0266, 0.0227, 0.0202, 0.0231, 0.0213, 0.0199, 0.0230, 0.0176, 0.0185,
                      0.0180, 0.0175, 0.0145]
loss_list_val = [0.2495, 0.1962, 0.1916, 0.1439, 0.1093, 0.0850, 0.0956, 0.0896, 0.0793, 0.1285, 0.0828, 0.0649,
                 0.0749, 0.0725, 0.0885, 0.0714, 0.0752, 0.0925, 0.0572, 0.0743, 0.0521, 0.0772, 0.0706, 0.0654,
                 0.0703, 0.0545, 0.0551, 0.0605, 0.0583, 0.0530, 0.0486, 0.0555, 0.0626, 0.0581, 0.0562, 0.0665]

# In[] Step 8: Plot the performance of the model
x = list(itertools.chain(range(0, len(accuracy_list_training))))
plt.plot(x, accuracy_list_training, label = "Training Performance")
plt.plot(x, accuracy_list_val, label = "Validation Performance")
plt.legend()
plt.show()

# In[] Step 9: Plot the loss function of the model
# train_loss = [100 - acc for acc in train_accuracy]
# val_loss = [100 - acc for acc in val_accuracy]

plt.plot(x, loss_list_training, label = "Training Loss")
plt.plot(x, loss_list_val, label = "Validation Loss")
plt.legend()
plt.show()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:43:06 2024
Objective:
    - Train VGG-19 to differentiate MRI with tumor with that of no-tumor
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

# In[] Function/method counts the total number of files in the provided directory
def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)  # Add the number of files in each directory
    return file_count

# In[] Set route path for the data
rootPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Ensemble/Model_1/"
trainingPath = rootPath + "Train/"
valPath = rootPath + "Val/"

no_training_data = count_files(trainingPath)
no_val_data = count_files(valPath) 

# In[] Set Hyperparameter
WIDTH = 256
HEIGHT = 256
BATCH = 2**7
OUTPUT_SHAPE = 1 #  0: healthy and 1: tumor
EPOCH = 100

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
                        transforms.Normalize(mean=[0.18206], std = [0.16517]),
                ])
transform_val = transforms.Compose([
                        # Resize the image to fit the model
                        transforms.Resize(size=(HEIGHT, WIDTH)),
                        # Convert image to grayscale 
                        transforms.Grayscale(num_output_channels=1),
                        # Convert image to tensor object
                        transforms.ToTensor(),
                        # Normalize the 3-channeled tensor object
                        transforms.Normalize(mean=[0.18206], std = [0.16517])
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
weights = models.VGG19_Weights.DEFAULT 
model = models.vgg19(weights = weights) 

# In[] Modify the first layer to receive grayscale images
# Get the pretrained model's first layer
input_layer = model.features[0]

# Create a new Conv2d layer with 1 input channel instead of 3
model.features[0] = nn.Conv2d(
                        in_channels=1,               # Set to 1 for grayscale images
                        out_channels=input_layer.out_channels,
                        kernel_size=input_layer.kernel_size,
                        stride=input_layer.stride,
                        padding=input_layer.padding,

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

# Set the trainablility of denseblock4 True 
# Unfreeze layers starting from denselayer8
unfreeze = False
for name, layer in model.features.named_children():
    if name == "30":
        unfreeze = True  # Start unfreezing from here

    if unfreeze:
        for params in layer.parameters():
            params.requires_grad = True
# unfreeze all classifier parameters
for params in model.classifier.parameters():
    params.requires_grad = True 

# modify the output shape and connection layer of the model
model.classifier[6] = nn.Linear( 
                        in_features = 4096,
                        out_features = OUTPUT_SHAPE,
                        bias = True)

    
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
loss_fn = nn.BCEWithLogitsLoss()
# In[] Step 7: Start the training loop
start_time = timer()

# Setup training and save the results
train_accuracy, val_accuracy = main_loop( model,
                                         train_dataloader,
                                         val_dataloader,
                                         optimizer,
                                         criterion = loss_fn,
                                         epochs = EPOCH,
                                         scheduler = scheduler,
                                         save_path = "vgg_19.pth")

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# In[] Step 8: Plot the performance of the model
x = list(itertools.chain(range(0, len(train_accuracy))))
plt.plot(x, train_accuracy, label = "Training Performance")
plt.plot(x, val_accuracy, label = "Validation Performance")
plt.legend()
plt.show()

# In[] Step 9: Plot the loss function of the model
train_loss = [100 - acc for acc in train_accuracy]
val_loss = [100 - acc for acc in val_accuracy]

plt.plot(x, train_loss, label = "Training Loss")
plt.plot(x, val_loss, label = "Validation Loss")
plt.legend()
plt.show()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:10:44 2024
Objective:
    - Using ensemble technique and test dataset, we evaluate the performance of the 
    trained models
@author: dagi
"""
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn 
from utils import model_accuracy, calculate_mean_std
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

# In[] Function/method counts the total number of files in the provided directory
def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)  # Add the number of files in each directory
    return file_count

# In[] Set route path for the data
testPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Ensemble/Model_2/Test/"

size_test_data = count_files(testPath)

# In[] Setting Hyperparameter
WIDTH = 256
HEIGHT = 256
OUTPUT_SHAPE = 1
BATCH = 1

# In[] Calculating the mean and std of test dataset
mean, std = calculate_mean_std(testPath)

# In[] Set transform function
transform_fn = transforms.Compose([
                        # Resize the image to fit the model
                        transforms.Resize(size=(HEIGHT, WIDTH)),
                        # Convert image to grayscale 
                        transforms.Grayscale(num_output_channels=1),
                        # Convert image to tensor object
                        transforms.ToTensor(),
                        # Normalize the tensor object
                        transforms.Normalize(mean=[mean], std = [std]),
                ])

# In[] Setting up the dataset and dataloader
test_dataset = datasets.ImageFolder(
                    root = testPath,
                    transform = transform_fn,
                    target_transform=None)


test_dataloader = DataLoader(
                        dataset = test_dataset,
                        batch_size = BATCH,
                        num_workers = 4,
                        shuffle = False,
                        pin_memory = True)

# In[] Set paths for the different models we have
resnet_path = "resnet_50.pth"
densenet_path = "densenet.pth"
vgg_path = "vgg_19.pth"
efficient_path = "efficient_net.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss() 

# In[] Load the model
resnet = torch.load(resnet_path, weights_only=False)
densenet = torch.load(densenet_path, weights_only=False)
vgg = torch.load(vgg_path, weights_only=False)
efficient = torch.load(efficient_path, weights_only=False)

resnet = resnet.to(device)
densenet = densenet.to(device)
vgg = vgg.to(device)
efficient = efficient.to(device)

# In[] Helper function for model prediction and thresholding
def prediction_model(model, image, threshold=0.5):
    predictions = model(image)

    # convert predictions to probabilities
    probabilities = F.softmax(predictions, dim=1)
    output = torch.argmax(probabilities, dim=1)
    return output, predictions

# In[] Prepare models and evaluation mode
resnet.eval()
densenet.eval()
vgg.eval()
efficient.eval()

true_labels = []
resnet_labels = []
densenet_labels = []
vgg_labels = []
efficient_labels = []

running_loss = 0.0

with torch.no_grad():
    for index, (image, label) in enumerate(tqdm(test_dataloader, desc="Evaluating the model", leave=False)):
        image, label = image.to(device), label.to(device)

        # prediction by resnet model
        resnet_pred, resnet_logits = prediction_model(resnet, image)
        resnet_labels.append(resnet_pred.item())
        
        # prediction by the densenet Model
        densenet_pred, densenet_logits = prediction_model(densenet, image)
        densenet_labels.append(densenet_pred.item())
        
        # predictions by the VGG Model 
        vgg_pred, vgg_logits = prediction_model(vgg, image)
        vgg_labels.append(vgg_pred.item())
        
        # predictions by the EfficientNet Model 
        efficient_pred, efficient_logits = prediction_model(efficient, image)
        efficient_labels.append(efficient_pred.item())
        
        # The actual label of mri scan
        true_labels.append(label.item())
    
# In[] majority wins vote among resnet_labels, densenet_labels, vgg_labels
vote_glioma = 0
vote_meningioma = 0
vote_pituitary = 0
ensemble_labels = []

for index in range(len(true_labels)):
    # count the votes
    votes = [
        resnet_labels[index],
        densenet_labels[index],
        vgg_labels[index],
        efficient_labels[index]
    ]
    
    # Count votes for each category
    vote_glioma = votes.count(0)
    vote_meningioma = votes.count(1)
    vote_pituitary = votes.count(2)
    
    # Determine the winner (majority vote)
    vote_count = np.array([vote_glioma, vote_meningioma, vote_pituitary])
    winner = np.argmax(vote_count)
    ensemble_labels.append(winner)

# In[] Calculate metrics after loop

ensemble_accuracy = model_accuracy(ensemble_labels, true_labels) / size_test_data
resnet_accuracy = model_accuracy(resnet_labels, true_labels) / size_test_data
densenet_accuracy = model_accuracy(densenet_labels, true_labels) / size_test_data
vgg_accuracy = model_accuracy(vgg_labels, true_labels) / size_test_data
efficient_accuracy = model_accuracy(efficient_labels, true_labels) / size_test_data
# Log performance
txt_lines = [
    "System's Performance\n",
    f"\t- Ensemble model accuracy: {model_accuracy(ensemble_labels, true_labels)}/{size_test_data} ie {ensemble_accuracy * 100:.2f}%\n",
    f"\t- Resnet model accuracy: {model_accuracy(resnet_labels, true_labels)}/{size_test_data} ie {resnet_accuracy * 100:.2f}%\n",
    f"\t- Densenet model accuracy: {model_accuracy(densenet_labels, true_labels)}/{size_test_data} ie {densenet_accuracy * 100:.2f}%\n",
    f"\t- VGG model accuracy: {model_accuracy(vgg_labels, true_labels)}/{size_test_data} ie {vgg_accuracy * 100:.2f}%\n",
    f"\t- Efficient model accuracy: {model_accuracy(efficient_labels, true_labels)}/{size_test_data} ie {efficient_accuracy * 100:.2f}%\n",
]
with open("Task_2 performance report.txt", 'w') as f:
    f.writelines(txt_lines)
# In[]
report_card = []
report_card.append("Index\t Truelabel \tEnsemble\t ResNet \tDenseNet \tVGG \tEfficientNet\n")
for index in range(len(true_labels)):
    line = f"{index}\t {true_labels[index]} \t\t{ensemble_labels[index]}\t\t {resnet_labels[index]} \t\t{densenet_labels[index]} \t\t{vgg_labels[index]} \t\t{efficient_labels[index]}\n"
    report_card.append(line)
    

with open("reportCard.txt", "w") as f:
    f.writelines(report_card)

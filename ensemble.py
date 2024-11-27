#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:55:17 2024
Objective:
    - Create dataset to test the performance of overall system.
    - Dataset is arranged in this manner:
        - no_tumor: brought from Test dataset in Task_1
        - glioma: brought from Test dataset in Task_2
        - meningioma: brought from Test dataset in Task_2
        - pituitary: brought from Test dataset in Task_2
@author: dagi
"""
import os 
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn 
from Task_2.utils import model_accuracy, calculate_mean_std
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
testPath = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/Ensemble/Overall/"

size_test_data = count_files(testPath)

# In[] Setting Hyperparameter
WIDTH = 256
HEIGHT = 256
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
resnet_1_path = "Task_1/resnet_50.pth"
densenet_1_path = "Task_1/densenet.pth"
vgg_1_path = "Task_1/vgg_19.pth"

resnet_2_path = "Task_2/resnet_50.pth"
densenet_2_path = "Task_2/densenet.pth"
vgg_2_path = "Task_2/vgg_19.pth"
efficient_path = "Task_2/efficient_net.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss() 

# In[] Load the model
resnet_1 = torch.load(resnet_1_path, weights_only=False)
densenet_1 = torch.load(densenet_1_path, weights_only=False)
vgg_1 = torch.load(vgg_1_path, weights_only=False)

resnet_2 = torch.load(resnet_2_path, weights_only=False)
densenet_2 = torch.load(densenet_2_path, weights_only=False)
vgg_2 = torch.load(vgg_2_path, weights_only=False)
efficient = torch.load(efficient_path, weights_only=False)

resnet_1 = resnet_1.to(device)
densenet_1 = densenet_1.to(device)
vgg_1 = vgg_1.to(device)

resnet_2 = resnet_2.to(device)
densenet_2 = densenet_2.to(device)
vgg_2 = vgg_2.to(device)
efficient = efficient.to(device)

# In[] Helper function for model prediction and thresholding
def prediction_model(model, image, threshold=0.5):
    predictions = model(image)

    # convert predictions to probabilities
    probabilities = F.softmax(predictions, dim=1)
    output = torch.argmax(probabilities, dim=1)
    return output, predictions

# In[] Helper function for model prediction and thresholding
def predict_with_threshold(model, image, threshold=0.5):
    logits = model(image).squeeze(dim=1)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).long()
    return predictions, logits

# In[] Prepare models and evaluation mode
resnet_1.eval()
densenet_1.eval()
vgg_1.eval()

resnet_2.eval()
densenet_2.eval()
vgg_2.eval()
efficient.eval()

true_labels = []
predicted_labels = []

running_loss = 0.0

with torch.no_grad():
    for index, (image, label) in enumerate(tqdm(test_dataloader, desc="Evaluating the model", leave=False)):
        image, label = image.to(device), label.to(device)

        # Task 1 prediction
        resnet_1_pred, resnet_1_logits = predict_with_threshold(resnet_1, image)
        densenet_1_pred, densenet_1_logits = predict_with_threshold(densenet_1, image)
        vgg_1_pred, vgg_1_logits = predict_with_threshold(vgg_1, image)
        
        # count the votes
        votes = [
            resnet_1_pred,
            densenet_1_pred,
            vgg_1_pred
        ]

        # Count votes for each category
        vote_no_tumor = votes.count(0)
        vote_tumor = votes.count(1)
        
        # Determine the winner (majority vote)
        winner = 0 if vote_no_tumor > vote_tumor else 1
        
        if winner == 0:  # No tumor found
            true_labels.append(label.item())
            predicted_labels.append(2)
            # running_loss += loss_fn(model_1_logits, model_1_label.float())
            continue  # Skip to the next image if no tumor is found

        # Task_2 Prediction
        # Let all the models in Task_2 predict the category of the image
        resnet_2_pred, resnet_2_logits = prediction_model(resnet_2, image)
        
        # prediction by the densenet Model
        densenet_2_pred, densenet_2_logits = prediction_model(densenet_2, image)
        
        # predictions by the VGG Model 
        vgg_2_pred, vgg_2_logits = prediction_model(vgg_2, image)
        
        # predictions by the EfficientNet Model 
        efficient_pred, efficient_logits = prediction_model(efficient, image)
        
        # vote on the prediction        
        votes = [resnet_2_pred, densenet_2_pred, vgg_2_pred, efficient_pred]
        # Count the vote on each category
        vote_glioma = votes.count(0)
        vote_meningioma = votes.count(1)
        vote_pituitary = votes.count(2)
        
        # count the vote and declare the winner
        vote_count = np.array([vote_glioma, vote_meningioma, vote_pituitary])
        winner = np.argmax(vote_count)
        if winner == 0:
            predicted_labels.append(winner)
        elif winner == 1:
            predicted_labels.append(winner)
        else:
            predicted_labels.append(3)
        # The actual label of mri scan
        true_labels.append(label.item())
    
# In[] Calculate metrics after loop

ensemble_accuracy = model_accuracy(predicted_labels, true_labels) / size_test_data
noTumor_predictions = predicted_labels.count(2)
pituitary_predictions = predicted_labels.count(3)
glioma_predictions = predicted_labels.count(0)
meningioma_predictions = predicted_labels.count(1) 

# Log performance
txt_lines = [
    "System's Performance\n",
    f"\t- Ensemble model accuracy: {model_accuracy(predicted_labels, true_labels)}/{size_test_data} ie {ensemble_accuracy * 100:.2f}%\n",
    f"\t- Accuracy on Healthy Dataset: {noTumor_predictions}/{true_labels.count(2)} ie  {noTumor_predictions/true_labels.count(2)*100:.2f}%\n",
    f"\t- Accuracy on Glioma Dataset: {glioma_predictions}/{true_labels.count(0)} ie  {glioma_predictions/true_labels.count(0)*100:.2f}%\n",
    f"\t- Accuracy on Meningioma Dataset: {meningioma_predictions}/{true_labels.count(1)} ie  {meningioma_predictions/true_labels.count(1)*100:.2f}%\n",
    f"\t- Accuracy on Pituitary Dataset: {pituitary_predictions}/{true_labels.count(3)} ie  {pituitary_predictions/true_labels.count(3)*100:.2f}%\n",
]
with open("Ensemble performance report.txt", 'w') as f:
    f.writelines(txt_lines)
# In[]


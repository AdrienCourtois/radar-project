# Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# General
import os
import sys
import time
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GitHub imports
sys.path.append('radar-project')
 
# Project
from utils.dataset import ImageDataset_8channels, ImageDataset
from utils.save import save_checkpoint, load_checkpoint
from loss import focale_loss, iou_loss
from attention_unet import AttentionUNet

# Bacthes 
batch_size = 2
true_batch_size = 10

# Load datasets
dataset = ImageDataset("AOI_3_Paris_Train/RGB-normalized/train", "AOI_3_Paris_Train/segmentation/train", transform=True)
dataset2 = ImageDataset("AOI_3_Paris_Train/RGB-normalized/test", "AOI_3_Paris_Train/segmentation/test", transform=False)

# Train test split
torch.manual_seed(42)
dataset_train, _ = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset)-int(0.8 * len(dataset))])
torch.manual_seed(42)
_, dataset_test = torch.utils.data.random_split(dataset2, [int(0.8 * len(dataset)), len(dataset)-int(0.8 * len(dataset))])

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

# Load model
model = AttentionUNet(filters=64, n_block=4, depth=6, channels=3)
model.initialize(focal_trick=True)
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load checkpoint
checkpoint_folder = "gdrive/My Drive/finetune-vienna-paris/"
resume = checkpoint_folder + "model_best.pth.tar"

model, optimizer, start_epoch, best_IOU, cpt = load_checkpoint(model, optimizer, resume)


########
# Main #
########
n_epochs = 10

for epoch in range(start_epoch, start_epoch + n_epochs):
    
    #########
    # Train #
    #########
    
    model.train()
    total_loss = 0
    train_metric = 0
    
    for idx, (x, y) in enumerate(train_loader):
        cpt += 1

        x, y = x.cuda(), y.cuda()
        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)
        
        # Loss
        loss = focale_loss(y_pred, y).mean()
        loss.backward()

        # Batch trick
        if idx * batch_size % true_batch_size == 0 and idx > 0:
            optimizer.step()
            optimizer.zero_grad()
    
        if idx % 100 == 0 and idx > 0:
            print("{} done on {} total.".format(idx, len(train_loader)))
        
        # Mask
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        # Metrics
        train_metric += iou_loss(y_pred, y).sum().item()
        total_loss += loss.item()

    # Optimizer            
    optimizer.step()
    optimizer.zero_grad()

    # Metrics
    training_IOU = train_metric/len(train_loader)
    print("Epoch {}: Training IOU: {}".format(epoch+1, training_IOU))

    
    ##############
    # Evaluation #
    ##############

    torch.cuda.empty_cache()
    model.eval()
    total_metric = 0

    for idx, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)

        # Mask
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        # Metric               
        total_metric += iou_loss(y_pred, y).sum().item()
    
    # Scores
    val_IOU = total_metric / len(test_loader)
    train_loss = total_loss / len(train_loader)
    print("Epoch {}: Training loss: {} ; Validation IOU: {}".format(epoch+1, train_loss, val_IOU))
    
    
    ########
    # Save #
    ######## 

    is_best = val_IOU > best_IOU
    best_IOU = max(val_IOU, best_IOU)
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_IOU': best_IOU,
            'optimizer': optimizer.state_dict(),
            'iter': cpt,
        }, is_best, checkpoint_folder)

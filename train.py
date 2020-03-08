import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

# GitHub imports
sys.path.append('radar-project')

from utils.dataset import ImageDataset
from attention_unet import AttentionUNet

# Dataset
batch_size = 10

dataset = ImageDataset("AOI_3_Paris_Train/RGB-normalized", "AOI_3_Paris_Train/segmentation", transform=False)

# Train test split
dataset_train, dataset_test = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset)-int(0.8 * len(dataset))])

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

model = AttentionUNet(filters=64, n_block=4, depth=6)
model.initialize()

model = model.cuda()

#########
# TRAIN #
#########

n_epochs = 10

smooth_l1_loss = nn.SmoothL1Loss()
def iou_loss(y_pred, y_true):
    return ((y_pred * y_true).sum()) / (y_pred.sum() + y_true.sum() - (y_pred * y_true).sum())
def dice_score(y_pred, y_true):
    return 2 * ((y_pred * y_true).sum()) / (y_pred.sum() + y_true.sum())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(n_epochs):

    # Train
    model.train()

    total_loss = 0
    n_train = 0

    for idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()

        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)

        print(y_pred.sum())
        
        crossentropy = (-0.8 * y * torch.log(y_pred + 1e-8) - 0.2 * (1-y) * torch.log(1-y_pred + 1e-8)).mean()
        loss = iou_loss(y_pred, y) + crossentropy
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_train += 1

    # Evaluation
    model.eval()

    total_metric = 0
    n_val = 0

    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)

        total_metric += dice_score(y_pred, y).item()
        n_val += 1
    
    print("Epoch {}, training loss {}, validation metric {}".format(epoch+1, total_loss / n_train, total_metric / n_val))
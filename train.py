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

from utils.dataset import ImageDataset
from attention_unet import AttentionUNet

# Dataset
batch_size = 1
true_batch_size = 10

dataset = ImageDataset("../AOI_3_Paris_Train/RGB-normalized", "../AOI_3_Paris_Train/segmentation", transform=False)

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
    num = (y_pred * y_true).sum((1,2,3))
    denom = (y_pred + y_true - y_pred * y_true).sum((1,2,3))

    if y_true.sum() == 0:
        num = ((1-y_pred) * (1-y_true)).sum((1,2,3))
        denom = (1-y_pred + 1-y_true - (1-y_pred) * (1-y_true)).sum((1,2,3))

    return num / denom

def dice_score(y_pred, y_true):
    num = 2 * (y_pred * y_true).sum((1,2,3))
    denom = (y_pred + y_true).sum((1,2,3))

    return 2 * num / (denom + 1e-10)

def focale_loss(y_pred, y_true, alpha=0.75, gamma=2):
    # y_pred: tensor [B, 1, H, W] binary predicted mask
    # y_true: tensor [B, 1, H, W] binary ground truth mask
    # --
    # Output: tensor [B] loss for each prediction of the batch


    m1 = y_true == 1
    m0 = y_true == 0

    p_t = torch.zeros(y_pred.size())
    alpha_t = torch.zeros(y_pred.size())

    if y_pred.is_cuda:
        alpha_t = alpha_t.cuda()
        p_t = p_t.cuda()

    p_t[m1] = y_pred[m1]
    p_t[m0] = 1-y_pred[m0]

    alpha_t[m1] = alpha
    alpha_t[m0] = 1-alpha

    L = - alpha_t * ((1 - p_t) ** gamma) * torch.log(p_t + 1e-10)

    return L.sum((1,2,3))


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(n_epochs):

    # Train
    model.train()
    # Reset Kevin truc
    #for m in model.modules():
    #    if isinstance(m, nn.BatchNorm2d):
    #        m.track_running_stats=True

    total_loss = 0
    train_metric = 0

    for idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()

        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)
        
        #crossentropy = (-0.8 * y * torch.log(y_pred + 1e-8) - 0.2 * (1-y) * torch.log(1-y_pred + 1e-8)).sum((1,2,3))
        loss = focale_loss(y_pred, y).mean() #crossentropy.mean() # + iou_loss(y_pred, y)

        if loss != loss:
            print("nnnnnnnannnnnnnnn: ", loss)

        loss.backward()

        if idx * batch_size % true_batch_size == true_batch_size-1:
            optimizer.step()
            optimizer.zero_grad()
    
        if idx % 100 == 0 and idx > 0:
            print("{} done on {} total.".format(idx, len(train_loader)))
        
        # Compute accuracy
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        train_metric += iou_loss(y_pred, y).sum().item()
        total_loss += loss.item()

    optimizer.step()
    optimizer.zero_grad()

    print("Epoch {}: Training IOU: {}".format(epoch+1, train_metric/len(train_loader)))

    torch.cuda.empty_cache()

    # Evaluation
    model.eval()
    # Trigger Kevin truc
    #for m in model.modules():
    #    if isinstance(m, nn.BatchNorm2d):
    #        m.track_running_stats=False

    total_metric = 0

    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        y_pred = model(x)
        y_pred = torch.sigmoid(y_pred)

        # Compute metric
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        total_metric += iou_loss(y_pred, y).sum().item()
    
    print("Epoch {}: Training loss: {} ; Validation IOU: {}".format(epoch+1, total_loss / len(train_loader), total_metric / len(test_loader)))

torch.save(model.state_dict(), "./training1.pth")
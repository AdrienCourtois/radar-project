"""
Inference of a pretrained Network.

This script takes multiple parameters as input:
- The link to the weights with the format .pth.tar
- The link to the image we want to test our network on.
The produced mask will be saved as `result.png` in the current directory.

Usage:
python eval.py weights.pth.tar image.png
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
from utils.optim import Lookahead, RAdam
import argparse
import os
###################
# Argument parser #
###################

parser = argparse.ArgumentParser(description='Usage: python eval.py weights.pth.tar image.png')
parser.add_argument("weights", help="The path to the weights")
parser.add_argument("image", help="The path to the image")

args = parser.parse_args()


###################
# Load the images #
###################

if parser.weights and os.path.isfile(parser.weights) and parser.image and os.path.isfile(parser.image):
    img = Image.open(parser.image)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    img = transform(img)

    if torch.cuda.is_available():
        img = img.cuda()
else:
    raise Exception("You must spectify a valid path to the weights and a valid image.")


#########
# Utils #
#########

def renormalize(x):
    x = x[0,:,:]
    x = x.permute((1,2,0)).cpu().detach().numpy()
    x = (x * 255).astype(np.uint8)

    x = np.repeat(x, 3, axis=2)

    return x


###################
# Loading weights #
###################

# Load model
model = AttentionUNet(filters=64, n_block=4, depth=6)

if torch.cuda.is_available():
    model = model.cuda()

# Load weights
print(f"Loading model: {resume}.")
checkpoint = torch.load(parser.weights)
model.load_state_dict(checkpoint['state_dict'])
best_IOU = checkpoint['best_IOU']
print(f"Loaded model: IOU: {best_IOU:0.3f}.")


######################
# Running evaluation #
######################

model.eval()

with torch.no_grad():
    img = img[None].cuda()
    y_pred = model(img)
    y_pred = torch.sigmoid(y_pred)

    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    mask = renormalize(y_pred)
    plt.imsave("result.png", mask)
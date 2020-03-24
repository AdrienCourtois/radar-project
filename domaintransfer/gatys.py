"""
Style Transfer using the Gatys paper.

This script takes two parameters as an input, the source (`source.png`), from which the style will be copied, and the target (`target.png`) the image to which the style will be applied. 
The resulting image will be saved as gatys_result.png.

Usage:
python gatys.py source.png target.png
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

"""
Architecture of the VGG19
VGG19-bn
1st layer: (0) Conv2d(64), (3) Conv2d(64), (6) Maxpool2d
2nd layer: (7) Conv2d(128), (10) Conv2d(128), (13) Maxpool2d
3rd layer: (14) Conv2d(256), (17) Conv2d(256), (20) Conv2d(256), (23) Conv2d(256), (26) Maxpool2d
4th layer: (27) Conv2d(512), (30) Conv2d(512), (33) Conv2d(512), (36) Conv2d(512), (39) Maxpool2d
4th layer: (40) Conv2d(512), (43) Conv2d(512), (46) Conv2d(512), (49) Conv2d(512), (52) Maxpool2d
"""

###################
# Argument parser #
###################

parser = argparse.ArgumentParser(description='Usage: python gatys.py source.png target.png')
parser.add_argument("source", help="The source image")
parser.add_argument("target", help="The target image")
parser.add_argument("-v", "--verbose", help="Enable it to see the evolution of the image and of the loss.", default=False)

args = parser.parse_args()


###################
# Load the images #
###################

if parser.source and os.path.isfile(parser.source) and parser.target and os.path.isfile(parser.target):
    img_style = Image.open(parser.source)
    img_content = Image.open(parser.target)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    img_style = transform(img_style)[None]
    img_content = transform(img_content)[None]

    if torch.cuda.is_available():
        img_style = img_style.cuda()
        img_content = img_content.cuda()
else:
    raise Exception("You must spectify a source and target image.")


#########
# Utils #
#########

def renormalize(x):
    if len(x.size()) == 4:
        x = x[0]
    
    x = x.permute((1,2,0)).cpu().detach().numpy()
    x = x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    x = x.clip(0,1)

    return x

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)


####################
# Model definition #
####################
vgg = torchvision.models.vgg19_bn(pretrained=True)

if torch.cuda.is_available():
    vgg = vgg.cuda()


###############
# Layers used #
###############
layers = list(list(vgg.children())[0].children())

style_layers = [layers[0], layers[7], layers[14], layers[27], layers[40]]
content_layers = [layers[0]] 

# Weights
style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
content_weights = [1e0]


##########################################
# Hooks to save the output of the layers #
##########################################

# Style hook
style_activation = [None for i in range(len(style_layers))]

def get_style_activation(idx):
    def hook(model, input, output):
        style_activation[idx] = output
    return hook

for idx, layer in enumerate(style_layers):
    layer.register_forward_hook(get_style_activation(idx))

# Content hook
content_activation = [None for i in range(len(content_layers))]

def get_content_activation(idx):
    def hook(model, input, output):
        content_activation[idx] = output
    return hook

for idx, layer in enumerate(content_layers):
    layer.register_forward_hook(get_content_activation(idx))


#########################################
# Definition of the training objectives #
#########################################

vgg(img_style)
style_targets = [GramMatrix()(A).detach() for A in style_activation]

vgg(img_content)
content_targets = [A.detach() for A in content_activation]


###################
# Actual training #
###################

# Init image
opt_img = torch.autograd.Variable(torch.zeros(img_style.size()).normal_(), requires_grad=True)

if torch.cuda.is_available():
    opt_img = opt_img.cuda()

# Hyper parameters
max_iter = 4000
show_iter = 500

# Optimizers
optimizer = Lookahead(RAdam([opt_img], lr=1e-1))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95)

# Losses
gram_loss = GramMSELoss()
mse_loss = nn.SmoothL1Loss()

for n_iter in range(max_iter):
    optimizer.zero_grad()
    vgg(opt_img)

    style_loss = sum(style_weights[a] * gram_loss(A, style_targets[a]) for a, A in enumerate(style_activation))
    content_loss = sum(content_weights[a] * mse_loss(A, content_targets[a]) for a, A in enumerate(content_activation))

    loss = content_loss + style_loss
    loss.backward()

    optimizer.step()

    
    if n_iter % show_iter == show_iter-1 and parser.verbose:
        print('Iteration: %d, loss: %f' % (n_iter+1, loss.item()))

        plt.imshow(renormalize(opt_img))
        plt.show()

plt.imsave("gatys_result.png", renormalize(opt_img))
"""
Algorithme Reinhard Color Transfert

This script takes two parameters as an input, the source (`source.png`), from which the style will be copied, and the target (`target.png`) the image to which the style will be applied. 
The resulting image will be saved as reinhard_result.png.
The Reinhard algorithm is basically just histogram renormalization in the Lab space.

Usage:
python reinhard.py source.png target.png
"""

from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import argparse
import os


###################
# Argument parser #
###################

parser = argparse.ArgumentParser(description='Usage: python reinhard.py source.png target.png')
parser.add_argument("source", help="The source image")
parser.add_argument("target", help="The target image")

args = parser.parse_args()


###################
# Load the images #
###################

if not (parser.source and os.path.isfile(parser.source) and parser.target and os.path.isfile(parser.target)):
    raise Exception("You must spectify a valid source and target image.")


##########################################
# Set of functions to make Reinhard work #
##########################################

def getImage(source):
    return cv2.cvtColor(np.array(cv2.resize(cv2.imread(source), (500, 500))), cv2.COLOR_BGR2LAB).astype("float32")

def getMean(img):
    l,a,b = cv2.split(img)
    
    return l.mean(), a.mean(), b.mean()

def getStd(img):
    l,a,b = cv2.split(img)
    
    return l.std(), a.std(), b.std()

def center(img):
    l,a,b = cv2.split(img)
    mean = getMean(img)
    
    l -= mean[0]
    a -= mean[1]
    b -= mean[2]
    
    return cv2.merge([l,a,b])

def setStd(img, new_std):
    l,a,b = cv2.split(img)
    std = getStd(img)
    
    l = new_std[0] * l / std[0]
    a = new_std[1] * a / std[1]
    b = new_std[2] * b / std[2]
    
    return cv2.merge([l,a,b])

def setMean(img, new_mean):
    l,a,b = cv2.split(img)
    
    l += new_mean[0]
    a += new_mean[1]
    b += new_mean[2]
    
    return cv2.merge([l,a,b])

def scaleRange(img):
    l,a,b = cv2.split(img)
    
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    
    return cv2.merge([l,a,b])

def ReinhardColorTransfert(source_url, target_url):
    source, target = getImage(source_url), getImage(target_url)
    
    mean_source = getMean(source)
    std_source = getStd(source)
    
    target = center(target)
    target = setStd(target, std_source)
    target = setMean(target, mean_source)
    
    target = scaleRange(target)
    
    return cv2.cvtColor(target.astype("uint8"), cv2.COLOR_LAB2RGB)


######################
# Output computation #
######################

plt.imsave("reinhard_result.png", ReinhardColorTransfert(parser.source, parser.target))
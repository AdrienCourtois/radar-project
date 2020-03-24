"""
Sliced Optimal Transport

This script takes two parameters as an input, the source (`source.png`), from which the style will be copied, and the target (`target.png`) the image to which the style will be applied. 
The resulting image will be saved as reinhard_result.png.
This algorithm is a histogram transfer in 3D.

Usage:
python transport.py source.png target.png
"""

from PIL import Image
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def equalizeHistogramm(A, B) :
    # target : A, an image NxNx3
    # source : B, an image NxNx3
    # apply the histogramm of B onto A
    
    ind = np.argsort(A)
    C = np.zeros(A.shape, A.dtype)
    C[ind] = np.sort(B)
    C = C.reshape(C.shape)
    return C
 
def flattenMatrix(A) :
    # flatten an image in a single dimension
    # result of dimensions N²x3
    
    return np.reshape(A, (A.shape[0]*A.shape[1], A.shape[2]))
 
def generateModPlageVar(F, G):
    #F and G are images, we compute a modified image of F
   
    #compute a new orthogonal basis
    tetha = st.ortho_group.rvs(3)
   
    #change the basis
    F = np.dot(F, tetha)
    G = np.dot(G, tetha)
   
    #flatten matrices
    flattenF = flattenMatrix(F)
    flattenG = flattenMatrix(G)
   
    # divide the image F into 3 color canals
    F1 = np.zeros((flattenF.shape[0]))
    F2 = np.zeros((flattenF.shape[0]))
    F3 = np.zeros((flattenF.shape[0]))
    
    F1[:] = flattenF[:,0]
    F2[:] = flattenF[:,1]
    F3[:] = flattenF[:,2]
           
    # # divide the image G into 3 color canals
    G1 = np.zeros((flattenG.shape[0]))
    G2 = np.zeros((flattenG.shape[0]))
    G3 = np.zeros((flattenG.shape[0]))
    
    G1[:] = flattenG[:,0]
    G2[:] = flattenG[:,1]
    G3[:] = flattenG[:,2]
   
    #equalisation of the histogramms of all colors canals in the new basis
    (F1, F2, F3) = (equalizeHistogramm(F1, G1), equalizeHistogramm(F2, G2), equalizeHistogramm(F3, G3))
   
    #return to the initial basis
    tethainv = np.transpose(tetha)
   
    #regenerateF
    #fullfill the new image
    
    c = 0
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
                F[i][j][0] = F1[c]
                F[i][j][1] = F2[c]
                F[i][j][2] = F3[c]
                c = c+1
    F = np.dot(F, tethainv)
    return F

def SlicedOptimalTransfer(source, target, opened = False, returnSpeed = False):
    # load the images and reshape them
    if not opened:
        source =  np.array(Image.open(source).resize((500,500))) # Read image in RGB
        target = np.array(Image.open(target).resize((500,500))) # Read image in RGB
    else:
        source = np.copy(source)
        target = np.copy(target)
    
    hist_source, bins = np.histogram(source, bins = 255)

    #step
    ToL = 0.15
    steps = 300
    
    speed = []

    #gradient descent
    for i in range(steps):
        if i % 10 == 0 and i != 0:
            hist1, bins = np.histogram(target, bins = 255)
            
            speed.append(np.linalg.norm(hist1, 2))
            
        modTarget = generateModPlageVar(target, source)
        target = (1 - ToL) * target + ToL * modTarget

    #re-normalize
    result = target
    result[result < 0] = 0
    result[result > 255] = 255
    result = result.astype(np.uint8)
    
    if not returnSpeed:
        return result
    else:
        return result, np.array(speed)
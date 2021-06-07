import numpy as np
import os
# import cv2 as cv
import skimage
import skimage.feature
import skimage.viewer
import sys

sigma = 2.0
low_threshold = 0.1
high_threshold = 0.3

#from matplotlib import pyplot as plt
# path = "SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks/"
# outputDirectory = "SBU-shadow/SBUTrain4KRecoveredSmall/EdgeMasks/"
path = "D:/ISTD_Dataset/train/train_B/"
outputDirectory = "D:/ISTD_Dataset/train/edgeMasks/"

if not os.path.exists(outputDirectory):
    os.mkdir(outputDirectory)

list = os.listdir(path)

for i in list:
    #you can ignore this, i just tried another version
    image = skimage.io.imread(fname=path + i, as_gray=True)
    edges = skimage.feature.canny(
        image=image,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    skimage.io.imsave(outputDirectory + "/" + i, edges)
    # img = cv.imread(path + i, 0)
    # edgeimg = cv.Canny(img, img.shape[1], img.shape[0])
    # cv.imwrite(outputDirectory + "/" + i, edgeimg)




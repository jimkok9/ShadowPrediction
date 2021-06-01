import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
path = "UCF/GroundTruth/"
outputDirectory = "UCF/EdgeMasks"
if not os.path.exists(outputDirectory):
    os.mkdir(outputDirectory)

list = os.listdir(path)

for i in list:
    img = cv.imread(path + i, 0)
    edgeimg = cv.Canny(img, img.shape[1], img.shape[0])
    cv.imwrite(outputDirectory + "/" + i, edgeimg)




import numpy as np
import os
from csv import writer

from test_MT import test_calculate_metric

modelFolders = ["D:/computerVisionModels/ISTD_Jim/", "D:/computerVisionModels/SBU_Jim/","D:/computerVisionModels/SBU2_Dekel/"]
dataFolders = ["D:/ISTD_Dataset/test", "D:/SBU-shadow/SBU-Test", "D:/SBU-shadow/SBU-Test"]
outputFolder = ['D:/results/JimISTD/','D:/results/JimSBU/','D:/results/DekelSBU/']
modelNames = ["ISTD","SBU","SBU"]
iter = np.arange(100,1100,100)
print(iter)
with open(outputFolder[1] + 'results.csv', 'a') as f_object:
    writer_object = writer(f_object)
    for i in iter:
        snapshot_path = modelFolders[1] + modelNames[1] + "_iter_" + str(i) + '.pth'
        data_path = dataFolders[1]

        scores = test_calculate_metric(snapshot_path, data_path, "D:/testFolder")
        writer_object.writerow(scores)
    f_object.close()

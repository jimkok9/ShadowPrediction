import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from PIL import Image
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, groundTruthShadowsDirectory, groundTruthShadowEdgesDirectory, imagesDirectory, transform=None, target_transform=None):
        images = os.listdir(imagesDirectory)
        self.imagesDirectory = imagesDirectory
        self.groundTruthShadowsDirectory = groundTruthShadowsDirectory
        self.groundTruthShadowEdgesDirectory = groundTruthShadowEdgesDirectory
        self.imageIndex = {}
        self.data = []
        cnt = 0
        for img in images:
            self.imageIndex[cnt] = img
            cnt = cnt + 1
            self.data.append(img)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.imageIndex[idx]
        input = Image.open(self.imagesDirectory + "/" + img)
        input = ToTensor()(input).unsqueeze(0)
        label = False
        mask = None
        shadowEdgeMask = None
        if(os.path.exists(self.groundTruthShadowsDirectory + "/" + img[:-4] + ".png")):
            mask = Image.open(self.groundTruthShadowsDirectory + "/" + img[:-4] + ".png")
            mask = ToTensor()(mask).unsqueeze(0)
            shadowEdgeMask = Image.open(self.groundTruthShadowEdgesDirectory + "/" + img[:-4] + ".png")
            shadowEdgeMask = ToTensor()(shadowEdgeMask).unsqueeze(0)
            label = True
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        sample = {"image": input, "groundTruth": mask, "groundTruthEdge": shadowEdgeMask, "label": label}
        return sample
import sys, os
from PIL import Image
from resnext101_EF import ResNeXt101
from torchvision.transforms import ToTensor
from torch import nn
from MTMT import build_model

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.BatchNorm2d(list_k[1][i]), nn.ReLU(inplace=True)))

        self.convert0 = nn.ModuleList(up0)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

if __name__ == "__main__":
    config_resnext101 = {'convert': [[64, 256, 512, 1024, 2048], [32, 64, 64, 64, 64]],
                         'merge1': [[32, 0, 32, 3, 1], [64, 0, 64, 3, 1], [64, 0, 64, 5, 2], [64, 0, 64, 5, 2], [64, 0, 64, 7, 3]], 'merge2': [[32], [64, 64, 64, 64]]}
    img = Image.open("UCF/InputImages/04822305_2305_3329_2561_3585.jpg")
    img = ToTensor()(img).unsqueeze(0)

    ResNext = ResNeXt101()
    ResNextFeatures = ResNext.forward(img)

    convertResNext = ConvertLayer(config_resnext101['convert'])
    EF = convertResNext.forward(ResNextFeatures)
    print(EF[0].shape)
    print(EF[1].shape)
    print(EF[2].shape)
    print(EF[3].shape)
    print(EF[4].shape)

    








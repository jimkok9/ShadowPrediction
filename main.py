import torch.nn.functional as F
from PIL import Image
from resnext101_EF import ResNeXt101
from torchvision.transforms import ToTensor
from torch import nn
import torch
import matplotlib.pyplot as plt


class ConvertResNext(nn.Module):
    def __init__(self):
        super(ConvertResNext, self).__init__()
        convertLayers = []
        input_channel_size = [64, 256, 512, 1024, 2048]
        output_channel_size = [32, 64, 64, 64, 64]
        for i in range(len(input_channel_size)):
            convertLayers.append(nn.Sequential(nn.Conv2d(input_channel_size[i], output_channel_size[i], 1, 1, bias=False),
                                     nn.BatchNorm2d(output_channel_size[i]), nn.ReLU(inplace=True)))
        self.convert = nn.ModuleList(convertLayers)

    def forward(self, outputResNext):
        EF = []
        for i in range(len(outputResNext)):
            EF.append(self.convert[i](outputResNext[i]))
        return EF


class EFtoDF(nn.Module):
    def __init__(self):
        super(EFtoDF, self).__init__()
        conv, shortConnections = [], []
        list_k = [[32, 0, 32, 3, 1], [64, 0, 64, 3, 1], [64, 0, 64, 5, 2], [64, 0, 64, 5, 2], [64, 0, 64, 7, 3]]
        for i, ik in enumerate(list_k):
            if 0 < i < 4:
                shortConnections.append(nn.Sequential(nn.Conv2d(64*(i+1), 64, 1, 1, 1),
                                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
            conv.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                      nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                      nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True)))

        oneXoneConv = nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True))

        self.conv = conv
        self.shortCon = shortConnections
        self.oneXoneConv = oneXoneConv

    def forward(self, EF):
        DF5 = self.conv[4](EF[4])

        EF5x2 = F.interpolate(EF[4], EF[3].size()[2:], mode='bilinear', align_corners=True)
        DF4 = self.conv[3](self.shortCon[0](torch.cat([EF5x2, EF[3]], dim=1)))

        EF4x2 = F.interpolate(EF[3], EF[2].size()[2:], mode='bilinear', align_corners=True)
        EF5x4 = F.interpolate(EF[4], EF[2].size()[2:], mode='bilinear', align_corners=True)
        DF3 = self.conv[2](self.shortCon[1](torch.cat([EF4x2, EF5x4, EF[2]], dim=1)))

        EF3x2 = F.interpolate(EF[2], EF[1].size()[2:], mode='bilinear', align_corners=True)
        EF4x4 = F.interpolate(EF[3], EF[1].size()[2:], mode='bilinear', align_corners=True)
        EF5x8 = F.interpolate(EF[4], EF[1].size()[2:], mode='bilinear', align_corners=True)
        DF2 = self.conv[1](self.shortCon[2](torch.cat([EF3x2, EF4x4, EF5x8, EF[1]], dim=1)))

        DF1 = EF[0] + F.interpolate(self.oneXoneConv(DF5), EF[0].size()[2:], mode='bilinear', align_corners=True)

        return [DF1, DF2, DF3, DF4, DF5]

class EF5ToSC(nn.Module):
    def __init__(self):
        super(EF5ToSC, self).__init__()
        self.linear = nn.Linear(64, 1)

    def forward(self, EF5):
        vector = F.adaptive_avg_pool2d(EF5, output_size=1)
        vector = vector.view(vector.size(0), -1)
        sc_actual = self.linear(vector)

        return sc_actual


class DFtoPred(nn.Module):
    def __init__(self):
        super(DFtoPred, self).__init__()
        predictions = []
        for i in range(0, 5):
            inputChannels = 32 if i==0 else 64
            predictions.append(nn.Sequential(
                nn.Conv2d(inputChannels, 64, 3, 1, 1), nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 16, 3, 1, 1), nn.Conv2d(16, 1, 1), nn.Sigmoid()
            ))
        self.predictions = predictions

    def forward(self, DF, sizeInput):
        pred0 = F.interpolate(self.predictions[0](DF[0]), sizeInput, mode='bilinear', align_corners=True)
        pred1 = F.interpolate(self.predictions[1](DF[1]), sizeInput, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(self.predictions[2](DF[2]), sizeInput, mode='bilinear', align_corners=True)
        pred3 = F.interpolate(self.predictions[3](DF[3]), sizeInput, mode='bilinear', align_corners=True)
        pred4 = F.interpolate(self.predictions[4](DF[4]), sizeInput, mode='bilinear', align_corners=True)

        return [pred0, pred1, pred2, pred3, pred4]

class DFtoRF(nn.Module):
    def __init__(self):
        super(DFtoRF, self).__init__()
        oneXoneConvs = []
        for i in range(0, 5):
            oneXoneConvs.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.oneXoneConvs = oneXoneConvs

    def forward(self, DF):
        RF2 = DF[0] + F.interpolate(self.oneXoneConvs[0](DF[1]), DF[0].size()[2:], mode='bilinear', align_corners=True)
        RF3 = DF[0] + F.interpolate(self.oneXoneConvs[1](DF[2]), DF[0].size()[2:], mode='bilinear', align_corners=True)
        RF4 = DF[0] + F.interpolate(self.oneXoneConvs[2](DF[3]), DF[0].size()[2:], mode='bilinear', align_corners=True)
        RF5 = DF[0] + F.interpolate(self.oneXoneConvs[3](DF[4]), DF[0].size()[2:], mode='bilinear', align_corners=True)

        return [RF2, RF3, RF4, RF5]

class RFtoPred(nn.Module):
    def __init__(self):
        super(RFtoPred, self).__init__()
        predictions = []
        for i in range(0, 5):
            predictions.append(nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1), nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 16, 3, 1, 1), nn.Conv2d(16, 1, 1), nn.Sigmoid()
            ))
        self.predictions = predictions

    def forward(self, RF, sizeInput):
        pred0 = F.interpolate(self.predictions[0](RF[0]), sizeInput, mode='bilinear', align_corners=True)
        pred1 = F.interpolate(self.predictions[1](RF[1]), sizeInput, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(self.predictions[2](RF[2]), sizeInput, mode='bilinear', align_corners=True)
        pred3 = F.interpolate(self.predictions[3](RF[3]), sizeInput, mode='bilinear', align_corners=True)
        S_f = F.interpolate(self.predictions[4](RF[0] + RF[1] + RF[2] + RF[3]), sizeInput, mode='bilinear', align_corners=True)

        return [pred0, pred1, pred2, pred3, S_f]

class MTMT(nn.Module):
    def __init__(self):
        super(MTMT, self).__init__()
        self.resNext = ResNeXt101()
        self.convert = ConvertResNext()
        self.EFtoDF = EFtoDF()
        self.DFtoPred = DFtoPred()
        self.DFtoRF = DFtoRF()
        self.RFtoPred = RFtoPred()

    def forward(self, img):
        img = ToTensor()(img).unsqueeze(0)
        size = img.size()[2:]
        EF = self.convert(self.resNext(img))
        DF = self.EFtoDF(EF)
        DFPred = self.DFtoPred(DF, size)
        RF = self.DFtoRF(DF)
        RFPred = self.RFtoPred(RF, size)

        return DFPred, RFPred




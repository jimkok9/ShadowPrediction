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
        ConvertLayers = []
        input_channel_size = [64, 256, 512, 1024, 2048]
        output_channel_size = [32, 64, 64, 64, 64]
        for i in range(len(input_channel_size)):
            ConvertLayers.append(nn.Sequential(nn.Conv2d(input_channel_size[i], output_channel_size[i], 1, 1, bias=False),
                                     nn.BatchNorm2d(output_channel_size[i]), nn.ReLU(inplace=True)))
        self.convert = nn.ModuleList(ConvertLayers)

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
        # device = torch.device('cuda')
        # for c in conv:
        #     c.to(device)
        # for s in shortConnections:
        #     s.to(device)
        self.conv = nn.ModuleList(conv)
        self.shortCon = nn.ModuleList(shortConnections)
        self.oneXoneConv = oneXoneConv

        #subitizing section
        self.number_per_fc = nn.Linear(list_k[1][2], 1)  # 64->1
        torch.nn.init.constant_(self.number_per_fc.weight, 0)

        # self.shadow_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.shadow_score = nn.Sequential(
            nn.Conv2d(list_k[1][2], list_k[1][2] // 4, 3, 1, 1), nn.BatchNorm2d(list_k[1][2] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[1][2] // 4, 1, 1)
        )
        # self.edge_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.edge_score = nn.Sequential(
            nn.Conv2d(list_k[0][2], list_k[0][2] // 4, 3, 1, 1), nn.BatchNorm2d(list_k[0][2] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[0][2] // 4, 1, 1)
        )

    def forward(self, EF, size):
        DFfeatures = []
        DF5 = self.conv[4](EF[4])
        vector = F.adaptive_avg_pool2d(DF5, output_size=1)
        vector = vector.view(vector.size(0), -1)
        up_subitizing = self.number_per_fc(vector)
        shadowScores, edgeScores = [], []

        DFfeatures.append(DF5)
        shadowScores.append(F.interpolate(self.shadow_score(DF5), size, mode='bilinear', align_corners=True))
        EF5x2 = F.interpolate(EF[4], EF[3].size()[2:], mode='bilinear', align_corners=True)
        DF4 = self.conv[3](self.shortCon[0](torch.cat([EF5x2, EF[3]], dim=1)))
        shadowScores.append(F.interpolate(self.shadow_score(DF4), size, mode='bilinear', align_corners=True))
        DFfeatures.append(DF4)

        EF4x2 = F.interpolate(EF[3], EF[2].size()[2:], mode='bilinear', align_corners=True)
        EF5x4 = F.interpolate(EF[4], EF[2].size()[2:], mode='bilinear', align_corners=True)
        DF3 = self.conv[2](self.shortCon[1](torch.cat([EF4x2, EF5x4, EF[2]], dim=1)))
        shadowScores.append(F.interpolate(self.shadow_score(DF3), size, mode='bilinear', align_corners=True))
        DFfeatures.append(DF3)

        EF3x2 = F.interpolate(EF[2], EF[1].size()[2:], mode='bilinear', align_corners=True)
        EF4x4 = F.interpolate(EF[3], EF[1].size()[2:], mode='bilinear', align_corners=True)
        EF5x8 = F.interpolate(EF[4], EF[1].size()[2:], mode='bilinear', align_corners=True)
        DF2 = self.conv[1](self.shortCon[2](torch.cat([EF3x2, EF4x4, EF5x8, EF[1]], dim=1)))
        DFfeatures.append(DF2)

        shadowScores.append(F.interpolate(self.shadow_score(DF2), size, mode='bilinear', align_corners=True))
        tmp = EF[0] + F.interpolate(self.oneXoneConv(DF5), EF[0].size()[2:], mode='bilinear', align_corners=True)
        DF1 = self.conv[0](tmp)
        DFfeatures.append(DF1)
        edgeScores.append(F.interpolate(self.edge_score(DF1), size, mode='bilinear', align_corners=True))
        return DFfeatures, up_subitizing, shadowScores, edgeScores

class DFtoPred(nn.Module):
    def __init__(self):
        super(DFtoPred, self).__init__()
        predictions = []
        for i in range(0, 5):
            inputChannels = 32 if i==0 else 64
            predictions.append(nn.Sequential(
                nn.Conv2d(inputChannels, 64, 3, 1, 1), nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 16, 3, 1, 1), nn.Conv2d(16, 1, 1), nn.Sigmoid()
            ))

        self.predictions = nn.ModuleList(predictions)

    def forward(self, DF, sizeInput):
        pred0 = F.interpolate(self.predictions[0](DF[0]), sizeInput, mode='bilinear', align_corners=True)
        pred1 = F.interpolate(self.predictions[1](DF[1]), sizeInput, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(self.predictions[2](DF[2]), sizeInput, mode='bilinear', align_corners=True)
        pred3 = F.interpolate(self.predictions[3](DF[3]), sizeInput, mode='bilinear', align_corners=True)
        pred4 = F.interpolate(self.predictions[4](DF[4]), sizeInput, mode='bilinear', align_corners=True)


        # returning pred 0 = DF1 prediction and pred 4 = DF5 prediction
        return [pred0, pred1, pred2, pred3, pred4]

class DFtoRF(nn.Module):
    def __init__(self):
        super(DFtoRF, self).__init__()
        list_k = [[32], [64, 64, 64, 64]];
        oneXoneConvs = []
        up = []
        tmp = []
        tmp_up = []
        feature_k = [[3, 1], [5, 2], [5, 2], [7, 3]]
        for idx, j in enumerate(list_k[1]):
            tmp.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))
            tmp_up.append(
                nn.Sequential(nn.Conv2d(32, 32, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(32),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(32, 32, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(32),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(32, 32, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(32),
                              nn.ReLU(inplace=True)))
        # oneXoneConvs.append(nn.ModuleList(tmp))
        # up.append(nn.ModuleList(tmp))
        self.oneXoneConvs = nn.ModuleList(tmp)
        self.up = nn.ModuleList(tmp_up)

        self.sub_score = nn.Sequential(
            nn.Conv2d(list_k[0][0], list_k[0][0] // 4, 3, 1, 1), nn.BatchNorm2d(list_k[0][0] // 4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[0][0] // 4, 1, 1)
        )

        self.relu = nn.ReLU()

    def forward(self, DF, sizeInput):
        up_score = []
        tmp_feature = []

        RF2 = DF[4] + F.interpolate(self.oneXoneConvs[0](DF[3]), DF[4].size()[2:], mode='bilinear', align_corners=True)
        tmp_f = self.up[0](RF2)
        up_score.append(F.interpolate(self.sub_score(tmp_f), sizeInput, mode='bilinear', align_corners=True))
        tmp_feature.append(tmp_f)

        RF3 = DF[4] + F.interpolate(self.oneXoneConvs[1](DF[2]), DF[4].size()[2:], mode='bilinear', align_corners=True)
        tmp_f = self.up[1](RF3)
        up_score.append(F.interpolate(self.sub_score(tmp_f), sizeInput, mode='bilinear', align_corners=True))
        tmp_feature.append(tmp_f)

        RF4 = DF[4] + F.interpolate(self.oneXoneConvs[2](DF[1]), DF[4].size()[2:], mode='bilinear', align_corners=True)
        tmp_f = self.up[2](RF4)
        up_score.append(F.interpolate(self.sub_score(tmp_f), sizeInput, mode='bilinear', align_corners=True))
        tmp_feature.append(tmp_f)

        RF5 = DF[4] + F.interpolate(self.oneXoneConvs[3](DF[0]), DF[4].size()[2:], mode='bilinear', align_corners=True)
        tmp_f = self.up[3](RF5)
        up_score.append(F.interpolate(self.sub_score(tmp_f), sizeInput, mode='bilinear', align_corners=True))
        tmp_feature.append(tmp_f)

        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea + 1]), tmp_feature[0].size()[2:],
                                                                 mode='bilinear', align_corners=True)))
        up_score.append(F.interpolate(self.sub_score(tmp_fea), sizeInput, mode='bilinear', align_corners=True))
        # up_score.append(F.interpolate(self.final_score(tmp_fea), x_size, mode='bilinear', align_corners=True))

        return up_score

class RFtoPred(nn.Module):
    def __init__(self):
        super(RFtoPred, self).__init__()
        predictions = []

        for i in range(0, 5):
            predictions.append(nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1), nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 16, 3, 1, 1), nn.Conv2d(16, 1, 1), nn.Sigmoid()
            ))

        self.predictions = nn.ModuleList(predictions)

    def forward(self, RF, sizeInput):
        pred0 = F.interpolate(self.predictions[0](RF[0]), sizeInput, mode='bilinear', align_corners=True)
        pred1 = F.interpolate(self.predictions[1](RF[1]), sizeInput, mode='bilinear', align_corners=True)
        pred2 = F.interpolate(self.predictions[2](RF[2]), sizeInput, mode='bilinear', align_corners=True)
        pred3 = F.interpolate(self.predictions[3](RF[3]), sizeInput, mode='bilinear', align_corners=True)
        S_f = F.interpolate(self.predictions[4](RF[0] + RF[1] + RF[2] + RF[3]), sizeInput, mode='bilinear', align_corners=True)
        #pred 0 = RF2 S_f = S_f
        return [pred0, pred1, pred2, pred3, S_f]

class MTMT(nn.ModuleList):
    def __init__(self):
        super(MTMT, self).__init__()
        self.resNext = ResNeXt101()
        self.convert = ConvertResNext()
        self.EFtoDF = EFtoDF()
        self.DFtoPred = DFtoPred()
        self.DFtoRF = DFtoRF()
        self.RFtoPred = RFtoPred()

    def forward(self, img):
        # img = ToTensor()(img).unsqueeze(0)
        size = img.size()[2:]
        EF = self.convert(self.resNext(img))
        DF, up_sabotizing, shadowScores, edgeScores = self.EFtoDF(EF, size)
        #DFPred = self.DFtoPred(DF, size)
        RFPred = self.DFtoRF(DF,size)
        #RFPred = self.RFtoPred(RF, size)

        return edgeScores, shadowScores, up_sabotizing, RFPred



  #  plt.imshow(pred2[4].reshape(257, 257, 1).detach().numpy())



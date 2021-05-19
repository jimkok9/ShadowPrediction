import torch.nn.functional as F
from PIL import Image
from resnext101_EF import ResNeXt101
from torchvision.transforms import ToTensor
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

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
        for i in range(0, 5):
            if i > 0 and i < 4:
             shortConnections.append(nn.Sequential(nn.Conv2d(64*(i+1), 64, 1, 1, 1),
                                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
            numChannels=32 if i == 0 else 64
            conv.append(nn.Sequential(nn.Conv2d(numChannels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))

        oneXoneConv = nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True))
        predict = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 16, 3, 1, 1), nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )
        self.conv = nn.ModuleList(conv)
        self.shortCon = nn.ModuleList(shortConnections)
        self.predict = nn.ModuleList(predict)
        self.oneXoneConv = nn.ModuleList(oneXoneConv)


    def forward(self, EF, sizeInput):
        DF5 = self.conv[4](EF[4])

        EF5x2 = F.interpolate(EF[4], EF[3].size()[2:], mode='bilinear', align_corners=True)
        print(self.conv)
        DF4 = self.conv[3](self.shortCon[3](torch.cat([EF5x2, EF[3]], dim=1)))

        EF4x2 = F.interpolate(EF[3], EF[2].size()[2:], mode='bilinear', align_corners=True)
        EF5x4 = F.interpolate(EF[4], EF[2].size()[2:], mode='bilinear', align_corners=True)
        DF3 = self.conv[2](self.shortCon[2](torch.cat([EF4x2, EF5x4, EF[2]], dim=1)))

        EF3x2 = F.interpolate(EF[2], EF[1].size()[2:], mode='bilinear', align_corners=True)
        EF4x4 = F.interpolate(EF[3], EF[1].size()[2:], mode='bilinear', align_corners=True)
        EF5x8 = F.interpolate(EF[4], EF[1].size()[2:], mode='bilinear', align_corners=True)
        DF2 = self.conv[1](self.shortCon[1](torch.cat([EF3x2, EF4x4, EF5x8, EF[1]], dim=1)))

        DF1 = EF[0] + F.interpolate(self.oneXoneConv(DF5), EF[0].size()[2:], mode='bilinear', align_corners=True)

        return [DF1, DF2, DF3, DF4, DF5]

class MergeLayer2(nn.Module):
    def __init__(self):
        super(MergeLayer2, self).__init__()
        list_k = [[32], [64, 64, 64, 64]]
        self.list_k = list_k
        trans, up, score = [], [], []
        for i in list_k[0]:
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3, 1], [5, 2], [5, 2], [7, 3]]
            for idx, j in enumerate(list_k[1]):
                tmp.append(nn.Sequential(nn.Conv2d(j, i, 1, 1, bias=False), nn.BatchNorm2d(i), nn.ReLU(inplace=True)))
                tmp_up.append(
                    nn.Sequential(nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
                                  nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(i), nn.ReLU(inplace=True),
                                  nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.BatchNorm2d(i), nn.ReLU(inplace=True)))
            trans.append(nn.ModuleList(tmp))
            up.append(nn.ModuleList(tmp_up))
        # self.sub_score = nn.Conv2d(list_k[0][0], 1, 3, 1, 1)
        self.sub_score = nn.Sequential(
            nn.Conv2d(list_k[0][0], list_k[0][0]//4, 3, 1, 1), nn.BatchNorm2d(list_k[0][0]//4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[0][0]//4, 1, 1)
        )

        self.trans, self.up = nn.ModuleList(trans), nn.ModuleList(up)
        # self.final_score = nn.Sequential(nn.Conv2d(list_k[0][0], list_k[0][0], 5, 1, 2), nn.ReLU(inplace=True),
        #                                  nn.Conv2d(list_k[0][0], 1, 3, 1, 1))
        # self.final_score = nn.Sequential(
        #     nn.Conv2d(list_k[0][0], list_k[0][0], 5, 1, 2), nn.BatchNorm2d(list_k[0][0]), nn.ReLU(inplace=True),
        #     nn.Conv2d(list_k[0][0], list_k[0][0]//4, 3, 1, 1), nn.BatchNorm2d(list_k[0][0]//4), nn.ReLU(inplace=True),
        #     nn.Dropout(0.1), nn.Conv2d(list_k[0][0]//4, 1, 1)
        # )
        self.relu = nn.ReLU()

    def forward(self, list_x, list_y, x_size):
        up_score, tmp_feature = [], []
        list_y = list_y[::-1]

        for i, i_x in enumerate(list_x):
            for j, j_x in enumerate(list_y):
                tmp = F.interpolate(self.trans[i][j](j_x), i_x.size()[2:], mode='bilinear', align_corners=True) + i_x
                tmp_f = self.up[i][j](tmp)
                up_score.append(F.interpolate(self.sub_score(tmp_f), x_size, mode='bilinear', align_corners=True))
                tmp_feature.append(tmp_f)


        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea + 1]), tmp_feature[0].size()[2:],
                                                                 mode='bilinear', align_corners=True)))
        up_score.append(F.interpolate(self.sub_score(tmp_fea), x_size, mode='bilinear', align_corners=True))
        # up_score.append(F.interpolate(self.final_score(tmp_fea), x_size, mode='bilinear', align_corners=True))

        return up_score

if __name__ == "__main__":
    img = Image.open("UCF/InputImages/04822305_2305_3329_2561_3585.jpg")
    img = ToTensor()(img).unsqueeze(0)

    ResNext = ResNeXt101()
    ResNextFeatures = ResNext.forward(img)

    convertResNext = ConvertResNext()
    EF = convertResNext.forward(ResNextFeatures)

    merge1_layers = EFtoDF()
    DF = merge1_layers(EF, [257, 257])
    print(DF.shape)
    # Pred , DF1         , DF2345, Pred

    # plt.imshow(up_edge[0].reshape(257, 257, 1).detach().numpy())
    # plt.show()
    #
    # merge2_layers = MergeLayer2()
    # up_score = merge2_layers(edge_feature, sal_feature, [257, 257])


#     predicted.append(F.interpolate(self.predict(DF2), sizeInput, mode='bilinear', align_corners=True))
#     predicted.append(F.interpolate(self.predict(DF3), sizeInput, mode='bilinear', align_corners=True))
#     predicted.append(F.interpolate(self.predict(DF4), sizeInput, mode='bilinear', align_corners=True))
#     predicted.append(F.interpolate(self.predict(DF5), sizeInput, mode='bilinear', align_corners=True))
#
#     up_edge.append(F.interpolate(self.edge_score(tmp), sizeInput, mode='bilinear', align_corners=True))
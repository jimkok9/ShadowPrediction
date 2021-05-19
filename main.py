import torch.nn.functional as F
from PIL import Image
from resnext101_EF import ResNeXt101
from torchvision.transforms import ToTensor
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

class ConvertLayer(nn.Module): # list_k: [[64, 256, 512, 1024, 2048], [32, 64, 64, 64, 64]]
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

#DSS merge
class MergeLayer1(nn.Module): # list_k: [[32, 0, 32, 3, 1], [64, 0, 64, 3, 1], [64, 0, 64, 5, 2], [64, 0, 64, 5, 2], [64, 0, 64, 7, 3]]

    def __init__(self, list_k):
        super(MergeLayer1, self).__init__()
        self.list_k = list_k
        trans, up, DSS = [], [], []

        for i, ik in enumerate(list_k):
            up.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.BatchNorm2d(ik[2]), nn.ReLU(inplace=True)))
            if i > 0 and i < len(list_k)-1: # i represent number
                DSS.append(nn.Sequential(nn.Conv2d(ik[0]*(i+1), ik[0], 1, 1, 1),
                                         nn.BatchNorm2d(ik[0]), nn.ReLU(inplace=True)))

        trans.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1, bias=False), nn.ReLU(inplace=True)))
        # self.shadow_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.shadow_score = nn.Sequential(
            nn.Conv2d(list_k[1][2], list_k[1][2]//4, 3, 1, 1), nn.BatchNorm2d(list_k[1][2]//4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[1][2]//4, 1, 1)
        )
        # self.edge_score = nn.Conv2d(list_k[0][2], 1, 3, 1, 1)
        self.edge_score = nn.Sequential(
            nn.Conv2d(list_k[0][2], list_k[0][2]//4, 3, 1, 1), nn.BatchNorm2d(list_k[0][2]//4), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(list_k[0][2]//4, 1, 1)
        )

        self.up = nn.ModuleList(up)
        self.relu = nn.ReLU()
        self.trans = nn.ModuleList(trans)
        self.DSS = nn.ModuleList(DSS)

        # subitizing section
        self.number_per_fc = nn.Linear(list_k[1][2], 1) #64->1
        torch.nn.init.constant_(self.number_per_fc.weight, 0)

    def forward(self, list_x, x_size):
        up_edge, up_sal, edge_feature, sal_feature, U_tmp = [], [], [], [], []

        num_f = len(list_x)
        tmp = self.up[num_f - 1](list_x[num_f - 1])
        sal_feature.append(tmp)
        U_tmp.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))
        # layer5
        up_tmp_x2 = F.interpolate(U_tmp[0], list_x[3].size()[2:], mode='bilinear', align_corners=True)
        U_tmp.append(self.DSS[0](torch.cat([up_tmp_x2, list_x[3]], dim=1)))
        tmp = self.up[3](U_tmp[-1])
        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))
        # layer4
        up_tmp_x2 = F.interpolate(U_tmp[1], list_x[2].size()[2:], mode='bilinear', align_corners=True)
        up_tmp_x4 = F.interpolate(U_tmp[0], list_x[2].size()[2:], mode='bilinear', align_corners=True)
        U_tmp.append(self.DSS[1](torch.cat([up_tmp_x4, up_tmp_x2, list_x[2]], dim=1)))
        tmp = self.up[2](U_tmp[-1])
        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))
        # layer3
        up_tmp_x2 = F.interpolate(U_tmp[2], list_x[1].size()[2:], mode='bilinear', align_corners=True)
        up_tmp_x4 = F.interpolate(U_tmp[1], list_x[1].size()[2:], mode='bilinear', align_corners=True)
        up_tmp_x8 = F.interpolate(U_tmp[0], list_x[1].size()[2:], mode='bilinear', align_corners=True)
        U_tmp.append(self.DSS[2](torch.cat([up_tmp_x8, up_tmp_x4, up_tmp_x2, list_x[1]], dim=1)))
        tmp = self.up[1](U_tmp[-1])
        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.shadow_score(tmp), x_size, mode='bilinear', align_corners=True))

        vector = F.adaptive_avg_pool2d(sal_feature[0], output_size=1)
        vector = vector.view(vector.size(0), -1)
        up_subitizing = self.number_per_fc(vector)

        # edge layer fuse
        U_tmp = list_x[0] + F.interpolate((self.trans[-1](sal_feature[0])), list_x[0].size()[2:], mode='bilinear', align_corners=True)
        tmp = self.up[0](U_tmp)
        edge_feature.append(tmp)

        up_edge.append(F.interpolate(self.edge_score(tmp), x_size, mode='bilinear', align_corners=True))
        return up_edge, edge_feature, up_sal, sal_feature , up_subitizing
               # Pred , DF1         , DF2345, Pred        ,           ,

class MergeLayer2(nn.Module): # list_k [[32], [64, 64, 64, 64]]
    def __init__(self, list_k):
        super(MergeLayer2, self).__init__()
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
    config_resnext101 = {'convert': [[64, 256, 512, 1024, 2048], [32, 64, 64, 64, 64]]}
    config = {'merge1': [[32, 0, 32, 3, 1], [64, 0, 64, 3, 1], [64, 0, 64, 5, 2], [64, 0, 64, 5, 2], [64, 0, 64, 7, 3]], 'merge2': [[32], [64, 64, 64, 64]]}
    img = Image.open("UCF/InputImages/04822305_2305_3329_2561_3585.jpg")
    img = ToTensor()(img).unsqueeze(0)

    ResNext = ResNeXt101()
    ResNextFeatures = ResNext.forward(img)

    convertResNext = ConvertLayer(config_resnext101['convert'])
    EF = convertResNext.forward(ResNextFeatures)

    merge1_layers = MergeLayer1(config['merge1'])
    up_edge, edge_feature, up_sal, sal_feature, up_subitizing = merge1_layers(EF, [257, 257])
    # Pred , DF1         , DF2345, Pred

    # print(up_edge[0].shape)
    # print(edge_feature[0].shape)
    # print(up_sal[0].shape)
    # print(sal_feature[0].shape)
    # plt.imshow(up_edge[0].reshape(257, 257, 1).detach().numpy())
    # plt.show()

    merge2_layers = MergeLayer2(config['merge2'])
    up_score = merge2_layers(edge_feature, sal_feature, [257, 257])
    print(up_score[0].shape)
    print(up_score[0])
    plt.imshow(up_score[0].reshape(257, 257, 1).detach().numpy())
    plt.show()







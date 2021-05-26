from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from createDataset import CustomImageDataset
from main import fullModel, ConvertResNext, EFtoDF, DFtoPred, DFtoRF, RFtoPred
import torch.nn.functional as F
from PIL import Image
from resnext101_EF import ResNeXt101
from torchvision.transforms import ToTensor
from torch import nn
import torch
import matplotlib.pyplot as plt

from util.util import TwoStreamBatchSampler


def train_loop(dataloader, teacherModel, studentModel, binaryCrossEntropyLoss = nn.BCELoss(), MSELoss = nn.MSELoss(), optimizer = None):
    size = len(dataloader.dataset)
    for batch, sampled_batch in enumerate(dataloader):
        print(batch, sampled_batch)
        imageBatch = sampled_batch["image"]
        labelBatch = sampled_batch["label"]
        shadowBatch = sampled_batch["groundTruth"]
        edgeBatch = sampled_batch["groundTruthEdge"]

        teacherOutput = teacherModel(imageBatch)


        # Compute prediction and loss
        # pred = teacherModel(X)
        # predShadowEdge = pred[0][0]
        # predShadowMask1 = pred[0][1:4]
        # predShadowMask2 = pred[1]
        # MSELoss()
        # #lossC =
        # lossEdge =
        # loss = loss_fn(pred, y)
        #
        # # Backpropagation
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        #
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def getIndices(dataset):
    labelled = []
    unlabelled = []
    for sample in range(len(dataset)):
        if dataset[sample]["label"] == True:
            labelled.append(sample)
        else:
            unlabelled.append(sample)

    return (labelled, unlabelled)

ResNext = ResNeXt101()
convertResNext = ConvertResNext()
toDF = EFtoDF()
toPred = DFtoPred()
toRF = DFtoRF()
toPred2 = RFtoPred()
teacherModel = fullModel(ResNext, convertResNext, toDF, toPred, toRF, toPred2)
studentModel = fullModel(ResNext, convertResNext, toDF, toPred, toRF, toPred2)
binaryCrossEntropyLoss = nn.BCELoss()
MSELoss = nn.MSELoss()
learning_rate = 0.05
batch_size = 6
labeled_bs = 61
optimizer = torch.optim.SGD(teacherModel.parameters(), lr=learning_rate)
dataset = CustomImageDataset(groundTruthShadowsDirectory = "UCF\GroundTruth", groundTruthShadowEdgesDirectory= "UCF\EdgeMasks", imagesDirectory = "UCF\InputImages")
# labelled, unlabelled = getIndices(dataset)
teacherModel.train()


if(len(unlabelled) > 0):
    batch_sampler = TwoStreamBatchSampler(labelled, unlabelled, batch_size, batch_size - labeled_bs)
else:
    batch_sampler = BatchSampler(SequentialSampler(range(10)), batch_size=batch_size, drop_last=False)
trainloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainloader, teacherModel, studentModel, binaryCrossEntropyLoss, MSELoss, optimizer)
    # test_loop(test_dataloader, model, loss_fn)
print("Done!")
import math
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
import utils
import network
from utils import losses
from utils import util
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.SBU import SBU, relabel_dataset
from dataloaders import joint_transforms
import gc

train_data_path = 'C:/Users/idvin/Documents/computerVision/ShadowPrediction/SBU-shadow/SBUTrain4KRecoveredSmall'
scale = 416
batch_size = 2
max_iterations = 10000
lr_decay = float(0.9)
base_lr = float(0.005)
labeled_bs = 2
consistency = 1
ema_decay = float(0.99)

if __name__ == "__main__":
    torch.cuda.empty_cache()

    model = network.MTMT().cuda()
    # ema_model = network.MTMT().cuda()
    # for param in ema_model.parameters():
    #     param.detach_()

    joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.Resize((scale, scale))
    ])
    val_joint_transform = joint_transforms.Compose([
        joint_transforms.Resize((scale, scale))
    ])

    target_transform = transforms.ToTensor()

    db_train = SBU(root=train_data_path, joint_transform=joint_transform, transform=transforms.ToTensor(), target_transform=target_transform, mod='union', edge=True)
    labeled_idxs, unlabeled_idxs = relabel_dataset(db_train, edge_able=True)
    batch_sampler = utils.util.TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler)


    model.train()
    # ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    consistency_criterion = utils.losses.sigmoid_mse_loss

    iter_num = 0
    # max_epoch = max_iterations//len(trainloader)+1
    print(len(trainloader))
    max_epoch = 10000
    # model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            print(i_batch)
            optimizer.param_groups[0]['lr'] = 2 * base_lr * (1 - float(iter_num) / max_iterations
                                                             ) ** lr_decay
            image_batch, label_batch, edge_batch, number_per_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda(), sampled_batch['edge'].cuda(), sampled_batch['number_per'].cuda()

            noise = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.2, 0.2)
            ema_inputs = image_batch + noise
            edge, up_shadow, SC, up_shadow_final = model(image_batch)

            # with torch.no_grad():
            #     edge_ema, up_shadow_ema, SC_ema, up_shadow_final_ema = ema_model(ema_inputs)

            gc.collect()
            ## calculate subitizing loss and subitizing consistency loss
            subitizing_loss = utils.losses.sigmoid_mse_loss(SC[:labeled_bs], number_per_batch[:labeled_bs])
            # subitizing_con_loss = utils.losses.sigmoid_mse_loss(SC[labeled_bs:], SC_ema[labeled_bs:])
            ## edge loss
            edge_loss = []
            edge_con_loss = []
            # for (ix, ix_ema) in zip(edge, edge_ema):
            for ix in edge:
                edge_loss.append(utils.losses.bce2d_new(ix[:labeled_bs], edge_batch[:labeled_bs], reduction='mean'))
                # edge_con_loss.append(consistency_criterion(ix[labeled_bs:], ix_ema[labeled_bs:]))
            edge_loss = sum(edge_loss)
            # edge_con_loss = sum(edge_con_loss)
            shadow_loss1 = []
            shadow_loss2 = []
            shadow_con_loss1 = []
            shadow_con_loss2 = []
            # for (ix, ix_ema) in zip(up_shadow, up_shadow_ema):
            for ix in up_shadow:
                shadow_loss1.append(F.binary_cross_entropy_with_logits(ix[:labeled_bs], label_batch[:labeled_bs], reduction='mean'))
                # shadow_con_loss1.append(consistency_criterion(ix[labeled_bs:], ix_ema[labeled_bs:]))

            # for (ix, ix_ema) in zip(up_shadow_final, up_shadow_final_ema):
            for ix in up_shadow_final:
                shadow_loss2.append(F.binary_cross_entropy_with_logits(ix[:labeled_bs], label_batch[:labeled_bs], reduction='mean'))
                # shadow_con_loss2.append(consistency_criterion(ix[labeled_bs:], ix_ema[labeled_bs:]))

            shadow_loss = sum(shadow_loss1) + sum(shadow_loss2)
            supervised_loss = shadow_loss + edge_loss * 10 + subitizing_loss * 1

            # consistency_weight = 10 * pow(math.e, (-5*(1-float(iter_num) / max_iterations)**2))
            # consistency_loss = consistency_weight * (edge_con_loss + sum(shadow_con_loss1) + sum(shadow_con_loss2) + subitizing_con_loss)

            # loss = supervised_loss + consistency_loss
            loss = supervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #     ema_param.data.mul_(1-ema_decay).add_(param.data.mul_(ema_decay))


            iter_num = iter_num + 1

            if iter_num > max_iterations:
                break


    torch.save(model.state_dict(), "models/model.pth")

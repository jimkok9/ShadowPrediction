import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import utils.losses as losses
import utils.ramps as ramps
from utils.util import AverageMeter, TwoStreamBatchSampler
from main import MTMT
from dataloaders.SBU import SBU, relabel_dataset
from dataloaders import joint_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='C:/Users\idvin\Documents\computerVision\ShadowPrediction\SBU-shadow', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='MTMT', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.005, help='maximum epoch number to train')
parser.add_argument('--lr_decay', type=float,  default=0.9, help='learning rate decay')
parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
parser.add_argument('--deterministic', type=int,  default=0, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=7.0, help='consistency_rampup')
parser.add_argument('--scale', type=int,  default=416, help='batch size of 8 with resolution of 416*416 is exactly OK')
parser.add_argument('--subitizing', type=float,  default=1, help='subitizing loss weight')
parser.add_argument('--repeat', type=int,  default=3, help='repeat')
args = parser.parse_args()
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
train_data_path = args.root_path
tmp_path = 'tmp_see'
if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)
print("---love--")
print("number of devices: " + str(torch.cuda.device_count()))
print("cuda available: " + str(torch.cuda.is_available()))
print("device: " + str(torch.cuda.current_device()))
print("---ueself--")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
lr_decay = args.lr_decay
loss_record = 0

cudnn.benchmark = False
cudnn.deterministic = True

num_classes = 2

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def create_model():
    net = MTMT()
    print("---begin---")
    print("number of devices: " + str(torch.cuda.device_count()))
    print("cuda available: " + str(torch.cuda.is_available()))
    print("device: " + str(torch.cuda.current_device()))
    # net.to("cuda")
    # device = torch.device('cuda')
    # net = net.to(device)
    # # net_cuda.checkCuda()
    return net


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    device = torch.device('cuda')
    model = create_model()
    model = model.to(device)
    ema_model = create_model()
    ema_model = ema_model.to(device)

    joint_transform = joint_transforms.Compose([
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.Resize((args.scale, args.scale))
    ])
    val_joint_transform = joint_transforms.Compose([
        joint_transforms.Resize((args.scale, args.scale))
    ])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    db_train = SBU(root=train_data_path, joint_transform=joint_transform, transform=img_transform, target_transform=target_transform, mod='union', edge=True)
    labeled_idxs, unlabeled_idxs = relabel_dataset(db_train, edge_able=True)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    # def worker_init_fn(worker_id):
    #     random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)

    model.train()
    ema_model.train()
    # ema_model.eval()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * base_lr},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': base_lr, 'weight_decay': 0.0001}
    ], momentum=0.9)


    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
        # consistency_criterion = utils.losses.sigmoid_mse_loss
    elif args.consistency_type == 'kl':
        # consistency_criterion = losses.softmax_kl_loss
        consistency_criterion = F.kl_div
    else:
        assert False, args.consistency_type

    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    print("training beginining")
    model.train()
    for epoch_num in range(max_epoch):

        time1 = time.time()
        shadow_loss2_record, shadow_con_loss2_record, edge_loss_record, edge_con_loss_record = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        subitizing_loss_record = AverageMeter()
        # loss2_h2l_record, loss3_h2l_record, loss4_h2l_record = AverageMeter(), AverageMeter(), AverageMeter()
        # loss1_l2h_record, loss2_l2h_record, loss3_l2h_record = AverageMeter(), AverageMeter(), AverageMeter()
        # loss4_l2h_record, consistency_record = AverageMeter(), AverageMeter()
        print("---before--")
        print("number of devices: " + str(torch.cuda.device_count()))
        print("cuda available: " + str(torch.cuda.is_available()))
        print("device: " + str(torch.cuda.current_device()))
        print("---trainloader--")
        print("epoch number: " + str(epoch_num))
        for i_batch, sampled_batch in enumerate(trainloader):
            print("batch num: " + str(i_batch))
            time2 = time.time()
            optimizer.param_groups[0]['lr'] = 2 * base_lr * (1 - float(iter_num) / max_iterations
                                                             ) ** lr_decay
            optimizer.param_groups[1]['lr'] = base_lr * (1 - float(iter_num) / max_iterations
                                                         ) ** lr_decay
            # print('fetch data cost {}'.format(time2-time1))
            device = torch.device('cuda')
            image_batch, label_batch, edge_batch, number_per_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['edge'], sampled_batch['number_per']
            image_batch, label_batch, edge_batch, number_per_batch = image_batch.to(device), label_batch.to(device), edge_batch.to(device), number_per_batch.to(device)
            noise = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.2, 0.2)
            ema_inputs = image_batch + noise
            ema_inputs = ema_inputs.to(device)

            print("the batches m8")
            print(image_batch.size())
            print(ema_inputs.size())
            print("the batches ")

            up_edge, up_shadow, up_subitizing, up_shadow_final = model(image_batch)
            with torch.no_grad():
                up_edge_ema, up_shadow_ema, up_subitizing_ema, up_shadow_final_ema = ema_model(ema_inputs)

            print("output sizes m8")
            print(up_edge[0].size())
            print(up_edge_ema[0].size())
            print("the outputs")

            ## calculate the loss
            ## subitizing loss
            subitizing_loss = losses.sigmoid_mse_loss(up_subitizing[:labeled_bs], number_per_batch[:labeled_bs])
            subitizing_con_loss = losses.sigmoid_mse_loss(up_subitizing[labeled_bs:], up_subitizing_ema[labeled_bs:])
            ## edge loss
            edge_loss = []
            edge_con_loss = []
            for (ix, ix_ema) in zip(up_edge, up_edge_ema):
                print(ix.size())
                print(edge_batch.size())
                print(labeled_bs)
                edge_loss.append(losses.bce2d_new(ix[:labeled_bs], edge_batch[:labeled_bs], reduction='mean'))
                edge_con_loss.append(consistency_criterion(ix[labeled_bs:], ix_ema[labeled_bs:]))
            edge_loss = sum(edge_loss)
            edge_con_loss = sum(edge_con_loss)
            shadow_loss1 = []
            shadow_loss2 = []
            shadow_con_loss1 = []
            shadow_con_loss2 = []
            for (ix, ix_ema) in zip(up_shadow, up_shadow_ema):
                print(len)
                shadow_loss1.append(F.binary_cross_entropy_with_logits(ix[:labeled_bs], label_batch[:labeled_bs], reduction='mean'))
                shadow_con_loss1.append(consistency_criterion(ix[labeled_bs:], ix_ema[labeled_bs:]))

            for (ix, ix_ema) in zip(up_shadow_final, up_shadow_final_ema):
                shadow_loss2.append(F.binary_cross_entropy_with_logits(ix[:labeled_bs], label_batch[:labeled_bs], reduction='mean'))
                shadow_con_loss2.append(consistency_criterion(ix[labeled_bs:], ix_ema[labeled_bs:]))

            shadow_loss = sum(shadow_loss1) + sum(shadow_loss2)
            supervised_loss = shadow_loss + edge_loss * args.edge + subitizing_loss*args.subitizing

            consistency_weight = get_current_consistency_weight(epoch_num)
            consistency_loss = consistency_weight * (edge_con_loss + sum(shadow_con_loss1) + sum(shadow_con_loss2) + subitizing_con_loss)

            loss = supervised_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
           # writer.add_scalar('lr', lr_, iter_num)
           # writer.add_scalar('loss/loss', loss, iter_num)

            # loss_all_record.update(loss.item(), batch_size)
            shadow_loss2_record.update(shadow_loss2[-1].item(), labeled_bs)
            edge_loss_record.update(edge_loss.item(), labeled_bs)
            shadow_con_loss2_record.update(shadow_con_loss2[-1].item(), batch_size-labeled_bs)
            edge_con_loss_record.update(edge_con_loss.item(), batch_size-labeled_bs)
            subitizing_loss_record.update(subitizing_loss, labeled_bs)

            logging.info('iteration %d : shadow_f : %f5 , edge: %f5 , subitizing: %f5, shadow_f_con: %f5  edge_con: %f5 loss_weight: %f5, lr: %f5' %
                         (iter_num, shadow_loss2_record.avg, edge_loss_record.avg, subitizing_loss_record.avg, shadow_con_loss2_record.avg,edge_con_loss_record.avg, consistency_weight, optimizer.param_groups[1]['lr']))
            loss_record = 'iteration %d : shadow_f : %f5 , edge: %f5 , shadow_f_con: %f5  edge_con: %f5 loss_weight: %f5, lr: %f5' % \
                          (iter_num, shadow_loss2_record.avg, edge_loss_record.avg, shadow_con_loss2_record.avg,edge_con_loss_record.avg, consistency_weight, optimizer.param_groups[1]['lr'])

            if iter_num % 200 == 0:
                vutils.save_image(torch.sigmoid(up_shadow_final[-1].data), tmp_path + '/iter%d-d_predict_f.jpg' % iter_num, normalize=True,
                                  padding=0)
                vutils.save_image(torch.sigmoid(up_shadow_final_ema[-1].data),
                                  tmp_path + '/iter%d-e_predict_f.jpg' % iter_num, normalize=True,
                                  padding=0)
                vutils.save_image(torch.sigmoid(up_shadow[-1].data), tmp_path + '/iter%d-c_predict.jpg' % iter_num,
                                  normalize=True,
                                  padding=0)
                vutils.save_image(torch.sigmoid(up_edge[-1].data), tmp_path + '/iter%d-g_edge.jpg' % iter_num,
                                  normalize=True, padding=0)
                vutils.save_image(torch.sigmoid(up_edge_ema[-1].data), tmp_path + '/iter%d-h_edge.jpg' % iter_num,
                                  normalize=True, padding=0)
                vutils.save_image(image_batch.data, tmp_path + '/iter%d-a_shadow-data.jpg' % iter_num, padding=0)
                vutils.save_image(label_batch.data, tmp_path + '/iter%d-b_shadow-target.jpg' % iter_num, padding=0)
                vutils.save_image(edge_batch.data, tmp_path + '/iter%d-f_edge-target.jpg' % iter_num, padding=0)
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    # save_mode_path_ema = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '_ema.pth')
    torch.save(model.state_dict(), save_mode_path)
    # torch.save(ema_model.state_dict(), save_mode_path_ema)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
    with open('record/loss_record_MTMT.txt', 'a') as f:
        f.write(snapshot_path+' ')
        f.write(str(loss_record)+'\r\n')
import os
import argparse
import torch
from test_MT_util import test_all_case
# from networks.EGNet import build_model
from main import MTMT
# from networks.EGNet_onlyDSS import build_model
# from networks.EGNet_task3 import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='D:/ISTD_Dataset/test', help='Name of Experiment')
#parser.add_argument('--root_path', type=str, default='/home/ext/chenzhihao/Datasets/UCF', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='D:/SBU-shadow/SBU-Test', help='Name of Experiment')
# parser.add_argument('--target_path', type=str, default='C:/Users/idvin/Documents/computerVision/ShadowPrediction/SBU-shadow/SBU-Test', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='EGNet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--base_lr', type=float,  default=0.005, help='base learning rate')
parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
parser.add_argument('--epoch_name', type=str,  default='iter_7000.pth', help='choose one epoch/iter as pretained')
parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
parser.add_argument('--scale', type=int,  default=416, help='batch size of 8 with resolution of 416*416 is exactly OK')
parser.add_argument('--subitizing', type=float,  default=5.0, help='subitizing loss weight')
parser.add_argument('--repeat', type=int,  default=6, help='repeat')


FLAGS = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
# snapshot_path = os.path.join('/home/chenzhihao/shadow_detection/shadow-MT/model_SBU_EGNet', 'baseline_edgeX5_C64', FLAGS.epoch_name)
# test_save_path = os.path.join('/home/chenzhihao/shadow_detection/shadow-MT/model_SBU_EGNet', 'baseline_edgeX5_C64', 'prediction')
# snapshot_path = "../model_SBU_EGNet/baselineC64_DSS/"+str(FLAGS.edge)+str(FLAGS.base_lr)+'/'+FLAGS.epoch_name

# snapshot_path = "../model_SBU_EGNet/baselineC64_DSS_unlabelcons/repeat"+str(FLAGS.repeat)+'_edge'+str(FLAGS.edge)+'lr'+str(FLAGS.base_lr)+'consistency'+str(FLAGS.consistency)+'subitizing'+str(FLAGS.subitizing)+'/'+FLAGS.epoch_name
# snapshot_path = "../model_SBU_EGNet/baselineC64_DSS_unlabelcons/"+'edge'+str(FLAGS.edge)+'lr'+str(FLAGS.base_lr)+'consistency'+str(FLAGS.consistency)+'/'+FLAGS.epoch_name
# test_save_path = "../model_SBU_EGNet/baselineC64_DSS_unlabelcons/prediction_SBU_sub"
# test_save_path = "/home/chenzhihao/shadow_detection/EGNet/prediction"+FLAGS.model+"_post/"
# snapshot_path = "../model_SBU_EGNet_ablation/onlyDSS/"+FLAGS.epoch_name
# snapshot_path = "../model_SBU_EGNet_ablation/meanteacher/consistency"+str(FLAGS.consistency)+"/"+FLAGS.epoch_name # meanteacher
# snapshot_path = '../model_SBU_EGNet/baselineC64_DSS/10.00.005/iter_7000.pth' # multi-tasks
# test_save_path = '../model_SBU_EGNet_ablation/multi-task/prediction'
# snapshot_path = 'D:/computerVisionModels/UCF_iter_5000.pth'
snapshot_path = 'D:/computerVisionModels/ISTD_Jim/ISTD_iter_10000.pth'
# snapshot_path = "../model_ISTD_EGNet/salience/iter_3000.pth"
# test_save_path = 'C:/Users/idvin/Documents/computerVision/ShadowPrediction/test/'
test_save_path = 'D:/ComputerVisionTests/ISTD'
# test_save_path = 'C:/Users/idvin/Documents/computerVision/ShadowPrediction/testUCF/'
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(snapshot_path)
num_classes = 1

datafolder = FLAGS.root_path + "/ShadowImages"
# datafolder = "C:/Users/idvin/Documents/computerVision/ShadowPrediction/SBU-shadow/SBU-Test/ShadowImages"

img_list = os.listdir(datafolder)
print("here!!!!")
print(len(img_list))
print(img_list)
img_list = [x[:-4] for x in img_list if '.jpg' in x]
data_path = [(os.path.join(FLAGS.root_path, 'ShadowImages', img_name + '.jpg'),
             os.path.join(FLAGS.root_path, 'ShadowMasks', img_name + '.png'))
            for img_name in img_list]


# target_path = "C:\Users\idvin\Documents\computerVision\ShadowPrediction\test"
data_path = [(FLAGS.root_path + '/ShadowImages/' + img_name + ".jpg",
             FLAGS.root_path + '/ShadowMasks/' + img_name + ".png")
            for img_name in img_list]
print("paths")
print(data_path)

def test_calculate_metric():
    net = MTMT().cuda()
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(snapshot_path))
    net.eval()

    avg_metric = test_all_case(net, data_path, num_classes=num_classes,
                               save_result=True, test_save_path=test_save_path, trans_scale=416)
    print(avg_metric)

    return avg_metric


def test_calculate_metric(snapshot_path,data_path,test_save_path):
    datafolder = data_path + "/ShadowImages"
    img_list = os.listdir(datafolder)
    img_list = [x[:-4] for x in img_list if '.jpg' in x]

    data_path = [(data_path + '/ShadowImages/' + img_name + ".jpg",
                  data_path + '/ShadowMasks/' + img_name + ".png")
                 for img_name in img_list]
    net = MTMT().cuda()
    net.load_state_dict(torch.load(snapshot_path))
    print("init weight from {}".format(snapshot_path))
    net.eval()

    avg_metric = test_all_case(net, data_path,
                               save_result=False, test_save_path=test_save_path, trans_scale=FLAGS.scale)
    print(avg_metric)
    del net
    return avg_metric

if __name__ == '__main__':
    metric = test_calculate_metric()
    with open('record/test_record_EGNet_meanteacher.txt', 'a') as f:
        f.write(snapshot_path+' ')
        f.write(str(metric)+' --UCF\r\n')
    print('Test ber results: {}'.format(metric))

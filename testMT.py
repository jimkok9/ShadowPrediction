import os
import torch
from network import MTMT
from test_MT_util import test_all_case

# root_path = 'D:/ISTD_Dataset/test/'
root_path = "C:/Users/idvin/Documents/computerVision/ShadowPrediction/SBU-shadow/SBU-Test/"
# snapshot_path = 'D:/computerVisionModels/ISTD/ISTD_iter_10000.pth'
snapshot_path = 'D:/computerVisionModels/SBU/SBU_iter_10000.pth'
test_save_path = 'D:/ComputerVisionTests/SBU/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

scale = 416
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
num_classes = 2

img_list = [os.path.splitext(f)[0] for f in os.listdir(root_path + 'ShadowImages/') if f.endswith('.jpg')]
data_path = [(root_path + 'ShadowImages/' + img_name + '.jpg',
             root_path + 'ShadowMasks/' + img_name + '.png')
            for img_name in img_list]


def test_calculate_metric():
    net = MTMT().cuda()
    net.load_state_dict(torch.load(snapshot_path))
    net.eval()

    avg_metric = test_all_case(net, data_path, num_classes=num_classes,
                               save_result=True, test_save_path=test_save_path, trans_scale=scale)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    with open('record/test.txt', 'a') as f:
        f.write(snapshot_path+' ')
        f.write(str(metric)+' --UCF\r\n')
    print('Test ber results: {}'.format(metric))
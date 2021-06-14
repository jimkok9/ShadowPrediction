import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import os
from Utils.util import crf_refine


def test_all_case(net, image_list, num_classes=2, save_result=True, test_save_path=None, trans_scale=416, GT_access=True):
    img_transform = transforms.Compose([
        transforms.Resize((trans_scale, trans_scale)),
        transforms.ToTensor(),
    ])
    to_pil = transforms.ToPILImage()
    TP, TN, Np, Nn = 0, 0, 0, 0
    ber_mean = 0
    for (img_path, target_path) in tqdm(image_list):
        print(img_path)
        img_name = img_path.split("/")[-1]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_var = img_transform(img).unsqueeze(0).cuda()
        up_edge, up_shadow, up_subitizing, up_shadow_final = net(img_var)

        prediction = np.array(to_pil(up_shadow_final[-1].data.squeeze(0).cpu()))
        prediction = np.uint8(prediction>=127.5)*255 # trick just for SBU
        prediction = crf_refine(np.array(img.convert('RGB').resize((trans_scale, trans_scale))), prediction)
        prediction = np.array(transforms.Resize((h, w))(Image.fromarray(prediction.astype('uint8')).convert('L')))

        # cal metric
        if GT_access:
            target = np.array(Image.open(target_path).convert('L'))
            TP_single, TN_single, Np_single, Nn_single, Union = cal_acc(prediction, target)
            ''' Calculate BER '''
            TP = TP + TP_single
            TN = TN + TN_single
            Np = Np + Np_single
            Nn = Nn + Nn_single
            ber_shadow = (1 - TP / Np) * 100
            ber_unshadow = (1 - TN / Nn) * 100
            ber_mean = 0.5 * (2 - TP / Np - TN / Nn) * 100
            print("Current ber is {}, shadow_ber is {}, unshadow ber is {}".format(ber_mean, ber_shadow, ber_unshadow))
        '''Save prediction'''
        if save_result:
            Image.fromarray(prediction).save(os.path.join(test_save_path, img_name[:-4]+'.png'), "PNG")

    return ber_mean, ber_shadow, ber_unshadow

def cal_acc(prediction, label, thr = 127.5):
    prediction = (np.uint8(prediction) > thr)
    label = (label > thr)
    prediction_tmp = prediction.astype(np.float)
    label_tmp = label.astype(np.float)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1-label_tmp))
    Union = np.sum(prediction_tmp) + Np - TP

    return TP, TN, Np, Nn, Union
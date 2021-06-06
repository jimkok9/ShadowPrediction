import torch
from network import MTMT
from PIL import Image
from torchvision import transforms
import numpy as np
from Utils.util import crf_refine
import torchvision.utils as vutils

img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

if __name__ == "__main__":
    net = MTMT().cuda()
    net.load_state_dict(torch.load("models/SBU10000Final.pth"))

    img = Image.open("C:/Users/Jim Kok/Desktop/SBU-shadow/SBUTrain4KRecoveredSmall\ShadowImages\lssd7.jpg").convert('RGB')
    w, h = img.size

    img_var = img_transform(img).unsqueeze(0).cuda()
    img_var = img_var * 255

    edge, shadows, SC, output = net(img_var)
    #[pred0], shadows, SC, [RF_S_f]
    res = output[-1]
    print(res)
    to_pil = transforms.ToPILImage()
    prediction = np.array(to_pil(res.data.squeeze(0).cpu()))
    print(prediction)
    prediction = np.uint8(prediction>=110)*255 # trick just for SBU
    print(prediction)
    prediction = crf_refine(np.array(img.convert('RGB').resize((416, 416))), prediction)
    prediction = np.array(transforms.Resize((h, w))(Image.fromarray(prediction.astype('uint8')).convert('L')))
    print(prediction)
    vutils.save_image(torch.from_numpy(prediction), 'rtest.jpg',  normalize=True, padding=0)
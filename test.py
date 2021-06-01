import torch
from network import MTMT
from PIL import Image
from torchvision import transforms
import numpy as np

img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])

if __name__ == "__main__":
    net = MTMT().cuda()
    net.load_state_dict(torch.load("models/SBU.pth"))

    img = Image.open("C:/Users/Jim Kok/Desktop/SBU-shadow/SBUTrain4KRecoveredSmall\ShadowImages\lssd5.jpg").convert('RGB')
    w, h = img.size

    img_var = img_transform(img).unsqueeze(0).cuda()
    img_var = img_var * 255
    print(img_var)

    edge, shadows, SC, output = net(img_var)
    #[pred0], shadows, SC, [RF_S_f]
    print(SC)
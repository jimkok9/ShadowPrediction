import sys, os
from PIL import Image
from resnext101_regular import ResNeXt101
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    img = Image.open("UCF/InputImages/04822305_2305_3329_2561_3585.jpg")
    img = ToTensor()(img).unsqueeze(0)
    print(img.shape)

    ResNext = ResNeXt101()
    EF5 = ResNext.forward(img)
    print(EF5.shape)


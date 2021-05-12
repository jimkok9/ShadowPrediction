from resnext101_regular import *
from PIL import Image


if __name__ == "__main__":
    EF = resnext101_regular.ResNeXt101()
    img = Image.open("UCF/InputImages/006.jpg")
    print(img)


from .resnext101_regular import ResNeXt101
import cv2

if __name__ == "__main__":
    EF = ResNeXt101()
    img = cv2.imread("UCF/InputImages/006.jpg")
    EF1 = EF.layer0()

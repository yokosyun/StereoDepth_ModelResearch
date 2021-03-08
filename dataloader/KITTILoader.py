import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess
import torch.nn as nn
from torchvision.utils import save_image

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert("RGB")


def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(
        self,
        left,
        right,
        left_disparity,
        training,
        loader=default_loader,
        dploader=disparity_loader,
    ):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            # th, tw = 368, 1232
            th, tw = 256, 512

            if True:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)

                left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
                right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

                dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
                dataL = dataL[y1 : y1 + th, x1 : x1 + tw]
            else:
                left_img = left_img.crop((w - tw, h - th, w, h))
                right_img = right_img.crop((w - tw, h - th, w, h))

                dataL = dataL.crop((w - tw, h - th, w, h))
                dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            if False:
                left_img = transforms.ToTensor()(left_img)
                right_img = transforms.ToTensor()(right_img)
            else:
                # normalize = {
                #     "mean": [0.485, 0.456, 0.406],
                #     "std": [0.229, 0.224, 0.225],
                # }
                # t_list = [
                #     transforms.ToTensor(),
                #     transforms.Normalize(**normalize),
                # ]
                # tmp = transforms.Compose(t_list)(left_img)
                # save_image(tmp, "tmp.png")

                processed = preprocess.get_transform(augment=False)
                left_img = processed(left_img)
                right_img = processed(right_img)

            # print("torch.max(left_img)=", torch.max(left_img))
            # print("torch.min(left_img)=", torch.min(left_img))

            # save_image(left_img, "left_img_org.png")
            # save_image((left_img + 1) / torch.max(left_img + 1), "left_img_max.png")
            # save_image((left_img + 1) / 2.0, "left_img.png")
            # save_image(left_img_processed, "left_img_processed.png")

            return left_img, right_img, dataL
        else:
            w, h = left_img.size

            left_img = left_img.crop((w - 1232, h - 368, w, h))
            right_img = right_img.crop((w - 1232, h - 368, w, h))
            w1, h1 = left_img.size

            dataL = dataL.crop((w - 1232, h - 368, w, h))
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            if False:
                left_img = transforms.ToTensor()(left_img)
                right_img = transforms.ToTensor()(right_img)
            else:
                processed = preprocess.get_transform(augment=False)
                left_img = processed(left_img)
                right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)

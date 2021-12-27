import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
import numpy as np


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5), padding=(1, 1))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        # output = output.view(img.size(0), -1)
        output = output.view(-1, 120)
        output = self.f4(output)
        output = self.f5(output)
        return output


class NumberImgLoader(Dataset):
    def default_loader(self, bgr_im):
        im_tensor = torch.from_numpy(bgr_im).float().unsqueeze(-1)
        im_tensor = im_tensor.permute(2, 0, 1)
        return im_tensor / 255.

    def __init__(self, root_dir=""):
        self.all_fulpathes = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                ful_path = os.path.join(root, file)
                if os.path.splitext(ful_path)[1] in ['.jpg', '.png']:
                    self.all_fulpathes.append(ful_path)
        return

    def __getitem__(self, idx):
        bgr_im = cv2.imread(self.all_fulpathes[idx], cv2.IMREAD_GRAYSCALE)
        return self.default_loader(bgr_im)

    def __len__(self):
        return len(self.all_fulpathes)


def visualise_im_tensor(im_tensor, idx):
    single_im = im_tensor[idx]
    single_im = single_im.permute(1, 2, 0) * 255.
    single_im = single_im.numpy().astype(np.uint8)

    cv2.imshow("im", single_im)
    cv2.waitKey(0)


if __name__ == '__main__':
    model = LeNet5()

    number_set = NumberImgLoader(root_dir="../ch01/mnist_playground_ims/")
    number_loader = DataLoader(number_set, batch_size=8, shuffle=True, num_workers=0)

    whole_iternum = len(number_loader)
    for i, imgs in enumerate(number_loader):
        out = model(imgs)
        print("iter:[{}/{}] out shape: {}".format(i, whole_iternum, out.shape))
        visualise_im_tensor(imgs, 0)

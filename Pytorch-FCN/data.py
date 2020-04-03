import pandas as pd
import os
import torch as t
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utiles import functional as ff
import skimage.transform
import numpy


TRAIN_ROOT = 'D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/CamVid/train' #/xxx/CamVid/train  #训练集的路径
TRAIN_LABEL = 'D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/CamVid/train_labels' #/xxx/CamVid/train_labels #训练集的标签路径
TEST_ROOT = 'D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/CamVid/val'  #/xxx/CamVid/val  #验证集路径
TEST_LABEL = 'D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/CamVid/val_labels' #/xxx/CamVid/val_labels #验证集标签路径

train_imgs = os.listdir(TRAIN_ROOT)
train_imgs = [os.path.join(TRAIN_ROOT, img) for img in train_imgs]
train_imgs.sort()

train_labels = os.listdir(TRAIN_LABEL)
train_labels = [os.path.join(TRAIN_LABEL, label) for label in train_labels]
train_labels.sort()

test_imgs = os.listdir(TEST_ROOT)
test_imgs = [os.path.join(TEST_ROOT, img) for img in test_imgs]
test_imgs.sort()

test_labels = os.listdir(TEST_LABEL)
test_labels = [os.path.join(TEST_LABEL, label) for label in test_labels]
test_labels.sort()


class FixedCrop(object):
    """
    Args:
        img (PIL Image): Image to be cropped.
        i, j, h, w (int): Image position to be cropped
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """

    def __init__(self, i, j, h, w, padding=0):
        self.i = i
        self.j = j
        self.h = h
        self.w = w
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        return ff.crop(img, self.i, self.j, self.h, self.w)


pd_label_color = pd.read_csv('D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/CamVid/class_dict.csv', sep=',')  #/media/zjy/shuju/CamVid_2D/CamVid/class_dict.csv
name_value = pd_label_color['name'].values  # ndarray type
num_class = len(name_value)

colormap = []
for i in range(len(pd_label_color.index)):
    # 通过行号索引行数据
    tmp = pd_label_color.iloc[i]
    color = []
    color.append(tmp['r'])
    color.append(tmp['g'])
    color.append(tmp['b'])
    colormap.append(color)


def center_crop(data, label, crop_size):
    height, width = crop_size 
    data, rect1 = ff.center_crop(data, (height, width))
    label = FixedCrop(*rect1)(label)

    return data, label


cm2lbl = np.zeros(256 ** 3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(img):
    data = np.array(img, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')


def img_transform(img, label, crop_size):
    img, label = center_crop(img, label, crop_size)
    label = numpy.array(label)
    label1 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2), order=0, mode='reflect',
                                      preserve_range=True)

    label2 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4), order=0, mode='reflect',
                                      preserve_range=True)

    label3 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8), order=0, mode='reflect',
                                      preserve_range=True)

    label4 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16), order=0, mode='reflect',
                                      preserve_range=True)

    label5 = skimage.transform.resize(label, (label.shape[0] // 32, label.shape[1] // 32), order=0, mode='reflect',
                                      preserve_range=True)

    label = Image.fromarray(label.astype('uint8'))
    label1 = Image.fromarray(label1.astype('uint8'))
    label2 = Image.fromarray(label2.astype('uint8'))
    label3 = Image.fromarray(label3.astype('uint8'))
    label4 = Image.fromarray(label4.astype('uint8'))
    label5 = Image.fromarray(label5.astype('uint8'))

    transform_img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    img = transform_img(img)
    label = image2label(label)
    label1 = image2label(label1)
    label2 = image2label(label2)
    label3 = image2label(label3)
    label4 = image2label(label4)
    label5 = image2label(label5)

    label = t.from_numpy(label)
    label1 = t.from_numpy(label1)
    label2 = t.from_numpy(label2)
    label3 = t.from_numpy(label3)
    label4 = t.from_numpy(label4)
    label5 = t.from_numpy(label5)

    return img, label, label1, label2, label3, label4, label5


class CamvidDataset(Dataset):
    def __init__(self, train=True, crop_size=None, transform=None):
        self.train = train
        self.train_imgs = train_imgs
        self.train_labels = train_labels

        self.test_imgs = test_imgs
        self.test_labels = test_labels

        if self.train:
            self.imgs = self.train_imgs
            self.labels = self.train_labels
        else:
            self.imgs = self.test_imgs
            self.labels = self.test_labels

        self.crop_size = crop_size
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label, label1, label2, label3, label4, label5 = self.transforms(img, label, self.crop_size)

        sample = {'img': img, 'label': label, 'label1': label1, 'label2': label2, 'label3': label3, 'label4': label4,
                  'label5': label5}

        return sample

    def __len__(self):
        return len(self.imgs)


input_size = (352, 480)
Cam_train = CamvidDataset(True, input_size, img_transform)
Cam_val = CamvidDataset(False, input_size, img_transform)


import os
import pandas as pd
from torch.utils.data import Dataset
from utiles.data import img_transform
from PIL import Image
import torch as t
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from FCN import FCN8s,VGGNet 


TEST_ROOT = 'D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/CamVid/test'  #'/xxx/CamVid/test'  #测试集路径
TEST_LABEL = 'D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/CamVid/test_labels'  #'/xxx/CamVid/test_labels' #测试集标签路径

imgs = os.listdir(TEST_ROOT)
imgs = [os.path.join(TEST_ROOT, img) for img in imgs]
imgs.sort()

labels = os.listdir(TEST_LABEL)
labels = [os.path.join(TEST_LABEL, label) for label in labels]
labels.sort()


input_size = (352, 480) #height width


class TestDataset(Dataset):
    def __init__(self, transform, crop_size):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transform
        self.crop_size = crop_size

    def __getitem__(self, index):
        img, label = Image.open(self.imgs[index]), Image.open(self.labels[index]).convert('RGB')
        #img, label = self.transforms(img, label, self.crop_size) 
        img, label, label1, label2, label3, label4, label5 = self.transforms(img, label, self.crop_size) #改
        #sample = {'img': img, 'label': label}
        sample = {'img': img, 'label': label, 'label1': label1, 'label2': label2, 'label3': label3, 'label4': label4,
                  'label5': label5} #改
        return sample

    def __len__(self):
        return len(self.imgs)


test_dataset = TestDataset(img_transform, input_size)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 


vgg_model = VGGNet(requires_grad=True) 
net = FCN8s(pretrained_net=vgg_model,n_class=12).cuda() 
net.load_state_dict(t.load("D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/logs/last.pth")) #模型加载路径
net.eval()

pd_label_color = pd.read_csv('D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/CamVid/class_dict.csv', sep=',') #CSV路径
name_value = pd_label_color['name'].values
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

cm = np.array(colormap).astype('uint8')

dir = "D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/imgs/pic" #保存图像的路径

for i, sample in enumerate(test_data):
    valImg = sample['img'].cuda()
    valLabel = sample['label'].long().cuda()
    out = net(valImg)
    out = F.log_softmax(out, dim=1)
    pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
    pre = cm[pre_label]
    pre1 = Image.fromarray(pre)
    pre1.save(dir + str(i) + '.png')
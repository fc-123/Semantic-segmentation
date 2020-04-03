from torch.autograd import Variable
from torch.utils.data import DataLoader
from utiles.evalution_segmentaion import eval_semantic_segmentation
import torch.nn.functional as F
import torch as t
from predict import test_dataset
from FCN import FCN8s,VGGNet 

BATCH_SIZE = 2
miou_list = [0]
test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 

vgg_model = VGGNet(requires_grad=True) 
net = FCN8s(pretrained_net=vgg_model,n_class=12)
net.eval()
net.cuda()
net.load_state_dict(t.load('D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/logs/last.pth'))  #加载模型

train_acc = 0
train_miou = 0
train_class_acc = 0
train_mpa = 0
error = 0


for i, sample in enumerate(test_data): #(data, label)-->sample
    # data = Variable(data).cuda()
    # label = Variable(label).cuda()
    # out = net(data)
    # out = F.log_softmax(out, dim=1)
    
    #我认为增添的
    data = sample['img'].cuda() #valImg= --> data
    label = sample['label'].long().cuda() #valLabel= --> label=
    out = net(data) #valImg --> data
    out = F.log_softmax(out, dim=1)

    pre_label = out.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_label]

    true_label = label.data.cpu().numpy()
    true_label = [i for i in true_label]

    eval_metrix = eval_semantic_segmentation(pre_label, true_label)
    train_acc = eval_metrix['mean_class_accuracy'] + train_acc
    train_miou = eval_metrix['miou'] + train_miou
    train_mpa = eval_metrix['pixel_accuracy'] + train_mpa
    if len(eval_metrix['class_accuracy']) < 12:
        eval_metrix['class_accuracy'] = 0
        train_class_acc = train_class_acc + eval_metrix['class_accuracy']
        error += 1
    else:
        train_class_acc = train_class_acc + eval_metrix['class_accuracy']

    print(eval_metrix['class_accuracy'], '================', i)


epoch_str = ('test_acc :{:.5f} ,test_miou:{:.5f}, test_mpa:{:.5f}, test_class_acc :{:}'.format(train_acc /(len(test_data)-error),
                                                            train_miou/(len(test_data)-error), train_mpa/(len(test_data)-error),
                                                            train_class_acc/(len(test_data)-error)))

if train_miou/(len(test_data)-error) > max(miou_list):
    miou_list.append(train_miou/(len(test_data)-error))
    print(epoch_str+'==========last')

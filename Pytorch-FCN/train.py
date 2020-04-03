from utiles.evalution_segmentaion import eval_semantic_segmentation
from torch import optim
from torch.autograd import Variable
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch as t
import data
from FCN import FCN8s,VGGNet #FCN网络

BATCH_SIZE = 2
train_data = DataLoader(data.Cam_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_data = DataLoader(data.Cam_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)



def main():

    
    vgg_model = VGGNet(requires_grad=True) 
    net = FCN8s(pretrained_net=vgg_model,n_class=12)
    net = net.cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    eval_miou_list = []
    best = [0]
    print('-----------------------train-----------------------')
    
    
    for epoch in range(30):
        if epoch % 10 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        #global net  #自认为加的
        net = net.train()
        prec_time = datetime.now()
        for i, sample in enumerate(train_data):
            imgdata = Variable(sample['img'].cuda())
            imglabel = Variable(sample['label'].long().cuda())

            optimizer.zero_grad()
            out = net(imgdata)
            out = F.log_softmax(out, dim=1)

            loss = criterion(out, imglabel)

            loss.backward()
            optimizer.step()
            train_loss = loss.item() + train_loss

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = imglabel.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc = eval_metrix['mean_class_accuracy'] + train_acc
            train_miou = eval_metrix['miou'] + train_miou
            train_class_acc = train_class_acc + eval_metrix['class_accuracy']

        net = net.eval()
        eval_loss = 0
        eval_acc = 0
        eval_miou = 0
        eval_class_acc = 0

        for j, sample in enumerate(val_data):
            valImg = Variable(sample['img'].cuda())
            valLabel = Variable(sample['label'].long().cuda())

            out = net(valImg)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, valLabel)
            eval_loss = loss.item() + eval_loss
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = valLabel.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrics = eval_semantic_segmentation(pre_label, true_label)
            eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
            eval_miou = eval_metrics['miou'] + eval_miou
            eval_class_acc = eval_metrix['class_accuracy'] + eval_class_acc

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)

        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, Train_class_acc:{:} \
        Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} ,Valid Class Acc:{:}'.format(
            epoch, train_loss / len(train_data), train_acc / len(train_data), train_miou / len(train_data), train_class_acc / len(train_data),
               eval_loss / len(train_data), eval_acc/len(val_data), eval_miou/len(val_data),eval_class_acc / len(val_data)))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(epoch_str + time_str)

        if (max(best) <=  eval_miou/len(val_data)):
            best.append(eval_miou/len(val_data))
            t.save(net.state_dict(),  'D:/机器学习/cvpaper/03语义分割-fcn论文原文及代码附件(1)/code/logs/last.pth') # 'xxx.pth' #保存模型


if __name__ == '__main__':
    main()


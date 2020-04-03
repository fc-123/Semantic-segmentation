### Pytorch复现FCN网络  
#### 1、环境配置  
Windows10，pytorch=1.3，python=3.6  
参考博客：https://github.com/wkentaro/pytorch-fcn  
#### 2、文件说明  
CamVid文件夹：数据集，里面包含训练集，验证集，测试集；  
logs文件夹：存放训练后的模型文件.pth;  
imgs文件夹：存放预测后的图像；  
data.py:数据处理；FCN.py:网络模型文件,包含FCN32s、FCN16s、FCN8s;  
#### 3、复现步骤：  
模型训练：  
    python train.py  
模型测试：  
    python test.py  
模型预测：  
    python predict.py  
#### 4、CamVid数据集下载：  
https://download.csdn.net/download/weixin_44753371/12299379
#### 5、训练自己的数据集  
可以详情参考博客：  
https://blog.csdn.net/weixin_44753371/article/details/105292287

# 张娇昱进度记录

## 18.0416-0516 月目标：
图像篡改的深度学习方向：
* 1 tensorflow课程完成
* 2 论文精读2-3篇
* 3 复现重采样特征+LSTM(Bunk的论文)。

### 18.0423-0502 第二周进度
* 1 tensorflow课程学至lesson09.挣扎style—transfer
* 2 bunk论文，深度学习模型的框架。
* 3 cs231N，补了一周关于CNN和RNN的课和LSTM

### 未解决问题
* 1 bunk论文。
  1.  各个步骤，LSTM模型中的输入输出数据大小
  2.  关于输入LSTM中的resampling feature map 到底怎么获得？文中两个叙述，一个用卷积层，一个用linear filter，暂时采用后者。
  3.  动手写代码啊。。。赶紧开始写啊张娇昱orz。切入点在哪里赶紧想。。。
* 2 tensorflow tfRecord 比较重要，可以用来制作自己的数据集。回头细看。
* 3 python下的opencv ,tensorflow eager. 暂时感觉不重要。搁置。

### 18.0416-0422 第一周进度  
* 1 tensorflow课程学至lesson05（笔记在tf文件夹）  
* 2 bunk论文初看（mendeley）。简单整理了机器学习手动设计特征的，关于LSTM步骤还是比较模糊。  
* 3 LSTM在tensorflow下的实现例子  

### 已解决问题
* 1 bunk论文。  
叙述模糊，需要再看的点：  
  1. 6重特征值的选取？第一步是手动设计重采样特征，第二步是训练6个binary classifier，选择有两个隐藏层的手工设计的神经网络。目前感觉两步不是承接，第一步对应机器学习方式，第二步对应LSTM方式。  
  答：两步都属于手工设计的特征。在论文中的第二个模型中的第一步，首先参考论文19，用了线性滤波器获得残差图像（residual image），在此基础上计算出p-map。   2. 手动特征最后是信号-波形图，需要阈值判断峰值以及排除JPG压缩导致的周期性峰值，但是这部分Mahdian的论文也没有细说。需要找一下代码或者Ostu阈值相关论文。  
  答：目前看来。深度模型暂时不用histograms。  


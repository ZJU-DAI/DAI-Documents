# 对图像进行语义性篡改的实现
## 网络细节

除了文本编码过程中未采用stackgan中的condition augmention，其他结构均采用了论文中的设计。训练过程中数据集采用的Oxford102。

## 原图

![img](./image/original.png)

## 预处理后

![img](./image/ori2.png)

the flower shown has yellow anther red pistil and bright red petals.

this flower has petals that are yellow, white and purple and has dark lines

the petals on this flower are white with a yellow center

this flower has a lot of small round pink petals.

this flower is orange in color, and has petals that are ruffled and rounded.

the flower has yellow petals and the center of it is brown.

this flower has petals that are blue and white.
                      
these white flowers have petals that start off white in color and end in a white towards the tips.

## 300次epcho后的结果

![img](./image/train_300.png)

## epcho 0 

![img](./image/train_00.png)

## epcho 50

![img](./image/train_50.png)

## epoche 100

![img](./image/train_100.png)

## epoche 150

![img](./image/train_150.png)

## epoche 200

![img](./image/train_200.png)

## epoche 250

![img](./image/train_250.png)


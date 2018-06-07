## Tensorflow

用pip安装
```bash
$ pip3 install tensorflow
```
这里自动默认安装最新版疼送人flow1.7，之前在用自己的笔记本安装时，import tensorflow 导入时出现illegal instruction，后经过查找资料，笔记本cpu为A6-3420m，不支持该版本用pip安装。只能降级安装tensorflow1.5以下的版本。
```bash
$ pip3 install tensorflow==1.5
```
使用下面命令可以查找安装位置
```bash
$ pip3 show tensorflow
```
## Tensorboard
> * 用来显示流图

add this line to use TensorBoard
```python
writer=tf.summary.FileWriter("./graphs",sess.graph)
```
close the writer when you're  done using it
```python
writer.close()
```
使用tensorboard命令跑起来
```bash
$ tensorboard --logdir="./graphs" --port 6006
```
在linux下使用tensorboard命令无效
```bash
tensorboard : command not found
```
### linux环境下通过下面指令可以达到同样的效果
首先找到tensorboard文件夹
```bash
$ pip3 show tensorboard
```
进入该文件夹
```bash
$ cd ~/.../tensorboard
```
在运行该文件夹下的main.py
```bash
$ python3 main.py --logdir="./garphs" 
```
成功后就可以打开浏览器进入：http://localhost:6006 在graph下查看流图

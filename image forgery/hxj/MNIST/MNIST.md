# MNIST入门
MNIST数据集导入方式都在[input_data.py](https://github.com/STUDENT-ONE/Tensorflow/blob/master/MNIST/input_data.py),写入下面代码即可导入数据：
```python
import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
```
MNIST数据集的标签是介于0到9的数字，使用的标签数据是"one-hot vectors"，一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。在MNIST训练数据集中，mnist.train.images是一个形状为[55000,784]的张量，mnist.train.labels是一个[55000,10]的数字矩阵。
## tf.argmax
    返回一个张量某个纬度中的最大值的索引。
```python
correct_predictino=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
```
这里返回一个布尔数组，为了计算我们分类的准确率，我们将布尔值转换为浮点数来表示然后取平均值。
```python
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
```

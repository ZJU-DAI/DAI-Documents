# Tensorflow个人学习笔记

## Lecture 1

每个TF程序包括两个过程

1. 组建图
2. 采用session来执行图中的operation

### Graphs

Graphs 就是图，单纯的用来描述整个计算过程。

### Sessions

Graph即图，Session是用来执行图中的Operation（操作）

```Python
tf.Session.run(fetches,
               feed_dict=None,
               options=None,
               run_metadata=None)
```



### Tensor

Tensor是一个n维的array(数组)，也就是数据

0维 数字

1维 向量

2维 矩阵

### 例子

```python
g = tf.Graph() #建立图
with g.as_default(): #将g设为默认图
	x = tf.add(3, 5) #图中添加operation
sess = tf.Session(graph = g) #为图建立session
with tf.Session() as sess: #在session中执行图中的operation
	sess.run(x)
```

## Lecture 2

### Constants(常量)

```python
tf.constants(
	value, 
	dtype=None,
	shape=None,
	name='const',
	verify_shape=False
)
#赋予指定值的tensor方法
tf.zeros(shape, dtype=tf.float32, name=None) #初始化一个为0的tensor
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True) #将输入的tensor初始化为0
tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
#和上面两个类似不过是1

#将常量作为序列
tf.lin_space(start, stop, num, name=None) 
tf.lin_space(10.0, 13.0, 4) ==> [10. 11. 12. 13.]
tf.range(start, limit=None, delta=1, dtype=None, name='range')
tf.range(3, 18, 3) ==> [3 6 9 12 15]
tf.range(5) ==> [0 1 2 3 4]
```

### Variables(变量)

```python
#用tf.get_variable方法创建变量
s = tf.get_variable("scalar", initializer=tf.constant(2)) 
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())
```

tf.constant是operation（操作），而tf.Variable是一种有许多操作的类

变量在session中执行之前都需要先进行初始化。

### Placeholder

形如Y=wX+b的函数式，Y和X就是Placeholder。

因为有些数据一开始并没有，但是后期计算时可能会用到，因此需要placeholder。

```python
tf.placeholder(dtype, shape=None, name=None)
# 建立类型为float32，包含三个元素的向量的placeholder
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b
with tf.Session() as sess:
	print(sess.run(c, feed_dict={a: [1, 2, 3]})) 	# the tensor a is the key, not the string ‘a’
# >> [6, 7, 8]
```

其中Placeholder可以用通过dictionary来为其幅值。

feed_dict同样也可以feed TF中的operation。

例如

```python
# create operations, tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
	# compute the value of b given a is 15
	sess.run(b, feed_dict={a: 15}) 				# >> 45
```

### LazyLoading

lazyloading 问题是在session中graph建立了多个重复节点导致内存被过度占用。

例如

```python
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(z)
```

lazyloading 的做法会为了省力把z = tf.add(x,y)直接写到session中，这样会导致graph中建立了上述重复的十个加法节点，从而导致了内存溢出。

## Lecture 3

### Model 1 Linear Regression 

Code:examples/03_linreg_starter.py

#### Phase 1

- Step 1:读取数据

- Step 2:为输入和标签建立Placeholder

- Step 3:建立权重和偏差变量

- Step4:Inference

- Step5:确定loss的表达式

- Step6:建立Optimizer
#### Phase 2

- Step 1:Intialize variables
- Step 2:run optimizer

#### 用tf.data进行改进

采用tf.data可以直接对数据进行inference，而不需要利用placeholder并为其feed数据来进行inference。

- store data in tf.data.Dataset

- create an iterator through samples in Dataset(make one shot/initializable)

#### 并不用时刻采用tf.data

- 对于原形设计来说，通过feed dict更加简洁好写。

- 当拥有多个数据源或者复杂的预处理时候，tf.data就变得难以使用

- NLP 数据通常都是整形序列。在这种情况下，tf.data的效果并不明显

#### Optimizers

Optimizer是一种operation用于优化loss，当tensorflow执行Optimizer时，他会执行该操作所依赖的图。

### Model 2 Logistic Regression

Minst:

X:image of handwritten digit

Y:the digtial value

Model:

Inference:Y_predicted=softmax(X*w+b)	   

Cross entropy loss = -log(Y_predicted)

## Lecture 5

### 如何构建Tensorflow模型

#### Phase 1: 组建图

1. 导入数据（用placeholder或者tf.data）
2. 定义权重
3. 定义推倒模型
4. 定义损失函数
5. 定义优化器

#### Phase 2: 执行计算

1. 为所有模型变量进行初始化
2. 将迭代器进行初始化/放入训练数据
3. 对训练数据执行推断模型，通过计算每个输入得到当前模型参数的输出。
4. 计算cost
5. 调整模型参数以最小化/最大化cost

### Name scope和Variable scope

利用tf Name scope 将一些节点操作整理在一起，使得代码可读性更高，在tensorboard上可以更加方便的阅读。

而variable scope使得变量可以进行分享。具体看下面这个例子

```python

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')

def two_hidden_layers(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.Variable(tf.random_normal([100, 50]), name="h1_weights")
    b1 = tf.Variable(tf.zeros([50]), name="h1_biases")
    h1 = tf.matmul(x, w1) + b1
    assert h1.shape.as_list() == [200, 50]  
    w2 = tf.Variable(tf.random_normal([50, 10]), name="h2_weights")
    b2 = tf.Variable(tf.zeros([10]), name="h2_biases")
    logits = tf.matmul(h1, w2) + b2
    return logits

logits1 = two_hidden_layers(x1)
logits2 = two_hidden_layers(x2)
```

以上代码在tensorboard上看起来就是两个重复的图，tensorflow会创建两个不同的变量集合，但实际上，我们希望为这些输入创建的是相同的变量。所以就需要采用tf.get_variable方法。但是采用该方法的时候，它会先检测该变量是否已经存在，如果存在就复用，如果不存在就建立一个新的。但是仅把tf.Variable 替换成tf.get_variable会出现错误，为避免这个错误，就需要用到varscope。

```python
with tf.variable_scope('two_layers') as scope:
    logits1 = two_hidden_layers_2(x1)
    scope.reuse_variables()
    logits2 = two_hidden_layers_2(x2)
```

这样在tensorboard中就只会看到只有一组变量了。

### Graph collections

collection是一个全局的储存机智，他不会受到变量名空间的影响，一旦经过保存，就可以从collection中取得。

### Manage experiments

#### tf.train.Saver()

tf.train.Saver()是用于存储模型中的一些参数，这些参数往往经过了部分的过程和epochs。因此我们可以用与重新存储或者训练我们的模型。

#### tf.summary

通过summary操作，可以使得我们的loss，accuracy等在tensorboard中可视化。

## Lecture 6

卷积神经网络结构通常由Convolutional Layer卷积层，Pooling Layer池化层和Fully-ConnectedLayer全连接层组成。

### 卷积层

**概述** 首先每个filter在空间上 宽度和高度都非常小，但是深度和输入一致。在前向传播时，filter在每个输入数据上进行滑动，然后计算filter和输入数据的结果。其实就类似图像信号处理中的提取特征。

**参数** 在卷积的过程中有三个重要的参数，depth，stide，zero-padding。
1. 首先输出的depth和filter数量通常一致，每个filter都在对输入数据提取不同的特征。
2. 其次，滑动filter的过程中，需要设定stride，stride是每次filter滑动的距离。
3. zero-pading就是对输入进行补0，可以控制输出的空间尺寸。

**公式** 输出的数据体在空间上的尺寸可以通过输入的尺寸W，filter尺寸F，stride尺寸S和zeropadding尺寸P的函数来进行计算。这里假设所有的结构都是正方形。那么输出的尺寸就是(W-F+2P)/S+1。

![pic1](pic/pic1.jpg)

如图所示，fitler尺寸为3，输入是5，zeropadding为1。左侧：使用stride为1，所以输出是5。右侧：使用stride为2，则输出为3。在这里stride不能采用3，因为fitler无法整体滑动穿过整个输入。

**zeropadding** 在上述的左侧例子中，由于使用了zeropadding，使得输入输出的维度都是5。当不使用zeropadding时，输出维度就只有3。通常情况下，stride为1时候，zeropadding的值通常是(F-1)/2。

**stride** stride，filter，zeropadding的参数通常情况都是相互限制。例如当W为10，filter为3，这时候就不能采用stride=2，这样会导致fitler无法整体划过输入数据。

**小结** 
- 输入尺寸为W1·H1·D1
- fitler数量K，fitler尺寸F，stride S，zeropadding P
- 输出尺寸W2·H2·D2
- W2和H2都等于(W1-F+2P)/S+1
- D2=K
- 由于参数共享，每个filter中包含F·F·D1个权重，那么卷积层就一共有F·F·D1·K个权重和K个偏置。

### 池化层

在连续的卷积层之间会周期性的插入池化层。它的作用主要是降维，从而减少网络中的参数数量，节约了计算资源消耗，同时还能控制过拟合。池化层采用MAX操作，对输入的每一个深度切片进行独立操作，改变其空间尺寸。最常见的形式是采用2·2的fitler，以stride为2 对每个切片进行降为。每个max操作是取filter中的最大值，深度保持不变。

- 输入尺寸为W1·H1·D1
- fitler空间F
- strideS
- 输出尺寸W2·H2·D2。其中W2和H2都等于(W1-F)/S+1，D2=D1。
- 池化层中很少采用zeropadding

### 全连接层

全连接层中，神经元对于前一层中的所有激活数据都是全部链接的，他们的激活可以先采用矩阵乘法，再加上偏差。关于FC和CONV之间的转化，后期补充

### 总结

总的来说卷积神经网络通常是由三种layer构成，分别是conv，pooling，fc。RELU激活函数通常是紧跟conv为了去线性化。

在卷积神经网络中，最常见的排列形式就是把conv和relu放在一起，后面紧跟pooling层。然后重复如此将图像在空间上缩小的一个足够小的尺寸，然后通过最后的FC得到输出。大致结构就是这样。
input->[[conv->relu]*n->pool]*m->[fc->relu]*k->fc

## Lecture 7

### 直接卷积

L7 主要讲述的是如何在Tensorflow中实现卷积神经网络。

在Tensorflow中预置了多种卷积层模板，通常情况下我们都采用2d卷积。

```python
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```

Input:Batch size(N)·H·W·C
Filter:H·W·Input Channel·Output Channel
Strides：4元的1维tensor，代表四个方向的strides
Padding：'SAME' OR 'VALID'（same就是补全，valid就是舍弃）
Dialations: The dilation factor. If set to k > 1, there will be k-1 skipped cells between each filter element on that dimension.  
Data_format: default to NHWC

对于stride，在第一个和第四个维度尽量不要使用除了1以外的数字。因为我们不希望跳过一个批次中的人和样本或者图像中的任何一个通道。这和扩张一样。

### 在MNIST中运用CNN

在之前的logistic regression作业中，已经采用了全连接层。这次试试看用卷积神经网络的效果如何

对于MNIST数据集，我们将采用两个卷积层，每个卷积层后面都会紧跟一个relu激活函数和一个池化层。最后街上两个全连接层。stride采用[1,1,1,1].

![pic](pic/pic2.png)

由于将重复多次的操作，因此要注意代码的复用性。使用variable scope非常重要，这使得我们可以在不同的图层之间使用同名变量。例如一个变量名为weights的变量在变量域con1中就是conv1/weights。

#### 卷积层

我们将采用tf.nn.conv2d作为卷积层。一个比较常用的方法就是把卷积层和激活函数归为同一组，创建conv_relu以供两层共同使用。

```python
def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters],
                                initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [filters],
                                initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(conv + biases, name=scope.name)
```
#### 池化层

池化层是一种降采样手段，用于降低从卷积层中提取的特征图的维度，以此来降低处理时间。池化层用最具有代表性的特征来替代部分区域的数据。目前最流行的池化算法就是最大池化，用部分区域内的最大值来替换该部分区域的数据。其他的一些算法是平均池化，将该区域内的数值进行平均。

在该实验中采用tf.nn.max_pool用于池化层，我们将创建一个maxpool方法用于所有的池化层。

```python
def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs, 
                            ksize=[1, ksize, ksize, 1], 
                            strides=[1, stride, stride, 1],
                            padding=padding)
    return pool
```

#### 全连接层

这个就不多说了，前面几讲其实都在用FC来实现模型。

```python
def fully_connected(inputs, out_dim, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out
```

#### 组合

将这些代码进行复用组合，就可以形成我们的模型

```python
def inference(self):
        conv1 = conv_relu(inputs=self.img,
                        filters=32,
                        k_size=5,
                        stride=1,
                        padding='SAME',
                        scope_name='conv1')
        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
        conv2 = conv_relu(inputs=pool1,
                        filters=64,
                        k_size=5,
                        stride=1,
                        padding='SAME',
                        scope_name='conv2')
        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = tf.nn.relu(fully_connected(pool2, 1024, 'fc'))
        dropout = tf.layers.dropout(fc, self.keep_prob, training=self.training, name='dropout')
        
        self.logits = fully_connected(dropout, self.n_classes, 'logits')
```

在训练过程中，我们可能会选择性评估当前epoch下的模型的准确度。在Tensorboard上，我们可以track到loss和accuracy

```python
def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
```

#### tf.layers

除了手动自己编写卷积层，连接层代码之外。TensorFlow同样提供了一个module叫做tf.layers，它可以提供许多预制的层。例如Kerase，Sonnet。

因此，当我们想建立一个包括relu激活函数的卷积层时候，可以这样写。

```python
conv1 = tf.layers.conv2d(inputs=self.img,
                                  filters=32,
                                  kernel_size=[5, 5],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  name='conv1'
```

对于全连接层和最大池化层可以这样

```python
pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                        pool_size=[2, 2], 
                                        strides=2,
                                        name='pool1')

fc = tf.layers.dense(pool2, 1024, activation=tf.nn.relu, name='fc')
```

使用tf.layers非常直观，但是有一点点需要注意，当使用tf.layers.dropout的时候，需要一个其他的变量来指明其在训练模式还是评估模式。在训练过程中，我们希望丢弃一些神经元，但是我们希望在评估的时候能够充分利用所有的神经元。

```python
dropout = tf.layers.dropout(fc,
                                    self.keep_prob,
                                    training=self.training,
                                    name='dropout')
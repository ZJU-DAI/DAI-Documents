# Tensorflow个人学习笔记

## Lecture1

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

###Constants(常量)

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
####Phase 2

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


# More constants
    tf.constant(value,dtype=None,shape=None,name='Const',verify_shape=False)
> * tf.zeros(shape,dtype=tf.float32,name=None)
eg:    tf.zeros([2,3],tf.int32)==>[[0,0,0],[0,0,0]]
> * tf.zeros_like(input_tensor,dtype=None,name=None,optimize=True)
eg:    tf.zeros_like(input_tensor)==>[[0,0],[0,0],[0,0]]
> * tf.fill(dims,value,name=None)
eg:    tf.fill([2,3],8)==>[[8,8,8],[8,8,8]]
> * tf.linspace(start,stop,num,name=None) 和np.linspace 有点不同
eg:    tf.linspace(10.0,13.0,4)==>[10.0 11.0 12.0 13.0]
> * tf.range(start,limit=None,delta=1,dtype=None,name='range')
eg:    tf.range(3,18,3)==>[3,6,9,12,15]

# Some operations
```
a=tf.constant([3,6])
b=tf.constant([2,2])
```
> * tf.add(a,b) # >>[5 8]
> * tf.add_n([a,b,b]) # >> [7,10]. Equivalent to a + b + b
> * tf.mul(a,b) # >> [6 12] 
> * tf.matmul(a,b) # >> ValueError
> * tf.matmul(tf.reshape(a,[1,2]),tf.reshape(b,[2,1])) # >> [[18]]
> * tf.div(a,b) # >> [1 3]
> * tf.mod(a,b) # >> [1 0]

# Varoables
> * create variable a with scalar value
a=tf.Variable(2,name="scalar")
> * create variable b as a vector
b=tf.Variable(2,3],name="vector")
> * create variable c as a 2x2 matrix
c=tf.Variable([[0,1],[2,3]],name="matrix")
> * create variable w as 784x10 tensor,filled with zeros
w=tf.Variable(tf.zeros([784,10]))

## 初始化variables
初始化所有变量
```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
      sess.run(init)
```
初始化变量子集
```python
init_ab=tf.variables_initializer([a,b],name="init_ab")
with tf.Session() as sess:
      sess.run(init_ab)
```
初始化单个变量
```python
w=tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
      sess.run(w.initializer)
```
### tf.Variable.assign()
```python
w=tf.Variable(10)
w.assign(100)
with tf.Session() as sess:
     sess.run(w.initializer)
     print(w.eval())
```
结果输出为10，w.assign(100)并没有分配100给w，它产生了一个分配操作，这个操作需要跑起来才能生效。
```python
w=tf.Variable(10)
assign_op=w.assign(100)
with tf.Session() as sess:
     #sess.run(w.initializer)，不需要初始化，assign-op已经完成了这个。
     sess.run(assign_op)
print(w.eval())
```
### assign_add() and assign_sub()
```python
my_var=tf.Variable(10)
with tf.Session() as sess:
     sess.run(my_var.initializer)
     #increment by 10
     sess.run(my_var.assign_add(10)) # >> 20
     #decrement by 2
     sess.run(my_var.assign_sub(2)) # >> 18
```
    这两个函数不会初始化变量my_var，这些操作需要原值。
如果需要初始化分别计算需要两个session
```python
w=tf.Variable(10)
sess1=tf.Session()
sess2=tf.Session()

sess1.run(w.initializer)
sess2.run(w.initializer)

print(sess1.run(w.assign_add(10))) # >> 20
print(sess2.run(w.assign_sub(2)))  # >> 8

sess1.close()
sess2.close()
```
### Session vs InteractiveSession
    区别：InteractiveSession可以在没有指定会话对象的情况下运行变量。Session使用with...as...后不需要使用close关闭对话，而调用InteractiveSession需要在最后调用close。使用tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session然后再定义操作，而使用tf.Session()来构建会话需要在会话构建之前定义好全部操作然后在构建会话。
```python
sess=tf.InteractiveSession()
a=tf.constant(5.0)
b=tf.constant(6.0)
c=a*b
print(c.eval())
sess.close()
```
### Placeholders
    tf.placeholder(dtype,shape=None,name=None)
    用于定义过程，在执行的时候再赋具体的值。
```python
# create a placeholder of type float 32-bit,shape is a vector of 3 elements
a=tf.placeholder(tf.float32,shape=[3])

# create a constant of type float 32-bit, shape is a vector of 3 elements
b=tf.constant([5,5,5],tf.float32)

# use the placeholder as you would a constant or a variable
c=a+b # Short for tf.add(a,b)
with tf.Session() as sess:
      # print(sess.run(c)) ==> Error because a doesn't have any value
      # feed [1,2,3] to placeholder a via the dict {a:[1,2,3]}
      # fetch value of c
      print sess.run(c,{a:[1,2,3]}) # the tensor a is the key, not the string 'a'
```
Feeding values to tf ops
```python
a=tf.add(2,5)
b=tf.multiply(a,3)
with tf.Session() as sess；
     replace_dict={a:15}
     sess.run(b,feed_dict=replace_dict) #return 45 not 21
```
### lazy loading
```python
x=tf.Variable(10,name='x')
y=tf.Variable(20,name='y')
z=tf.add(x,y)

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     writer=tf.summary.FileWriter('./graphs',sess.graph)
     for _ in range(10):
           sess.run(z)
     writer.close()
```

### Optimizer
```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

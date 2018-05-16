import tensorflow as tf

# 1a: Create two random 0-d tensors x and y of any distribution
#     Create a Tensorflow object that returns x+y if x>y, and x-y otherwise.

x=tf.random_normal([])
y=tf.random_normal([])   # 0-d tensor is a constant
#tf.cond() is like if...else...
out=tf.cond(tf.greater(x,y),lambda:tf.add(x,y),lambda:tf.subtract(x,y))


# 1b: Create two 0-d tensors x and y randomly selected from the range [-1,1).
#     return x+y if x<y,x-y if x>y, 0 otherwise.

x=tf.random_uniform([],-1,1,dtype=tf.float32)
y=tf.random_uniform([],-1,1,dtype=tf.float32)
#tf.case() is like switch 
out=tf.case({tf.less(x,y):lambda:tf.add(x,y),tf.greater(x,y):lambda:tf.subtract(x,y)},
            default=lambda:tf.constant(0.0),exclusive=True)


# 1c: Create the tensor x of the value [[0,-2,-1],[0,1,2]]
#     and y as a tensor of zeros with the same shape as x.
#     return a boolen tensor that yields Trues if x equals y element-wise.

x=tf.constant([[0,-2,-1][0,1,2]])
y=tf.zeros_like(x)
out=tf.equal(x,y)

# 1d: Create the tensor x of value
#     get the indices of elements in x whose values are greater than 30.
#     then extract elements whose values are greater than 30.

x=tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
               30.97266006,  26.67541885,  38.08450317,  20.74983215,
               34.94445419,  34.45999146,  29.06485367,  36.01657104,
               27.88236427,  20.56035233,  30.20379066,  29.51215172,
               33.71149445,  28.59134293,  36.05556488,  28.66994858])
indices=tf.where(x>30)
out=tf.gater(x,indices)

# 1e: Create a random 2-d tensor of size 6x6 with the diagonal values of 1, 2,...,6

values = tf.range(1,7)
out = tf.diag(values)


# 1f: Create a random 2-d tensor of size 10x10 from any distribution
#     calculate its determinant

m=tf.random_normal([10,10],mean=10,stddev=1)# shape , mean  , standard deviation of the normal distribution 
out=tf.matrix_determinant(m) #  calculate its determinant

# 1g: Create tensor x with value [5,2,3,5,10,6,2,3,4,2,1,1,0,9]
#     return the unique elements in x 

x=tf.constant([5,2,3,5,10,6,2,3,4,2,1,1,0,9])
unique_values,indices=tf.unique(x) # unique_values ,index

# 1h: Create two tensors x and y of shape 300 from any normal distribution,
#     as long as they are from the same distribution.
#     use tf.cond() to return:
#     -the mean squared error of (x-y) if the average of all elements in (x-y) is negative,
#     or 
#     -the sum of absolute value of all elements in the tensor (x-y) otherwise.

x=tf.random_normal([300],mean=5,stddev=1)
y=tf.random_normal([300],mean=5,stddev=1)
average=tf.reduce_mean(x-y)
def f1(): return tf.reduce_mean(tf.square(x-y))
def f2(): return tf.reduce_sum(tf.abs(x-y))
out=tf.cond(average<0,f1,f2)

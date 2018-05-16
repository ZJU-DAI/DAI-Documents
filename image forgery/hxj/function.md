## tf.random
------
### tf.random_normal
    从正态分布输出随机值。
```python
random_normal(shape,mean=0.0,stddev1.0,dtype=tf.float32,seed=None,name=None)
```
### tf.random_uniform
    从均匀分布返回随机值。
```python
random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)
```
## tf.case()
类似c中的switch，pred：lambda:f1(),pred:lambda:f2(),default=lambda:f3()


## tf.cond()
类似c中的if else pred:lambda:f1(),lambda:f2()


## tf.cast()
转化数据格式的类型 (x,dtype,name)


## np.newaxis
在行或列上增加一个维度，
```
eg：(6,)  ==> [1 2 3 4 5 6]
    (1,6) ==> [[1 2 3 4 5 6]]
    (6,1) ==> [ [1]
                [2]
                [3]
                [4]
                [5]
                [6]]
```

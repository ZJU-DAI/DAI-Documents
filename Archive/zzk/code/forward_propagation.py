import tensorflow as tf
from numpy.random import RandomState
batch_size=8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#[0.7,0.9]代表一维向量，[[0.7,0.9]]代表1*3矩阵
#x=tf.constant([[0.7,0.9]])
x=tf.placeholder(tf.float32,shape=[None,2],name="x-input")
y_=tf.placeholder(tf.float32,shape=[None,1],name="y-input")

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

y=tf.sigmoid(y)
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


rdm=RandomState(1)
dataset_size=128

X=rdm.rand(dataset_size,2)
#[[int(x1+x2 < 1) for(x1,x2) in X]]的shape是（1,dataset_size）
#[[int(x1+x2 < 1)] for(x1,x2) in X]的shape是（dataset_size,1）
Y=[[int(x1+x2 < 1)] for(x1,x2) in X]

writer=tf.summary.FileWriter("E:\Study\代码\graphs",tf.get_default_graph())

sess=tf.Session()

init_op=tf.global_variables_initializer()
sess.run(init_op)
#sess.run(w1.initializer)
#sess.run(w2.initializer)

print(sess.run(w1))
print(sess.run(w2))

STEPS=5000
for i in range(STEPS):
	start=(i*batch_size) % dataset_size
	end=min(start+batch_size,dataset_size)

	sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

	if i % 1000 == 0:
		total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
		print("After %d trainint step,cross entropy on all data is %g" % (i,total_cross_entropy))

print(sess.run(w1))
print(sess.run(w2))
#print(sess.run(y))

sess.close()
writer.close()

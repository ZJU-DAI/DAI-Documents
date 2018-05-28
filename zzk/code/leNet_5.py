import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

#INPUT_NODE_NUM = 28
OUTPUT_NODE_NUM = 10

IMAGE_SIZE = 28
NUM_CHANNEL = 1
NUM_LABEL = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

BATCH_SIZE = 128
def inference(input_tensor,train,regularizer):
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable('weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNEL,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.1))
	
		conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
	
	with tf.variable_scope('layer2-pool1'):
		pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	
	with tf.variable_scope('layer3-conv2'):
		conv2_weights = tf.get_variable('weight',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable('bias',[CONV2_DEEP],initializer=tf.constant_initializer(0.1))
		
		conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
		
	with tf.variable_scope('layer4-pool2'):
		pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	
	print(pool2)
	pool_shape = pool2.get_shape().as_list()
	nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
	
	print(pool2)
	#reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
	reshaped = tf.reshape(pool2,[-1,nodes])
	#reshaped = tf.reshape(pool2,tf.convert_to_tensor([None,nodes]))
	
	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable('weight',[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc1_biases = tf.get_variable('bias',[FC_SIZE],initializer=tf.constant_initializer(0.1))

		fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
	
	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable('weight',[FC_SIZE,NUM_LABEL],initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc2_biases = tf.get_variable('bias',[NUM_LABEL],initializer=tf.constant_initializer(0.1))
		
		logit = tf.matmul(fc1,fc2_weights)+fc2_biases
	
	return logit

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
train_reshaped = np.reshape(mnist.train.images,[len(mnist.train.images),IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
validation_reshaped = np.reshape(mnist.validation.images,[len(mnist.validation.images),IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])

x = tf.placeholder(tf.float32,shape=[None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL],name='x-input')
y_ = tf.placeholder(tf.float32,shape=[None,NUM_LABEL],name='y-input')

y = inference(x,False,False)

cross_entropys = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
loss = tf.reduce_mean(cross_entropys)

train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	start_time = time.time()
	for i in range(1000):
		start = (i*BATCH_SIZE)%len(mnist.train.images)
		end = min(start+BATCH_SIZE,len(mnist.train.images))
		#x_batch,y_batch = mnist.train.next_batch()
		loss_value,_ = sess.run([loss,train_step],feed_dict={x:train_reshaped[start:end],y_:mnist.train.labels[start:end]})
		print('loss_value-%d %g'% (i,loss_value))
	
	print('Total time: {0} seconds'.format(time.time() - start_time))
	print('accuracy:',sess.run(accuracy,feed_dict={x:validation_reshaped,y_:mnist.validation.labels}))

import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import time

FM_CF_SIZE = 1
FM_SF_SIZE = 1
FM_E1F_SIZE = 1
FM_E2F_SIZE = 3

IMAGE_SIZE = 224
NUM_CHANNEL = 3
NUM_LABEL = 5

CONV1_DEEP = 96
CONV1_SIZE = 7
CONV2_DEEP = 5
CONV2_SIZE = 1

BATCH_SIZE = 8
NUM_TRAIN_EXAMPLE = 10000

INPUT_DATA_PATH = r'E:\Study\flower_photos.npy'

def fire_module(input_data,deepths,scope=None):
	with tf.variable_scope(scope):
		num_channel_in = input_data.get_shape().as_list()
		#print(num_channel_in)
		
		s_weights = tf.get_variable('s_weight',shape=[FM_SF_SIZE,FM_SF_SIZE,num_channel_in[3],deepths[0]],initializer=tf.truncated_normal_initializer(stddev=0.1))
		s_biases = tf.get_variable('s_bias',shape=[deepths[0]],initializer=tf.constant_initializer(0.1))
		
		s_conv = tf.nn.conv2d(input_data,s_weights,strides=[1,1,1,1],padding='VALID')
		s_relu = tf.nn.relu(tf.nn.bias_add(s_conv,s_biases))
		
		e1_weights = tf.get_variable('e1_weight',shape=[FM_E1F_SIZE,FM_E1F_SIZE,deepths[0],deepths[1]],initializer=tf.truncated_normal_initializer(stddev=0.1))
		e1_biases = tf.get_variable('e1_bias',shape=[deepths[1]],initializer=tf.constant_initializer(0.1))
		
		e1_conv = tf.nn.conv2d(s_relu,e1_weights,strides=[1,1,1,1],padding='VALID')
		e1_relu = tf.nn.relu(tf.nn.bias_add(e1_conv,e1_biases))
		
		e2_weights = tf.get_variable('e2_weight',shape=[FM_E2F_SIZE,FM_E2F_SIZE,deepths[0],deepths[2]],initializer=tf.truncated_normal_initializer(stddev=0.1))
		e2_biases = tf.get_variable('e2_bias',shape=[deepths[2]],initializer=tf.constant_initializer(0.1))
		
		e2_conv = tf.nn.conv2d(s_relu,e2_weights,strides=[1,1,1,1],padding='SAME')
		e2_relu = tf.nn.relu(tf.nn.bias_add(e2_conv,e2_biases))
		
		output = tf.concat([e1_relu,e2_relu],3)
		print('fire_output',output)
		
	return output

def squeezeNet(input_data):
	with tf.variable_scope('squeezeNet'):
		with tf.variable_scope('layer1-conv1'):
			weights = tf.get_variable('weight',shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNEL,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
			biases = tf.get_variable('bias',shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.1))
		
			##SAME:112 VALID:109 都不是论文中提到的111
			conv1 = tf.nn.conv2d(input_data,weights,strides=[1,2,2,1],padding='VALID')
			relu1 = tf.nn.relu(tf.nn.bias_add(conv1,biases))
		
		with tf.variable_scope('layer2-maxpool1'):
			maxpool1 = tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
			
		with tf.variable_scope('layer3-fire2'):
			fire2 = fire_module(maxpool1,[16,64,64],scope='fire2')
		
		with tf.variable_scope('layer4-fire3'):
			fire3 = fire_module(fire2,[16,64,64],scope='fire3')
		
		with tf.variable_scope('layer5-fire4'):
			fire4 = fire_module(fire3,[32,128,128],scope='fire4')
		
		with tf.variable_scope('layer6-maxpool2'):
			maxpool2 = tf.nn.max_pool(fire4,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
		
		with tf.variable_scope('layer7-fire5'):
			fire5 = fire_module(maxpool2,[32,128,128],scope='fire5')
		
		with tf.variable_scope('layer8-fire6'):
			fire6 = fire_module(fire5,[48,192,192],scope='fire6')
		
		with tf.variable_scope('layer9-fire7'):
			fire7 = fire_module(fire6,[48,192,192],scope='fire7')

		with tf.variable_scope('layer10-fire8'):
			fire8 = fire_module(fire7,[64,256,256],scope='fire8')		

		with tf.variable_scope('layer11-maxpool3'):
			maxpool3 = tf.nn.max_pool(fire8,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

		with tf.variable_scope('layer12-fire9'):
			fire9 = fire_module(maxpool3,[64,256,256],scope='fire9')			
		
		conv2_in_shape = fire9.get_shape().as_list()
		with tf.variable_scope('layer13-conv2'):
			weights = tf.get_variable('weight',[CONV2_SIZE,CONV2_SIZE,conv2_in_shape[3],CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
			biases = tf.get_variable('bias',[CONV2_DEEP],initializer=tf.constant_initializer(0.1))
			
			conv2 = tf.nn.conv2d(fire9,weights,strides=[1,1,1,1],padding='SAME')
			relu2 = tf.nn.relu(tf.nn.bias_add(conv2,biases))
		
		with tf.variable_scope('layer14-avgpool1'):
			avgpool1 = tf.nn.avg_pool(relu2,ksize=[1,13,13,1],strides=[1,1,1,1],padding='VALID')
			
		reshape = avgpool1.get_shape().as_list()
		nodes = reshape[1]*reshape[2]*reshape[3]
		fc = tf.reshape(avgpool1,[-1,nodes])

	return fc

datas = np.load(INPUT_DATA_PATH)
training_images = datas[0]
training_labels = datas[1]
testing_images = datas[4]
testing_label = datas[5]
n_training_example = len(training_images)

x = tf.placeholder(tf.float32,shape=[None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL],name='x-input')
y_ = tf.placeholder(tf.float32,shape=[None,NUM_LABEL],name='y-input')

y = squeezeNet(x)

print(y)

#tf.summary.FileWriter('E:\Study\代码\graphs',tf.get_default_graph())

cross_entropys = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
loss = tf.reduce_mean(cross_entropys)

train_step = tf.train.RMSPropOptimizer(0.0001).minimize(loss)

#print(y_)
#print(y)
correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#rds = RandomState(1)
#X = rds.rand(10000,224,224,3)
#Y = rds.rand(10000,1000)

with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	start_time = time.time()
	for i in range(10000):
		start = (i*BATCH_SIZE)%n_training_example
		end = min(start+BATCH_SIZE,n_training_example)
		#x_batch,y_batch = mnist.train.next_batch()
		
		train_label_one_hot_value = sess.run(tf.one_hot(training_labels[start:end],NUM_LABEL))
		loss_value,_ = sess.run([loss,train_step],feed_dict={x:training_images[start:end],y_:train_label_one_hot_value})
		print('loss_value-%d %g'% (i,loss_value))
	
	test_label_one_hot_value = sess.run(tf.one_hot(testing_label[1:150],NUM_LABEL))
	print('Total time: {0} seconds'.format(time.time() - start_time))
	print('accuracy:',sess.run(accuracy,feed_dict={x:testing_images[1:150],y_:test_label_one_hot_value}))


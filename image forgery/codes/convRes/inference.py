# -*- coding: utf-8 -*-
import tensorflow as tf

#配置神经网络参数

NUM_CHANNELS = 1
NUM_LABELS = 2

def inference(input, train, regularizer):
	input_gray = tf.image.rgb_to_grayscale(input)
	#convRes
	with tf.variable_scope('conRes'):
		w_res = tf.get_variable(name='w_res', shape=[5, 5, 1, 12],
		                        initializer=tf.truncated_normal_initializer(stddev=0.1))
		w_res = tf.transpose(w_res, [3, 0, 1, 2])
		w_res = tf.reshape(w_res, [12, 5, 5])
		weights = tf.Variable(tf.zeros([12, 5, 5]))
		for i in range(w_res.shape[0]):
			tf.assign(weights[i], w_res[i])
			weights[i, 2, 2].assign(0)
			weights[i].assign(weights[i] / tf.reduce_sum(weights[i]))
			weights[i, 2, 2].assign(-1)
		weights = tf.expand_dims(weights, -1)
		print(weights.get_shape())
		weights = tf.transpose(weights, [1, 2, 3, 0])
		
		b_res = tf.get_variable(name='b_res', shape=[12], initializer=tf.constant_initializer(0.1))
		
		conv_res = tf.nn.conv2d(input=input_gray, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
		convRes = tf.nn.relu(tf.nn.bias_add(conv_res, b_res))
		print('convRes shape: {}'.format(convRes.get_shape().as_list()))
		
	with tf.variable_scope('conv1'):
		conv1_weights = tf.get_variable(name='weight', shape=[7, 7, 12, 64],
			                                initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1 = tf.nn.conv2d(input=convRes, filter=conv1_weights, strides=[1,2,2,1], padding='SAME')
		print('conv1 shape: {}'.format(conv1.get_shape().as_list()))
		
	with tf.name_scope('pool1'):
		pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
		lrn1 = tf.nn.local_response_normalization(input=pool1, name='lrn1')
		print('pool1 shape: {}'.format(lrn1.get_shape().as_list()))
			
	with tf.variable_scope('conv2'):
		conv2_weights = tf.get_variable(name='weight', shape=[5, 5, 64, 48],
			                                initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2 = tf.nn.conv2d(input=pool1, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		print('conv2 shape: {}'.format(conv2.get_shape().as_list()))
			
	with tf.name_scope('pool2'):
		pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
		lrn2 = tf.nn.local_response_normalization(input=pool2, name='lrn2')
		print('lrn2 shape: {}'.format(lrn2.get_shape().as_list()))
		
	feature_dim = lrn2.shape[1] * lrn2.shape[2] * lrn2.shape[3]
	lrn2_nodes = tf.reshape(lrn2, [-1, feature_dim])
		
	with tf.variable_scope('fc1'):
		fc1_weights = tf.get_variable(name='weight', shape=[feature_dim, 4096],
			                              initializer=tf.truncated_normal_initializer(stddev=0.1))
		#全连接需要正则化处理
		if regularizer != None:
			tf.add_to_collection(name='losses', value=regularizer(fc1_weights))
		fc1_biases = tf.get_variable(name='bias', shape=[4096],
			                             initializer=tf.constant_initializer(0.0))
		fc1 = tf.nn.relu(tf.matmul(lrn2_nodes, fc1_weights) + fc1_biases)
		if train:
			fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
		print('fc1 shape: {}'.format(fc1.get_shape().as_list()))
		
	with tf.variable_scope('fc2'):
		fc2_weights = tf.get_variable(name='weight', shape=[4096, 4096],
			                              initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer!= None:
			tf.add_to_collection(name='losses', value= regularizer(fc2_weights))
		fc2_biases = tf.get_variable(name='bias', shape=[4096],
				                             initializer=tf.constant_initializer(0.0))
		fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
		if train:
			fc2 = tf.nn.dropout(fc2, keep_prob=0.5)
		print('fc2 shape: {}'.format(fc2.get_shape().as_list()))
				
	with tf.variable_scope('fc3'):
		fc3_weights = tf.get_variable(name='weight', shape=[4096, NUM_LABELS],
			                              initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None:
			tf.add_to_collection(name='loses', value=regularizer(fc3_weights))
		fc3_biases = tf.get_variable(name='bias', shape=[NUM_LABELS],
			                             initializer=tf.constant_initializer(0.0))
		logit = tf.nn.relu(tf.matmul(fc2, fc3_weights) + fc3_biases)
		print('logit shape: {}'.format(logit.get_shape().as_list()))
	return logit
	
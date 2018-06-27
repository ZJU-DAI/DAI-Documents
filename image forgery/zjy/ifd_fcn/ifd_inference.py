# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn

import ifd_train
#define parameters
lstm_hidden_size = 256   #随意的数值=。=大意每个cell有多少个神经元
lstm_time_step = 64    #将fc2分成64块喂进lstm中
lstm_block_size = 64   #每块喂入的像素数
lstm_layer_num = 3
keep_prob = 0.9

def cut_to_blocks(input, batch_size, pathch_size, block_size):
	blocks = tf.Variable(tf.zeros([batch_size, lstm_time_step, block_size, block_size]))
	size = [batch_size, 8, 8]
	input = tf.reshape(input, [batch_size, pathch_size, pathch_size])
	idx =0
	step = int(pathch_size/block_size)
	for i in range(step):
		for j in range(step):
			tf.assign(blocks[idx], tf.slice(input, begin=[0, i, j], size=size))
			idx +=1
	print('blocks output shape: {}'.format(blocks.get_shape().as_list()))
	blocks = tf.reshape(blocks, shape=[batch_size, lstm_time_step, block_size*block_size])
	return blocks

def cat_to_map(input, batch_size):
	#lstm_ouput[time_step, [batch_size, output_each_cell]]
	map = tf.Variable(tf.zeros([batch_size, 16, 16*8]))
	for i in range(8):
		map_raw = tf.Variable(tf.zeros([batch_size, 16, 16]))
		for j in range(8):
			cell_out = input[i*8+j]
			cell_out = tf.reshape(cell_out, [batch_size, 16, 16])
			if j == 0:
				map_raw = cell_out
			else:
				map_raw = tf.concat(values=[map_raw, cell_out], axis=2)
		if i == 0:
			map = map_raw
		else:
			map = tf.concat([map, map_raw], axis=1)
	return map




def lstm_cell(hidden_size, keep_prob):
	cell = rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
	return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

#define the inference modle
def inference(input_tensor, n_classes, TRAIN_FLAG):

	batch_size = ifd_train.BATCH_SIZE
	input = tf.reshape(input_tensor, [-1, ifd_train.PATCH_SIZE, ifd_train.PATCH_SIZE, ifd_train.INCHANNEL])
	print('inputshape: {}'.format(input.get_shape().as_list()))
	with tf.variable_scope('conv1'):
		conv1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5], trainable=TRAIN_FLAG,
		                         padding='SAME', activation=tf.nn.relu, name='conv1')
		print('conv1 output shape: {}'.format(conv1.get_shape().as_list()))

	with tf.variable_scope('conv2'):
		conv2 = tf.layers.conv2d(inputs=conv1, filters=1, kernel_size=[5, 5], trainable=TRAIN_FLAG,
		                         padding='SAME', activation=tf.nn.relu, name='conv2')
		print('conv2 output shape: {}'.format(conv2.get_shape().as_list()))

	with tf.variable_scope('lstm'):
		_fc2 = cut_to_blocks(conv2, ifd_train.BATCH_SIZE, ifd_train.PATCH_SIZE, ifd_train.BLOCK_SIZE)
		print('_fc2 shape: {}'.format(_fc2.get_shape().as_list()))
		#use a fc layer to fit fc2 with lstm_hidden_size
		w_fc2 = tf.Variable(tf.random_normal([lstm_block_size, lstm_hidden_size]), name='w_fc2', trainable=TRAIN_FLAG)
		b_fc2 = tf.Variable(tf.constant(0.1, shape=[lstm_hidden_size]), name='b_fc2', trainable=TRAIN_FLAG)
		fc2 = tf.matmul(tf.reshape(_fc2, [-1, lstm_block_size]), w_fc2) + b_fc2
		fc2 = tf.reshape(fc2, [batch_size, lstm_time_step, lstm_hidden_size])
		print('fc2 shape: {}'.format(fc2.get_shape().as_list()))
		#set lstm
		mlstm_cell = rnn.MultiRNNCell([lstm_cell(lstm_hidden_size, keep_prob) for _ in range(lstm_layer_num)],
		                              state_is_tuple=True)
		init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
		lstm_ouput, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=fc2, initial_state=init_state,
		                                      time_major=False)
		print('lstm_ouput shape: {}'.format(lstm_ouput.get_shape().as_list()))
		# handle  lstm_ouput ==> list: [time_step, [batch_size, output_each_cell]]
		lstm_ouput = tf.unstack(tf.transpose(lstm_ouput, [1, 0, 2]))
		
		# patch_label
		w_pl = tf.get_variable(name='W_label_out', shape=[lstm_hidden_size, n_classes],
		                     dtype=tf.float32, trainable=TRAIN_FLAG,
		                     initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
		b_pl = tf.get_variable(name='b_label_out', shape=[n_classes], dtype=tf.float32,
		                     trainable=TRAIN_FLAG, initializer=tf.constant_initializer())
		logits = tf.matmul(lstm_ouput[-1], w_pl) + b_pl
		print('logits output shape: {}'.format(logits.get_shape().as_list()))
		
		# get f_lstm
		f_lstm = cat_to_map(lstm_ouput, ifd_train.BATCH_SIZE)
		print('f_lstm shape: {}'.format(f_lstm.get_shape().as_list()))
	
	with tf.variable_scope('conv3'):
		f_lstm = tf.reshape(f_lstm, [f_lstm.shape[0], f_lstm.shape[1], f_lstm.shape[2], 1])
		conv3 = tf.layers.conv2d(inputs=f_lstm, filters=32, kernel_size=[5, 5], trainable=TRAIN_FLAG,
		                         padding='SAME', activation=tf.nn.relu, name='conv3')
		print('conv3 output shape: {}'.format(conv3.get_shape().as_list()))
		
	with tf.variable_scope('pool1'):
		pool1 = tf.nn.max_pool(value=conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		print('pool1 output shape: {}'.format(pool1.get_shape().as_list()))
		
	with tf.variable_scope('fcn4'):
		w4 = tf.get_variable(name='w4', shape=[64, 64, 32, 4096], initializer=tf.glorot_uniform_initializer(),
		                     dtype=tf.float32, trainable=TRAIN_FLAG)
		b4 = tf.get_variable(name='b4', shape=[4096], trainable=TRAIN_FLAG,
		                     initializer=tf.constant_initializer())
		fcn4 = tf.nn.conv2d(input=pool1, filter=w4, strides=[1, 1, 1, 1],
		                    padding='SAME', name='fcn4')
		fcn4 = tf.nn.bias_add(fcn4, b4)
		print('fcn4 output shape: {}'.format(fcn4.get_shape().as_list()))
		
	with tf.variable_scope('fcn5'):
		w5 = tf.get_variable(name='w5', shape=[64, 64, 4096, ifd_train.N_CLASSES], initializer=tf.glorot_uniform_initializer(),
		                     dtype=tf.float32, trainable=TRAIN_FLAG)
		b5 = tf.get_variable(name='b5', shape=[ifd_train.N_CLASSES], trainable=TRAIN_FLAG,
		                     initializer=tf.constant_initializer())
		fcn5 = tf.nn.conv2d(input=fcn4, filter=w5, strides=[1, 1, 1, 1],
		                    padding='SAME', name='fcn5')
		fcn5 = tf.nn.bias_add(fcn5, b5)
		print('fcn5 output shape: {}'.format(fcn5.get_shape().as_list()))
	
	return logits, fcn5

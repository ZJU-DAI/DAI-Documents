# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn

import train
#define parameters
lstm_hidden_size = 64
lstm_block_size = 8

#define the inference modle
def inference(input_tensor, n_classes, TRAIN_FLAG):

	batch_size = train.BATCH_SIZE

	with tf.variable_scope('conv1'):
		conv1 = tf.layers.conv2d(inputs=input_tensor, filters=64, kernel_size=[5, 5], trainable=TRAIN_FLAG,
		                         padding='SAME', activation=tf.nn.relu, name='conv1')
		print('conv1 output shape: {}'.format(conv1.get_shape().as_list()))

	with tf.variable_scope('conv2'):
		conv2 = tf.layers.conv2d(inputs=conv1, filters=1, kernel_size=[5, 5], trainable=TRAIN_FLAG,
		                         padding='SAME', activation=tf.nn.relu, name='conv2')
		print('conv2 output shape: {}'.format(conv1.get_shape().as_list()))

	with tf.variable_scope('lstm'):
		fc2 = tf.reshape(conv2, [batch_size, lstm_hidden_size, lstm_block_size, lstm_block_size])
		print('fc2 shape: {}'.format(fc2.get_shape().as_list()))
		logits = tf.Variable(tf.zeros([batch_size, n_classes]), name='logits')

		for i in range(batch_size):
			with tf.variable_scope(str(i)+'_lstm'):
				block = fc2[i]
				lstm_cell = rnn.BasicLSTMCell(num_units=lstm_hidden_size, forget_bias=1.0)
				mlstm_cell = rnn.MultiRNNCell([lstm_cell], state_is_tuple=True)
				init_state = mlstm_cell.zero_state(64, dtype=tf.float32)
				lstm_ouput, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=block, initial_state=init_state,
				                                      time_major=False)
				h_state = lstm_ouput[-1:, -1, :]

				W1 = tf.get_variable(name='W_label_out', shape=[lstm_hidden_size, n_classes],
				                     dtype=tf.float32, trainable=TRAIN_FLAG,
				                     initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
				b1 = tf.get_variable(name='b_label_out', shape=[n_classes], dtype=tf.float32,
				                     trainable=TRAIN_FLAG, initializer=tf.constant_initializer())
				logit = tf.matmul(h_state, W1) + b1
				# print(i, 'patch_label output shape: {}'.format(logit.get_shape().as_list()))
				tf.assign(logits[i], logit)

		print('logits output shape: {}'.format(logits.get_shape().as_list()))


	return logits
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn

import ifd_train
#define parameters
lstm_hidden_size = 64
lstm_block_size = 8
lstm_layer_num = 3
keep_prob = 0.9

def lstm_cell(hidden_size, keep_prob):
	cell = rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
	return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

#define the inference modle
def inference(input_tensor, n_classes, TRAIN_FLAG):

	batch_size = ifd_train.BATCH_SIZE
	input = tf.reshape(input_tensor, [-1, ifd_train.PATCH_SIZE, ifd_train.PATCH_SIZE, ifd_train.INCHANNEL])
	with tf.variable_scope('conv1'):
		conv1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[5, 5], trainable=TRAIN_FLAG,
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
				#lstm_cell = rnn.BasicLSTMCell(num_units=lstm_hidden_size, forget_bias=1.0)
				mlstm_cell = rnn.MultiRNNCell([lstm_cell(lstm_hidden_size, keep_prob)for _ in range(lstm_layer_num)], state_is_tuple=True)
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
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

#set GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

import inference
import utils

#配置参数
BATCH_SIZE = 16
CHANNELS = 3
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DELAY = 0.99
REGULARAZTION_RATE = 0.0001
SAVE_STEP = 2
MOVING_AVERAGE_DELAY = 0.99
THRESHOLD = 0.125

#路径与文件名
INPUT_DATA = '../npy/0525_CORGERAGE-380.npy'
MODEL_SAVE_PATH = '../checkpoints/convRes'
MODEL_NAME = 'model.ckpt'
GRAPH_PATH = '../graphs/convRes'

def get_data(input):
	print("###############get data#################")
	prosessed_data = np.load(input)
	training_imgs = prosessed_data[0]
	training_labels = prosessed_data[1]
	validation_imgs = prosessed_data[2]
	validation_labels = prosessed_data[3]
	testing_imgs = prosessed_data[4]
	testing_labels = prosessed_data[5]
	print("%d train, %d validation, %d test" % (len(training_imgs),
	                                            len(validation_imgs),
	                                            len(testing_imgs)))
	print("###############finish get data!#################")
	num_f = 0
	labels, num_f = utils.get_label(training_labels, THRESHOLD, num_f)
	return training_imgs, labels, num_f


def train(imgs, labels, train_img_num, img_size):
	#输入输出的placeholder
	img = tf.placeholder(tf.float32, [None, img_size, img_size, CHANNELS])
	label = tf.placeholder(tf.int64, [None])
	
	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
	
	#调用前向网络
	pre_label = inference.inference(img, True, regularizer)
	global_step = tf.Variable(0, trainable=False)
	
	#滑动平均
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DELAY, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	
	with tf.name_scope('loss'):
		#logits = tf.arg_max(pre_label, 1)
		#print(logits.get_shape(), label.get_shape())
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= pre_label, labels= label)
		print('cross_entropy shape: {}'.format(cross_entropy.get_shape().as_list()))
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		print('cross_entropy_mean shape: {}'.format(cross_entropy_mean.get_shape().as_list()))
		loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
		
		#loss = cross_entropy_mean
	
	with tf.name_scope('lr'):
		learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE,
		                                           global_step=global_step,
		                                           decay_steps= train_img_num/BATCH_SIZE,
		                                           decay_rate=LEARNING_RATE_DELAY)
		
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
	with tf.control_dependencies([train_step, variable_averages_op]):
		train_op = tf.no_op(name='train')
	
	with tf.name_scope('summaries'):
		tf.summary.scalar('loss', loss)
		#tf.summary.histogram('histogram loss', loss)
		summary_op = tf.summary.merge_all()
		
	saver = tf.train.Saver()
	writer = tf.summary.FileWriter(GRAPH_PATH, tf.get_default_graph())
	with tf.Session() as sess:
		utils.safe_mkdir('../checkpoints')
		utils.safe_mkdir(MODEL_SAVE_PATH)
		tf.global_variables_initializer().run()
		
		print("############  check the saver  ################")
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(MODEL_SAVE_PATH + '/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('#####  seccessfully load the saver #######')
		train_epochs = int(train_img_num / BATCH_SIZE)
		print('train_epochs: ',  train_epochs)
		for i in range(train_epochs):
			start = i * BATCH_SIZE
			end = start + BATCH_SIZE
			_, loss_value, step, summaries = sess.run([train_op, loss, global_step, summary_op],
			                               feed_dict={img: imgs[start: end], label: labels[start: end]})
			writer.add_summary(summaries, global_step=step)
			if i % SAVE_STEP == 0:
				print("training steps: %d, loss on training batch is: %g." %(step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)
				

def main(argv=None):
	imgs, labels, num_f = get_data(INPUT_DATA)
	train_img_num = len(imgs)
	img_size = imgs[0].shape[0]
	train(imgs, labels, train_img_num, img_size)

if __name__ == "__main__":
	tf.app.run()
			
			
		
		
	
	
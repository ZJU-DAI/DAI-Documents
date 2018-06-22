# -*- coding: utf-8 -*-

import time
import tensorflow as tf

import inference
import train
import utils

EVAL_INTERVAL_SECS = 10

def get_data(input):
	print("###############get data#################")
	prosessed_data = np.load(input)
	validation_imgs = prosessed_data[2]
	validation_labels = prosessed_data[3]
	print("###############finish get data!#################")
	num_f = 0
	labels, num_f = utils.get_label(validation_labels, train.THRESHOLD, num_f)
	return validation_imgs, labels,num_f

def evaluate(imgs, labels, img_size):
	with tf.Graph() .as_default() as g :
	
		# 定义输入输出格式
		img = tf.placeholder(tf.float32, [None, img_size, img_size, train.CHANNELS])
		label = tf.placeholder(tf.int64, [None, 2])
		validate_feed = {img: imgs, label: labels}
		
		# 调用前向网络,此处正则损失设为None
		pre_label = inference.inference(img, False, None)
		
		#计算正确率
		correct_prediction = tf.equal(tf.argmax(pre_label, 1), label)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		#变量重命名 黑科技
		variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DELAY)
		variable_to_restore = variable_averages.variables_to_restore()
		
		saver = tf.train.Saver(variable_to_restore)
		
		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
				
				if ckpt and ckpt.model_checkpoint_state:
					saver.restore(sess, ckpt.model_checkpoint_state)
					global_step = ckpt.model_checkpoint_state.split('/')[-1].split('-')[-1]
					
					accuracy_score =sess.run(accuracy, feed_dict=validate_feed)
					
					print("training steps: %d, validation acc is: %g." % (global_step, accuracy_score))
				else :
					print("No checkpoint file found")
					return
				
				time.sleep(EVAL_INTERVAL_SECS)
				
				
def main (argv=None):
	imgs, labels, num_f = get_data(INPUT_DATA)
	img_size = imgs.shape[2]
	evaluate(imgs, labels, img_size)
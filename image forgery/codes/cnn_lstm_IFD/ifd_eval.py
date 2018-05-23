# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import tensorflow as tf


import ifd_inference as inf
import ifd_train
import utils
#隔一定时间加载模型，用测试集测试最新模型的正确率
EVAL_INTERVAL_SECS = 100
batch_size = ifd_train.BATCH_SIZE

def eval(INPUT_DATA):
	with tf.name_scope('data'):
		print("###############get data#################")
		prosessed_data = np.load(INPUT_DATA)
		testing_imgs = prosessed_data[4]
		testing_labels = prosessed_data[5]
		print("###############finish get data!#################")


	with tf.Graph().as_default() as g:
		#定义输入输出格式
		patch = tf.placeholder(tf.float32, [ifd_train.BATCH_SIZE, ifd_train.PATCH_SIZE, ifd_train.PATCH_SIZE, ifd_train.INCHANNEL], name='test_patch')
		labels = tf.placeholder(tf.int64, [ifd_train.BATCH_SIZE], name='test_label')

		#调用网络
		logits = inf.inference(patch, n_classes=ifd_train.N_CLASSES, TRAIN_FLAG=False)
		
		with tf.name_scope('accuracy'):
			label_one_hot = tf.one_hot(labels, depth=ifd_train.N_CLASSES, on_value=1)
			preds = tf.nn.softmax(logits)
			correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label_one_hot, 1))
			accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

		saver = tf.train.Saver()

		while True:
			with tf.Session() as sess:
				print("############  load the model  ################")
				ckpt = tf.train.get_checkpoint_state(ifd_train.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)

					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					
					test_EPOCH = len(testing_imgs)
					acc_total = 0
					for epoch in range(test_EPOCH):
						acc_total_epoch = 0
						start = 0
						end = ifd_train.BATCH_SIZE
						# load data
						patch_data, label_data = utils.get_patch(testing_imgs[epoch], testing_labels[epoch],
						                                         ifd_train.PATCH_SIZE, ifd_train.STRIDE,
						                                         ifd_train.IMG_SIZE,
						                                         ifd_train.INCHANNEL, ifd_train.THRESHOLD)
						N_PATCH = patch_data.shape[0]
						test_step = int(N_PATCH /ifd_train.BATCH_SIZE)
				
						for idx in range(test_step):
							
							accuracy_score = sess.run(accuracy, feed_dict={patch: patch_data[start: end],
					                                      labels:label_data[start: end]})
							accuracy_score /= ifd_train.BATCH_SIZE
							acc_total_epoch += accuracy_score
							start += ifd_train.BATCH_SIZE
							end += ifd_train.BATCH_SIZE
						print('Average Accuracy at step {0}: {1}'.format(epoch, acc_total_epoch / test_step))
						acc_total += acc_total_epoch
					print('Average Accuracy at step {0}: {1}'.format(global_step, acc_total / test_EPOCH))
				else:
					print('No checkpoint file found')
			return

def main(argv=None):
	eval(ifd_train.INPUT_DATA)

if __name__ == '__main__':
	tf.app.run()

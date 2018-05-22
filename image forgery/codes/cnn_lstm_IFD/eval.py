# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import tensorflow as tf


import inference as inf
import train

#隔一定时间加载模型，用测试集测试最新模型的正确率
EVAL_INTERVAL_SECS = 100
batch_size = train.BATCH_SIZE

def eval(INPUT_DATA):
	with tf.name_scope('data'):
		print("###############get data#################")
		prosessed_data = np.load(INPUT_DATA)
		training_patches = prosessed_data[0]
		n_training_patches = len(training_patches)
		training_labels = prosessed_data[1]
		validation_patches = prosessed_data[2]
		validation_label = prosessed_data[3]
		testing_patches = prosessed_data[4]
		testing_label = prosessed_data[5]
		print("%d train, %d validation, %d test" % (n_training_patches,
		                                            len(validation_patches),
		                                            len(testing_patches)))
		print("###############finish get data!#################")


	with tf.Graph().as_default() as g:
		#定义输入输出格式
		patch = tf.placeholder(tf.float32, [None, train.PATCH_SIZE, train.PATCH_SIZE, train.INCHANNEL], name='test_patch')
		labels = tf.placeholder(tf.int64, [None, train.N_CLASSES], name='test_label')

		#调用网络
		logits = inf.inference(patch, n_classes=train.N_CLASSES, TRAIN_FLAG=False)

		preds = tf.nn.softmax(logits)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

		saver = tf.train.Saver()

		while True:
			with tf.Session() as sess:
				print("############  load the model  ################")
				ckpt = tf.train.get_checkpoint_state(os.path.dirname(train.MODEL_SAVE_PATH + '/checkpoint'))
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)

					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

					test_step = int(testing_patches.shape[0]/train.BATCH_SIZE)
					acc_total = 0
					for idx in range(test_step):
						accuracy_score = sess.run(accuracy, feed_dict={patch: testing_patches[idx * batch_size:(idx + 1) * batch_size],
					                                      labels:testing_label[idx * batch_size:(idx + 1) * batch_size]})
						acc_total += accuracy_score

					print('Average Accuracy at step {0}: {1}'.format(global_step, acc_total / test_step))

				else:
					print('No checkpoint file found')
					return

def main(argv=None):
	eval(train.INPUT_DATA)

if __name__ == '__main__':
	tf.app.run()

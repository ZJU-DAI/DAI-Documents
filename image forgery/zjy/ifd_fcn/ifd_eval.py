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

def update_actualFlag(gt_label):
	scale = np.count_nonzero(gt_label) / (ifd_train.IMG_SIZE * ifd_train.IMG_SIZE)
	if scale > 0.05:
		actual_Flag = True
	else:
		actual_Flag = False
	return actual_Flag

def update_predictFlag(pred_l):
	idx = 0
	for i in range(pred_l.shape[0]):
		label = pred_l[i]
		if label[0] < label[1]:
			idx += 1
	scale = idx / pred_l.shape[0]
	if scale > 0.05:
		predict_Flag = True
	else:
		predict_Flag = False
	return predict_Flag
	

def write2file(OUTFILE, stride, THRESHOLD, PATCH_SIZE, TP, FP, TN, FN, accuracy):
	# write a record file
	with open(ifd_train.OUTFILE, 'wa') as f:
		f.write(stride+'\t'+THRESHOLD+'\t'+PATCH_SIZE+'\t'+TP+'\t'+FP+'\t'+TN+'\t'+FN+'\t'+accuracy)
	return


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
					acc_total_test = 0
					m_idx = 0
					
					TP = 0
					FP = 0
					TN = 0
					FN = 0
					
					# one epoch one img
					for epoch in range(test_EPOCH):
						# load data
						patch_data, label_data, m_idx = utils.get_patch(testing_imgs[epoch], testing_labels[epoch],
						                                                ifd_train.PATCH_SIZE, ifd_train.STRIDE,
						                                                ifd_train.IMG_SIZE,
						                                                ifd_train.INCHANNEL, ifd_train.THRESHOLD, m_idx)
						N_PATCH = patch_data.shape[0]
						
						acc_epoch, pred_labels = sess.run([accuracy, preds], feed_dict={patch: patch_data,
						                                                                labels: label_data})
						acc_epoch /= N_PATCH
						print('Average Accuracy at step {0}: {1}'.format(epoch, acc_epoch))
						acc_total_test += acc_epoch
						
						# update the actual_Flag
						actual_Flag = update_actualFlag(testing_labels[epoch])
						# update the predict_Flag
						pred_l = np.asarray(pred_labels)
						pred_l = np.reshape(pred_l, [-1, 2])
						predict_Flag = update_predictFlag(pred_l)
						print('actual_Flag:{0}: predict_Flag:{1}'.format(actual_Flag, predict_Flag))
						
						# update the actual_Flag
						actual_Flag = update_actualFlag(testing_labels[epoch])
						# update the predict_Flag
						pred_l = np.asarray(pred_labels)
						pred_l = np.reshape(pred_l, [-1, 2])
						predict_Flag = update_predictFlag(pred_l)
						#update TP, FP, TN, FN
						if predict_Flag and actual_Flag:
							TP += 1
						elif predict_Flag and (not actual_Flag):
							FP += 1
						elif (not predict_Flag) and (not actual_Flag):
							TN += 1
						else:
							FN += 1
					
					#the final accuracy
					acc = acc_total_test / test_EPOCH
					print('Average Accuracy at step {0}: {1}'.format(global_step, acc))
				
					write2file(ifd_train.OUTFILE, ifd_train.STRIDE, ifd_train.THRESHOLD, ifd_train.PATCH_SIZE,
					           TP, FP, TN, FN, acc)
				else:
					print('No checkpoint file found')
			return

def main(argv=None):
	eval(ifd_train.INPUT_DATA)

if __name__ == '__main__':
	tf.app.run()

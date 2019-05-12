# -*- coding: utf-8 -*-

#import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import tensorflow as tf
import numpy as np

import ifd_inference as inf
import utils

#set GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

#define parameters
IMG_SIZE = 586   #每张图片的大小
STRIDE = 32     #图片中提取patch的strides
BATCH_SIZE = 64  #每批数据中patch数
PATCH_SIZE = 64
BLOCK_SIZE = 8
INCHANNEL = 3
THRESHOLD = 0.125
N_CLASSES = 2
LEARNING_RATE_BASE = 0.8
lEARNING_RATE_DECAY = 0.99

LOSS_OUT_STEPS = 10   #间隔一定步数后输出一次loss，监控模型
SAVE_NUM = 10  #间隔多少张图片后，保存一次模型

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

TRAIN_FLAG = False
#INPUT_PATH = 'E:\\tf\\create_tfRecord\\npy-16\\'
INPUT_DATA = '../npy/0601_IMD.npy'
MODEL_SAVE_PATH ='checkpoints/cnn_lstm_IFD_8'
MODEL_NAME = 'model.ckpt'
GRAPH_PATH = './graphs/cnn_lstm_IFD'
OUTFILE = 'record/record.txt'


def train():
	with tf.name_scope('data'):
		print("###############get data#################")
		prosessed_data = np.load(INPUT_DATA)
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
		
	#define input / output
	patch = tf.placeholder(tf.float32, [BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, INCHANNEL], name='input-patch')
	labels = tf.placeholder(tf.int64, [BATCH_SIZE], name='patch-label')
	pixel_labels = tf.placeholder(tf.int64, [BATCH_SIZE, PATCH_SIZE, PATCH_SIZE], name='pixel-label')
	# pixel_labels = tf.reshape(pixel_labels, shape=[BATCH_SIZE*PATCH_SIZE*PATCH_SIZE])
	#import the model
	TRAIN_FLAG = True
	logits, fcn5 = inf.inference(patch, N_CLASSES, TRAIN_FLAG)
	global_step = tf.Variable(0, trainable=False)

	with tf.name_scope('loss'):
		print('labels:', labels.shape)
		label_one_hot = tf.one_hot(labels, depth=N_CLASSES, on_value=1)
		entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_one_hot, logits=logits)
		#entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels, 1), logits=logits)
		loss_p = tf.reduce_mean(entropy, name='loss_p')/BATCH_SIZE
		#loss_s
		pixel_logits = tf.reshape(fcn5, shape=[fcn5.shape[0]*fcn5.shape[1]*fcn5.shape[2], N_CLASSES])
		print('pixel_logits shape: {}'.format(pixel_logits.get_shape().as_list()))
		pixel_label_one_hot = tf.one_hot(pixel_labels, depth=N_CLASSES, on_value=1)
		print('pixel_label_one_hot shape: {}'.format(pixel_label_one_hot.get_shape().as_list()))
		entropy_pixel = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pixel_label_one_hot, logits=pixel_logits)
		loss_s = tf.reduce_mean(entropy_pixel, name='loss_s')/(BATCH_SIZE*PATCH_SIZE*PATCH_SIZE)
		loss = loss_p+loss_s
		
	
	with tf.name_scope('accuracy'):
		preds = tf.nn.softmax(logits)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label_one_hot, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	with tf.name_scope('learning_rate'):
		learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
		                                           decay_steps=BATCH_SIZE, decay_rate=lEARNING_RATE_DECAY)


	with tf.name_scope('summaries'):
		tf.summary.scalar('loss', loss)
		tf.summary.scalar('accuracy', accuracy)
		tf.summary.histogram('histogram loss', loss)
		summary_op = tf.summary.merge_all()

	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

	saver = tf.train.Saver()
	writer = tf.summary.FileWriter(GRAPH_PATH, tf.get_default_graph())
	with tf.Session(config=config) as sess:
		utils.safe_mkdir('checkpoints')
		utils.safe_mkdir(MODEL_SAVE_PATH)
		sess.run(tf.global_variables_initializer())

		print("############  check the saver  ################")
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(MODEL_SAVE_PATH+'/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('#####  seccessfully load the saver #######')
		step = global_step.eval()
		print("############  start training  ################")
		num_img = len(training_imgs)
		total_loss = 0
		TRAIN_FLAG = True
		for n_img in range(num_img):
			start_time = time.time()
			
			m_idx = 0  # 统计一下每张图中被篡改过patch的数量
			# 以img为单位输入切割后得patches数据
			patch_data, label_data, pixel_label, m_idx = utils.get_patch(training_imgs[n_img], training_labels[n_img],
			                                                PATCH_SIZE, STRIDE, IMG_SIZE, INCHANNEL, THRESHOLD, m_idx)
			N_PATCH = patch_data.shape[0]
			print('Patch num: {0}'.format(N_PATCH))
			#再分割为小batch送入inference
			num_batch = int(N_PATCH/BATCH_SIZE)
			start = 0
			end = BATCH_SIZE
			for batch_idx in range(num_batch):
			
				_, l, summaries, = sess.run([train_op, loss, summary_op],
			                            feed_dict={patch: patch_data[start: end], labels: label_data[start: end],
			                                       pixel_labels:pixel_label[start: end]})
				writer.add_summary(summaries, global_step=step)
				if (step + 1) % LOSS_OUT_STEPS == 0:
					print('Loss at step {0}: {1}'.format(step, l))
				step += 1
				total_loss += l
			# each SAVE_NUM epoches :calculate the accurancy on the validation data, and save the model
			if (n_img + 1) % SAVE_NUM == 0:
				print('Average loss at img {0}: {1}'.format(n_img, total_loss / (n_img*num_batch)))
				print('Took: {0} seconds for one epoch'.format(time.time() - start_time))
				saver.save(sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)
				

def main(argv=None):
	train()

if __name__ == '__main__':
	tf.app.run()


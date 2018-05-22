# -*- coding: utf-8 -*-

import os
import time
import tensorflow as tf
import numpy as np

import inference as inf

#define parameters
N_EPOCH =1        #每次训练的图片数
N_PATCH = 3000       #每张图片中的patch数
BATCH_SIZE = 64  #每批数据中patch数
PATCH_SIZE = 64
INCHANNEL = 3
N_CLASSES = 2
LEARNING_RATE_BASE = 0.8
lEARNING_RATE_DECAY = 0.99

LOSS_OUT_STEPS =  100   #间隔一定步数后输出一次loss，监控模型
SAVE_NUM = 16   #间隔多少张图片后，保存一次模型

TRAIN_FLAG = False
#FILE PATH
INPUT_DATA = '0521_IMD_03.npy'
MODEL_SAVE_PATH ='./checkpoints/cnn_lstm_IFD'
GRAPH_PATH = './graphs/cnn_lstm_IFD'


def train():
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


	#define input / output
	patch = tf.placeholder(tf.float32, [None, PATCH_SIZE, PATCH_SIZE, INCHANNEL], name='input-patch')
	labels = tf.placeholder(tf.int64, [None, N_CLASSES], name='input-label')

	#import the model
	TRAIN_FLAG = True
	logits = inf.inference(patch, N_CLASSES, TRAIN_FLAG)
	global_step = tf.Variable(0, trainable=False)

	with tf.name_scope('loss'):
		entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels, 1), logits=logits)
		loss = tf.reduce_mean(entropy, name='loss')

	with tf.name_scope('learning_rate'):
		learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
		                                           decay_steps=BATCH_SIZE, decay_rate=lEARNING_RATE_DECAY)


	with tf.name_scope('summaries'):
		tf.summary.scalar('loss', loss)
		#tf.summary.scalar('accuracy', accuracy)
		tf.summary.histogram('histogram loss', loss)
		summary_op = tf.summary.merge_all()

	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

	saver = tf.train.Saver()
	writer = tf.summary.FileWriter(GRAPH_PATH, tf.get_default_graph())
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		print("############  check the saver  ################")
		ckpt = tf.train.get_checkpoint_state(os.path.dirname(MODEL_SAVE_PATH+'/checkpoint'))
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		step = global_step.eval()
		print("############  strat training  ################")
		for epoch in range(N_EPOCH):
			start_time = time.time()
			total_loss =0
			train_step = int(N_PATCH/BATCH_SIZE)

			for batch_idx in range(train_step):
				_, l, summaries = sess.run([train_op, loss, summary_op],
				                           feed_dict={patch: training_patches[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE],
				                                      labels: training_labels[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]})
				writer.add_summary(summaries, global_step=step)
				#间隔一定步数后输出一次loss，监控模型
				if (step + 1) % LOSS_OUT_STEPS == 0:
					print('Loss at step {0}: {1}'.format(step, l))
					step += 1
				total_loss += l
			print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/train_step))
			print('Took: {0} seconds'.format(time.time() - start_time))

			if (epoch + 1) % SAVE_NUM == 0:
				saver.save(sess, save_path=MODEL_SAVE_PATH, global_step=step)

def main(argv=None):
	train()

if __name__ == '__main__':
	tf.app.run()


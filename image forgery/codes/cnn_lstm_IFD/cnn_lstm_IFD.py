""" Using convolutional net on MNIST dataset of handwritten digits
MNIST dataset: http://yann.lecun.com/exdb/mnist/
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 07
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

class LSTM(object):
	def __init__(self):
		print("###############init paraments#################")
		self.INPUT_DATA = '0521_IMD_02.npy'

		self.lr = 0.001     #学习率
		self.batch_size = 32   #每批数据的规模
		#self.keep_prob = tf.constant(0.75)
		self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step') #saver用的全局计数器
		self.n_classes = 2    #类别个数
		self.skip_step = 20   #显示loss函数的计数器
		self.n_test = 10000
		self.training = False

		self.patch_size = 64  #要改
		self.inchannels =3

		self.lstm_block_size = 8 #lstm输入每个cell中的像素数
		self.lstm_num_steps = 64 #单个数据中的序列长度
		self.lstm_hidden_size = 64 #每个隐藏层的节点数，此处我理解为cell数
		self.lstm_layer_num = 3 #LSTM layer 的层数

		#创建占位符
		self.patch = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.inchannels])
		self.label = tf.placeholder(tf.int64, [None, self.n_classes])
		self.block = tf.placeholder(tf.float32, [None, self.lstm_block_size])



	def get_data(self):
		with tf.name_scope('data'):
			print("###############get data#################")
			prosessed_data = np.load(self.INPUT_DATA)
			self.training_patches = prosessed_data[0]
			self.n_training_patches = len(self.training_patches)
			self.training_labels = prosessed_data[1]
			self.validation_patches = prosessed_data[2]
			self.validation_label = prosessed_data[3]
			self.testing_patches = prosessed_data[4]
			self.testing_label = prosessed_data[5]
			print("%d train, %d validation, %d test"%(self.n_training_patches,
			                                          len(self.validation_patches),
			                                          len(self.testing_patches)))
			print("###############finish get data!#################")


	def cnn_lstm(self):

		with tf.variable_scope('cnn1'):
			_patch = tf.reshape(self.patch, [self.batch_size, self.patch_size, self.patch_size, self.inchannels])
			conv1 = tf.layers.conv2d(inputs=_patch, filters=64,kernel_size=[5, 5], trainable=self.training,
			                         padding='SAME', activation=tf.nn.relu, name='conv1')
			conv2 = tf.layers.conv2d(inputs=conv1, filters=1, kernel_size=[5, 5], trainable=self.training,
			                         padding='SAME', activation=tf.nn.relu, name='conv2')
			print('conv2 output shape: {}'.format(conv2.get_shape().as_list()))



		#reshape conv2 into a 8*8 blocks 2D feature map
		#the size should be [block_num, block_size, block_size

		with tf.name_scope('lstm'):
			fc2 = tf.reshape(conv2, [self.batch_size, self.lstm_hidden_size, self.lstm_block_size, self.lstm_block_size])
			print('fc2 shape: {}'.format(fc2.get_shape().as_list()))
			lstm_batchsize = fc2.shape[0]
			self.logits = tf.Variable(tf.zeros([fc2.shape[0], self.n_classes]), name='logits')
			for i in range(fc2.shape[0]):
				with tf.variable_scope(str(i)+'_lism'):
					block = fc2[i]
					lstm_cell = rnn.BasicLSTMCell(num_units=self.lstm_hidden_size, forget_bias=1.0)
					mlstm_cell = rnn.MultiRNNCell([lstm_cell], state_is_tuple=True)

					init_state = mlstm_cell.zero_state(64, dtype=tf.float32)
					lstm_ouput, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=block, initial_state=init_state, time_major=False)

					#print('lstm output shape: {}'.format(lstm_ouput.get_shape().as_list()))
					h_state = lstm_ouput[-1:, -1, :]
					#print('h_state shape: {}'.format(h_state.get_shape().as_list()))
					#"reshape" the output
					# for patch_label
					W1 = tf.get_variable(name='W_label_out', shape=[self.lstm_hidden_size, self.n_classes],
					                     dtype=tf.float32, trainable=self.training,
					                     initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
					b1 = tf.get_variable(name='b_label_out', shape=[self.n_classes], dtype=tf.float32,
					                     trainable=self.training, initializer=tf.constant_initializer())
					logit = tf.matmul(h_state, W1) + b1
					#print(i, 'patch_label output shape: {}'.format(logit.get_shape().as_list()))
					tf.assign(self.logits[i], logit)

			print('logits output shape: {}'.format(self.logits.get_shape().as_list()))

			'''
			# for segmentation mask
			W2 = tf.get_variable(name='W_mask_out',
			                    shape=[h, 256],
			                    dtype=tf.float32,
			                    initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
			b2 = tf.get_variable(name='b_mask_out',
			                    shape=[],
			                    dtype=tf.float32,
			                    initializer=tf.constant_initializer())

			f_lstm = tf.matmul(lstm_ouput, W2) + b2
			print('f_lstm output shape: {}'.format(f_lstm.get_shape().as_list()))


			f_lstm = tf.reshape(f_lstm, [-1, 128, 128, 1])
		
		with tf.variable_scope("fcn"):
			conv3 = tf.layers.conv2d(inputs=f_lstm, filters=32, kernel_size=[5, 5],
				                         padding='SAME', activation=tf.nn.relu, name='conv3')
			print('conv3 output shape: {}'.format(conv3.get_shape().as_list()))
			pool1 = tf.layers.max_pooling2d(inputs=conv3,
		                                pool_size=[3, 3],
		                                strides = 2,
			                            padding='SAME',
		                                name='pool1')
			print('pool1 output shape: {}'.format(pool1.get_shape().as_list()))
			fconv4 = 
		'''


	def loss(self):
		with tf.name_scope('loss'):
			entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
			self.loss = tf.reduce_mean(entropy, name='loss')

	def optimize(self):
		self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
		                                                    global_step=self.gstep)

	def summary(self):
		with tf.name_scope('summaries'):
			tf.summary.scalar('loss', self.loss)
			tf.summary.scalar('accuracy', self.accuracy)
			tf.summary.histogram('histogram loss', self.loss)
			self.summary_op = tf.summary.merge_all()

	def eval(self):
		with tf.name_scope('predict'):
			preds = tf.nn.softmax(self.logits)
			correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	def build(self):
		'''
		Build the computation graph
		'''
		self.get_data()
		self.cnn_lstm()
		self.loss()
		self.optimize()
		self.eval()
		self.summary()


	def train_one_epoch(self, sess, saver,  writer, epoch, step):
	#def train_one_epoch(self, sess, saver, init,  epoch, step):
		start_time = time.time()
		self.training = True
		total_loss = 0
		n_batches = 0
		print(int(self.n_training_patches/self.batch_size))
		try:
			for i in range(int(self.n_training_patches/self.batch_size)):
				_, l, summaries = sess.run([self.opt, self.loss, self.summary_op],
			                            feed_dict={self.patch: self.training_patches[i*self.batch_size:(i+1)*self.batch_size],
			                                      self.label: self.training_labels[i*self.batch_size:(i+1)*self.batch_size]})
				writer.add_summary(summaries, global_step=step)

				# 间隔几步，输出loss
				if (step + 1) % self.skip_step == 0:
					print('Loss at step {0}: {1}'.format(step, l))
				print('Loss at batch {0}: {1}'.format(n_batches, l))
				step += 1
				total_loss += 1
				n_batches += 1
		except tf.errors.OutOfRangeError:
			pass
		saver.save(sess, 'checkpoints/cnn_LSTM_IFD', step)
		print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
		print('Took: {0} seconds'.format(time.time() - start_time))
		return step

	def eval_once(self, sess, writer, epoch, step):
	#def eval_once(self, sess, init, epoch, step):
		start_time = time.time()
		self.training = False
		total_correct_preds = 0

		try:

			accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op],
				                                     feed_dict={self.patch: self.testing_patches[0:self.batch_size],
				                                                self.label: self.testing_label[0:self.batch_size]})
			writer.add_summary(summaries, global_step=step)
			total_correct_preds += accuracy_batch
		except tf.errors.OutOfRangeError:
			pass

		print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.n_test))
		print('Took: {0} seconds'.format(time.time() - start_time))

	def train(self, n_epochs):
		safe_mkdir('checkpoints')
		safe_mkdir('checkpoints/convnet_layers')
		writer = tf.summary.FileWriter('./graphs/convnet_layers', tf.get_default_graph())

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/checkpoint'))
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)

			step = self.gstep.eval()
			print("############  strat training  ################")
			for epoch in range(n_epochs):
				step = self.train_one_epoch(sess, saver,  writer, epoch, step)
				self.eval_once(sess,  writer, epoch, step)
		writer.close()


if __name__ == '__main__':
	model = LSTM()
	model.build()
	model.train(n_epochs=1)

	######################
	#create.get_img_dataset("IMD", strides= 8, patch_size=64)

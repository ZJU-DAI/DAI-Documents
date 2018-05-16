import os

from tensorflow import TFRecordReader

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time

import tensorflow as tf
import numpy as np
import utils



class ConvNet(object):
	# Step 1 : set
	def _init_(self):
		self.lr = 1e-6
		self.batch_size = 32
		self.keep_prob = tf.constant(0.5)
		self.gstep = tf.Variable(0, dtype=tf.int32,
								 trainable=False, name='global_step')
		self.n_classes = 5
		self.skip_step = 200
		self.n_test = 10000
		self.training = True

		#data
		self.train_file = 'E:/tf/create_tfRecord/train/train_0516.tfrecords'
		self.test_file = 'E:/tf/create_tfRecord/train/test_0516.tfrecords'
		self.classes = {'CND', 'CNN', 'CRN', 'NND', 'NNN', 'NRD', 'NRN'}

		#for new conv layer
		self._weight = tf.Variable(tf.random_normal([5, 5, 1, 12], dtype=tf.float32))
		self._new_weight = tf.placeholder(dtype=tf.float32, shape=[5, 5, 1, 12], name='new_weight')

		self._weight_update = tf.assign(self._weight, self._new_weight)

	#在convRes layer 对卷积核进行限制
	def assign_weight(self, session, weight_value):
		weight_sum = tf.abs(tf.reduce_sum(weight_value, [1, 1, 0]))
		zero_weight = tf.Variable(tf.zeros([1, 12]))
		one_weight = tf.Variable(tf.fill([1, 12], -1.0))
		session.run(tf.assign(weight_value[2, 2], zero_weight))
		session.run(weight_sum)
		for i in range(weight_value.shape[0]):
			for j in range(weight_value.shape[1]):
				for k in range(weight_value.shape[3]):
					# print(sess.run(weight_sum[0,k]))
					session.run(tf.assign(weight_value[i, j, 0, k],
					                      tf.div(weight_value[i, j, 0, k], weight_sum[0, k])))

		session.run(tf.assign(weight_value[2, 2], one_weight))
		session.run(self._weight_update, feed_dict={self._new_weight: weight_value})



	# Step 2 : input data
	def get_data(self):
		with tf.name_scope('data'):
			train_data = tf.train.string_input_producer([self.train_file])
			test_data = tf.train.string_input_producer([self.test_file])
			reader = tf.TFRecordReader()
			_, serialized_example = reader.read(train_data) # 返回文件名和文件

			features = tf.parse_single_example(serialized_example,
	                                        features={
		                                       'label': tf.FixedLenFeature([], tf.int64),
		                                       'img_raw': tf.FixedLenFeature([], tf.string),
	                                       })  # 取出包含image和label的feature对象
			self.img = tf.decode_raw(features['img_raw'], tf.uint8)
			#self.img = tf.reshape(image, shape=[-1,227,227,1])
			self.label = tf.cast(features['label'], tf.int32)
			#此处可能有问题
			iterator = tf.data.Iterator.from_structure(train_data.output_types,
			                                           train_data.output_shapes)
			self.train_init = iterator.make_initializer(train_data)
			self.test_init = iterator.make_initializer(test_data)


	# Step 3 : build the model
	def inference(self):
		conv_res = tf.nn.conv2d(input=self.img,
		                        filter=self._weight,
		                        strides=1,
		                        padding='VALID',
		                        name='convRes')
		conv1 = tf.layers.conv2d(inputs=conv_res,
								 filters=64,
								 kernel_size=[7, 7],
								 strides=2,
								 padding='VALID',
								 activation=tf.nn.relu,
								 name='conv1')
		pool1 = tf.layers.max_pooling2d(inputs=conv1,
									    pool_size=[3, 3],
									    strides=2,
									    name='pool1')
		lrn1 = tf.nn.local_response_normalization(inputs=pool1,
												  name='lrn1')
		conv2 = tf.layers.conv2d(inputs=lrn1,
								 filters=48,
								 kernel_size=[5, 5],
								 padding='VALIDVALID',
								 activation=tf.nn.relu,
								 name='conv2')
		pool2 = tf.layers.max_pooling2d(inputs=conv2,
										pool_size=[3, 3],
										strides=2,
										name='pool2'
										)
		lrn2 = tf.nn.local_response_normalization(inputs=pool2,
												  name='lrn2')

		feature_dim = lrn2.shape[1] * lrn2.shape[2] * lrn2.shape[3]
		lrn2 = tf.reshape(lrn2, [-1, feature_dim])

		fc1 = tf.layers.dense(inputs=lrn2,
							  units=4096,
							  activation=tf.nn.relu,
							  name='fc1_relu')
		dropout1 = tf.layers.dropout(inputs=fc1,
									 rate = self.keep_prob,
									 training=self.training,
									 name = 'dropout1')
		fc2 = tf.layers.dense(inputs=dropout1,
							  units=4096,
							  activation=tf.nn.relu,
							  name='fc2_relu')
		dropout2 = tf.layers.dropout(inputs=fc2,
									 rate=self.keep_prob,
									 training=self.training,
									 name='dropout2')
		self.logits = tf.layers.dense(inputs=dropout2,
									  units=self.classes,
									  name='fc3')


	# Step 4 : define the loss function
	def loss(self):
		with tf.name_scope('loss'):
			entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
																  logits=self.logits)
			self.loss = tf.reduce_mean(entropy, name='loss')

	# Step 5 : difine optimizer
	# using SGD to minimize cost???why
	def optimize(self):
		#self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
		self.opt = tf.train.MomentumOptimizer(learning_rate=self.lr,
											  momentum=0.9)

	# Step 6 : 求accuracy
	# Count the number of right predictions in a batch
	def eval(self):
		with tf.name_scope('predict'):
			preds = tf.nn.softmax(self.logits)
			correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	# Step 7 : visualize with tensorboard
	# Create summaries to write on TensorBoard
	def summary(self):
		with tf.name_scope('summaries'):
			tf.summary.scalar('loss', self.loss)
			tf.summary.scalar('accuracy', self.accuracy)
			tf.summary.histogram('histogram', self.loss)
			self.summary_op = tf.summary.merge_all()

	###组合啦
	def build(self):
		self._init_()
		self.get_data()
		self.inference()
		self.loss()        ##不确定
		self.optimize()
		self.eval()        ##不确定
		self.summary()

	# 训练开始！
	# 先定义一次训练
	def train_one_epoch(self, sess, saver, init, writer, epoch, step):
		start_time = time.time()
		sess.run(init)
		self.training = True
		total_loss = 0
		n_batches = 0
		try:
			while True:
				_, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
				writer.add_summary(summaries, global_step=step)
				# 间隔几步，输出loss
				if (step + 1) % self.skip_step == 0:
					print('Loss at step {0}: {1}'.format(step, l))
				step += 1
				total_loss += 1
				n_batches += 1
		except tf.errors.OutOfRangeError:
			pass
		saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', step)
		print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
		print('Took: {0} seconds'.format(time.time() - start_time))
		return step

	# 单次的accuracy
	def eval_once(self, sess, init, writer, epoch, step):
		start_time = time.time()
		sess.run(init)
		self.training = False
		total_correct_preds = 0
		try:
			while True:
				accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
				writer.add_summary(summaries, global_step=step)
				total_correct_preds += accuracy_batch
		except tf.errors.OutOfRangeError:
			pass

		print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.n_test))
		print('Took: {0} seconds'.format(time.time() - start_time))

	# The train function alternates between training one epoch and evaluating
	def train(self, n_epochs):
		utils.safe_mkdir('checkpoints')
		utils.safe_mkdir('checkpoints/convnet_mnist')
		writer = tf.summary.FileWriter('./graphs/convnet', tf.get_default_graph())

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))

			# 如果已有数据，则读取以保存的数据
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)

			step = self.gstep.eval()

			for epoch in range(n_epochs):
				#update the weight
				self.assign_weight(sess, self._weight)

				step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
				#此处是否需要更新self._weight
				self.eval_once(sess, self.test_init, writer, epoch, step)
		writer.close()


if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=30)
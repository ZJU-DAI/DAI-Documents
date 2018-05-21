import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

INPUT_DATA = 'IMD'
OUTPUT_FILE = '0521_IMD_03.npy'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

def get_gt_label(sess, file_name, i, j, threshold):
	gt_file = 'gt'
	file_name = os.path.join(INPUT_DATA, gt_file, 'gt_'+file_name)
	image_raw_data = gfile.FastGFile(file_name, 'rb').read()
	image = tf.image.decode_png(image_raw_data)
	if image.dtype != tf.float32:
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.image.resize_images(image, [586, 586])
	#print(image.get_shape().as_list())
	image = tf.reshape(image, [586, 586])
	patch = tf.slice(image, [i * 8, j * 8], [64, 64])

	sum = tf.reduce_sum(patch)
	scale = sum / (255 * 64 * 64)
	label =tf.cond(scale > threshold, lambda: tf.constant(1), lambda: tf.constant(0))
	label_value = label
	label = tf.one_hot(label, depth=2, on_value=1)
	return sess.run([label, label_value])



def create_image_list(sess, testing_percentage, validation_percentage):

	training_images = []
	training_labels = []
	testing_images = []
	testing_labels = []
	validation_images = []
	validation_labels = []
	current_label = 0

	dir_name = 'AU'
	#获取一个子文件下的所有图片
	file_list = os.listdir(os.path.join(INPUT_DATA, dir_name))
	print(len(file_list))
	for file_name in file_list:
		print("get image:", file_name)
		# 读取图片，并将图片切为patch,64*64*3
		au_file_name = os.path.join(INPUT_DATA, dir_name, file_name)
		image_raw_data = gfile.FastGFile(au_file_name, 'rb').read()
		image = tf.image.decode_png(image_raw_data)
		if image.dtype != tf.float32:
			image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = tf.image.resize_images(image, [586, 586])

		size = [64, 64, 3]
		idx = 0
		for i in range(64):
			print("patch idx:", i)
			for j in range(64):
				#print("patch idx:", i, j)
				patch = tf.slice(image, [i * 8, j * 8, 0], size)
				patch_value = sess.run(patch)
				#print(patch.get_shape().as_list())

				label, label_value = get_gt_label(sess, file_name, i, j, threshold=tf.constant(0.0005))
				if label_value == 1:
					idx = idx+1
				chance = np.random.randint(100)
				if chance < validation_percentage:
					validation_images.append(patch_value)
					validation_labels.append(label)
				elif chance < (testing_percentage + validation_percentage):
					testing_images.append(patch_value)
					testing_labels.append(label)
				else:
					training_images.append(patch_value)
					training_labels.append(label)
			print(idx)

	#数据打乱顺序
	state = np.random.get_state()
	np.random.shuffle(training_images)
	np.random.set_state(state)
	np.random.shuffle(training_labels)

	return np.asarray([training_images, training_labels, validation_images, validation_labels,
	                   testing_images, testing_labels])

def main():
	with tf.Session() as sess:
		processed_data = create_image_list(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)

		np.save(OUTPUT_FILE, processed_data)

if __name__ == '__main__':
	main()
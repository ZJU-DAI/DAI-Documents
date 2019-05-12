import glob
import tensorflow as tf
import os.path
import numpy as np
from tensorflow.python.platform import gfile

INPUT_DATA_PATH = r'E:\Study\数据\flower_photos\flower_photos'
OUTPUT_FILE = r'E:\Study\数据\flower_photos\flower_photos\flower_photos.npy'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

def create_image_lists(sess,testing_percentage,validation_percentage):
	sub_dirs = [x[0] for x in os.walk(INPUT_DATA_PATH)]
	is_root_dir = True
	
	training_images = []
	training_labels = []
	testing_images = []
	testing_labels = []
	validation_images = []
	validation_labels = []
	
	current_label = 0
	
	for sub_dir in sub_dirs:
		if is_root_dir:
			is_root_dir = False
			continue
		print('sub_dir',sub_dir)
		#extensions = ['jpg','jpeg','JPG','JPEG']
		extensions = ['jpg','jpeg']
		file_list = []
		dir_name = os.path.basename(sub_dir)
		
		for extension in extensions:
			file_glob = os.path.join(INPUT_DATA_PATH,dir_name,'*.'+extension)
			print('file_glob',file_glob)
			file_list.extend(glob.glob(file_glob))
			if not file_list:
				continue
			
			print('file_list size:',len(file_list))
			for file_name in file_list:
				image_raw_data = gfile.FastGFile(file_name,'rb').read()
				image = tf.image.decode_jpeg(image_raw_data)
				if image.dtype != tf.float32:
					image = tf.image.convert_image_dtype(image,dtype=tf.float32)
				image = tf.image.resize_images(image,[224,224])
				image_value = sess.run(image)

				chance = np.random.randint(100)
				if chance < validation_percentage:
					validation_images.append(image_value)
					validation_labels.append(current_label)
				elif chance < (testing_percentage + validation_percentage):
					testing_images.append(image_value)
					testing_labels.append(current_label)
				else:
					training_images.append(image_value)
					training_labels.append(current_label)
			current_label += 1
			file_list = []
	print('training_images size:',len(training_images))
	print('validation_images size:',len(validation_images))
	print('testing_images size:',len(testing_images))
	return np.asarray([training_images,training_labels,validation_images,validation_labels,testing_images,testing_labels])
	
def main():
	with tf.Session() as sess:
		processed_data = create_image_lists(sess,TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
		print('process_data size',len(processed_data))
		np.save(OUTPUT_FILE,processed_data)

if __name__ == '__main__':
	main()

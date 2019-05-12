import os

import numpy as np


def safe_mkdir(path):
	""" Create a directory if there isn't one already. """
	try:
		os.mkdir(path)
	except OSError:
		pass


def get_label(training_labels, threshold, num_f):
	labels = []
	
	for i in range(len(training_labels)):
		gt = training_labels[i]
		white_num = np.count_nonzero(gt)
		scale = white_num / (gt.shape[0] * gt.shape[1])
		# print(white_num, patch_size*patch_size, scale)
		if scale > threshold:
			label = 1
			num_f += 1
		else:
			label = 0
		labels.append(label)
	return np.asarray(labels), num_f
import os
import numpy as np

def safe_mkdir(path):
	""" Create a directory if there isn't one already. """
	try:
		os.mkdir(path)
	except OSError:
		pass


def au_cut_to_patches(image, patch_size, strides, img_size, img_channels):
	out = []
	patch_num = int((img_size - patch_size) / strides + 1)
	for i in range(patch_num):
		for j in range(patch_num):
			out.append(
				image[i * strides:i * strides + patch_size, j * strides:j * strides + patch_size, 0:img_channels])
	return np.asarray(out)


def gt_cut_to_patches(image, patch_size, strides, img_size, threshold, m_idx):
	labels = []
	out = []
	patch_num = int((img_size - patch_size) / strides + 1)
	for i in range(patch_num):
		for j in range(patch_num):
			gt_patch = image[i * strides:i * strides + patch_size, j * strides:j * strides + patch_size]
			gt_patch /= 255
			white_num = np.count_nonzero(gt_patch)
			scale = white_num / (patch_size * patch_size)
			#print(white_num, patch_size*patch_size, scale)
			if scale > threshold:
				label = 1
				m_idx += 1
			else:
				label = 0
			labels.append(label)
			out.append(gt_patch)
	return np.asarray(out), np.asarray(labels), m_idx


def get_patch(img, gt_img, PATCH_SIZE, STRIDE, IMG_SIZE, INCHANNEL, THRESHOLD, m_idx):
	patches = au_cut_to_patches(img, PATCH_SIZE, STRIDE, IMG_SIZE, INCHANNEL)
	# print('patches shape： ', patches.shape)
	pixel_labels, labels, m_idx = gt_cut_to_patches(gt_img, PATCH_SIZE, STRIDE, IMG_SIZE, THRESHOLD, m_idx)
	# print('labels shape： ', labels.shape)
	
	return patches, labels, pixel_labels, m_idx

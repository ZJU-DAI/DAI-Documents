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


def gt_cut_to_patches(image, patch_size, strides, img_size, threshold):
	labels = []
	patch_num = int((img_size - patch_size) / strides + 1)
	for i in range(patch_num):
		for j in range(patch_num):
			gt_patch = image[i * strides:i * strides + patch_size, j * strides:j * strides + patch_size]
			sum = gt_patch.sum()
			scale = sum / (255 * patch_size * patch_size)
			if scale > threshold:
				label = 1
			else:
				label = 0
			labels.append(label)
	return np.asarray(labels)



def get_patch(img, gt_img, PATCH_SIZE, STRIDE, IMG_SIZE, INCHANNEL, THRESHOLD):
	patches = au_cut_to_patches(img, PATCH_SIZE, STRIDE, IMG_SIZE, INCHANNEL)
	#print('patches shape： ', patches.shape)
	labels = gt_cut_to_patches(gt_img, PATCH_SIZE, STRIDE, IMG_SIZE, THRESHOLD)
	#print('labels shape： ', labels.shape)
	return patches, labels
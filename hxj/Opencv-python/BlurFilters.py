import cv2 as cv
import numpy as np


# 添加黑白噪声
def salt(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
        return img


img = cv.imread('1.jpg', 0)
result = salt(img, 100)
'''
dst = cv.blur(img, (5, 5))  # 低通滤波器  (5, 5)-->滤波器的大小
gb = cv.GaussianBlur(img, (5, 5), 1.5)  # 高斯滤波器
'''
median = cv.medianBlur(result, 5)  # 中值滤波不会处理最大和最小值， 所以就不会受到噪声的影响。

cv.imshow('salt', result)
cv.imshow('Median', median)
cv.waitKey(0)
cv.destroyAllWindows()

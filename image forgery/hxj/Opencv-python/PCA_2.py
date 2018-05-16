import cv2 as cv
from numpy import *
import numpy as np

img = cv.imread('1.jpg')
x = np.array(img)
x = x.flatten()  # flatten() 把矩阵压成一维
print(x)
def pca(x):
    """
    :param x: 矩阵x，
    :return:  投影矩阵、方差、均值
    """

    # 获取维度
    num_data, dim = x.shape

    # 数据中心化

    mean_x = x.mean(axis=0)
    x = x - mean_x

    if dim > num_data:
        m = dot(x, x.T)  # 协方差矩阵
        e, Ev = linalg.eigh(m)  # 特征值e和特征向量Ev
        tmp = dot(x.T, Ev).T
        v = tmp[::-1]  # 反转
        s = sqrt(e)[::-1]  # 逆转特征值的排序
        for i in range(v.shape[1]):
            v[:, i] /= s
    else:
        u, s, v = linalg.svd(x)
        v = v[:num_data]  # 仅返回前num_data维的数据
    return v, s, mean_x

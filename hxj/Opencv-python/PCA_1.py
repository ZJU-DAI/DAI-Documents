import numpy as np


def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat, n):
    # 求协方差矩阵
    newData, meanVal = zeroMean()
    covMat = np.cov(newData, rowvar=0)  # rowvar = 0 表示传入的数据一行代表一个样本， 若非0 说明传入的数据一列代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量，特征向量按列放，一列代表一个特征向量
    eigval_indice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigVal_indice = eigval_indice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigVal_indice]  # 最大的n个特征值对应的特征向量
    low_datamat = newData * n_eigVect  # 低纬特征空间的数据
    reconmat = (low_datamat * n_eigVect.T) + meanVal  # 重构数据
    return low_datamat, reconmat

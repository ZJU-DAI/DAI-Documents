import math
import copy
import numpy as np


# 指定k个高斯分布参数，这里指定k=2。注意2个高斯分布具有相同均方差Sigma，分别为Mu1,Mu2。

def ini_data(Sigma, Mu1, Mu2, k, N):
    global X  # X产生的数据 ,k维向量

    global Mu  # 初始均值

    global Expectations

    X = np.zeros((1, N))

    Mu = np.random.random(2)  # 随机产生一个初始均值。

    Expectations = np.zeros((N, k))  # k个高斯分布，100个二维向量组成的矩阵。

    for i in range(0, N):

        if np.random.random(1) > 0.5:

            # 随机从均值为Mu1,Mu2的分布中取样。

            X[0, i] = np.random.normal() * Sigma + Mu1

        else:

            X[0, i] = np.random.normal() * Sigma + Mu2


# EM算法：步骤1，计算E[zij]

def e_step(Sigma, k, N):
    # 求期望。sigma协方差，k高斯混合模型数，N数据个数。

    global Expectations  # N个k维向量

    global Mu

    global X

    for i in range(0, N):
        Denom = 0

    for j in range(0, k):
        Denom += math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)

    # Denom  分母项  Mu(j)第j个高斯分布的均值。

    for j in range(0, k):
        Numer = math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)  # 分子项

    Expectations[i, j] = Numer / Denom  # 期望，计算出每一个高斯分布所占的期望，即该高斯分布以多大比例形成这个样本


# EM算法：步骤2，求最大化E[zij]的参数Mu

def m_step(k, N):
    # 最大化

    global Expectations  # 期望值

    global X  # 数据

    for j in range(0, k):
        # 遍历k个高斯混合模型数据

        Numer = 0  # 分子项

        Denom = 0  # 分母项

        for i in range(0, N):
            Numer += Expectations[i, j] * X[0, i]  # 每一个高斯分布的期望*该样本的值。

            Denom += Expectations[i, j]  # 第j个高斯分布的总期望值作为分母

            Mu[j] = Numer / Denom  # 第j个高斯分布新的均值，


# 算法迭代iter_num次，或达到精度Epsilon停止迭代

def run(Sigma, Mu1, Mu2, k, N, iter_num, Epsilon):
    ini_data(Sigma, Mu1, Mu2, k, N)

    for i in range(iter_num):

        Old_Mu = copy.deepcopy(Mu)  # 算法之前的MU

        e_step(Sigma, k, N)

        m_step(k, N)

        print(i, Mu)  # 经过EM算法之后的MU，

        if sum(abs(Mu - Old_Mu)) < Epsilon:
            break



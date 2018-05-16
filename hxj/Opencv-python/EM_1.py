import numpy as np
import scipy as sp
from scipy import stats

observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                         [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                         [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])


def em_single(priors, observations):
    """
    EM算法的单次迭代
    :param priors:
    :param observations:
    :return:
    """

    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
    theta_A = priors[0]
    theta_B = priors[1]

    # E step

    for observation in observations:
        len_observation = len(observation)
        num_heads = observation.sum()
        num_tails = len_observation - num_heads

        # 二项分布求解公式

        contribution_A = sp.stats.binom.pmf(num_heads, len_observation, theta_A)
        contribution_B = sp.stats.binom.pmf(num_heads, len_observation, theta_B)

        weight_A = contribution_A / (contribution_A + contribution_B)
        weight_B = contribution_B / (contribution_A + contribution_B)

        # 更新在当前参数下A，B硬币产生的正反面次数

        counts['A']['H'] += weight_A * num_heads
        counts['A']['T'] += weight_A * num_tails
        counts['B']['H'] += weight_B * num_heads
        counts['B']['T'] += weight_B * num_tails

    # M step

    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])

    return [new_theta_A, new_theta_B]


def em(observations, prior, tol=1e-6, iterations=10000):
    """
    EM算法
    :param boservation: 观测数据
    :param prior: 模型初值
    :param tol: 迭代结束阈值
    :param iterations: 最大迭代次数
    :return: 局部最优的模型参数
    """

    iteration = 0
    while iteration < iterations:
        new_prior = em_single(prior, observations)
        delta_chage = np.abs(prior[0] - new_prior[0])
        if delta_chage < tol:
            break
        else:
            prior = new_prior
            iteration += 1
    return [new_prior, iteration]


print(em(observations, [0.5, 0.6]))

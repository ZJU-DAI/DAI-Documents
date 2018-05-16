import cv2 as cv
import numpy as np

nb = 2

ns = 1

bayer = np.array([[0, 1], [1, 0]])


def prediction(im):
    hpred = [[0, 0.25, 0], [0.25, -1, 0.25], [0, 0.25, 0]]

    pred_error = cv.filter2D(im, hpred, )  # matlab 里面的imfilter() 第三个参数是replicate 图像大小通过外边界的值来填充扩展

    return pred_error


def getVarianceMap(im, bayer, dim):
    pattern = np.kron(np.ones((dim[0] / 2, dim[1] / 2)), bayer)

    mask = [[1, 0, 1, 0, 1, 0, 1]
            [0, 1, 0, 1, 0, 1, 0]
            [1, 0, 1, 0, 1, 0, 1]
            [0, 1, 0, 1, 0, 1, 0]
            [1, 0, 1, 0, 1, 0, 1]
            [0, 1, 0, 1, 0, 1, 0]
            [1, 0, 1, 0, 1, 0, 1]]

    win = gaussian_window() * mask

    mc = np.sum(np.sum(win))
    vc = 1 - np.sum(np.sum(win ** 2))
    win_mean = win / mc

    acquired = im * pattern
    mean_map_acquired = cv.filter2D(acquired, win_mean) * pattern
    sqmean_map_acquired = cv.filter2D(acquired ** 2, win_mean) * pattern
    var_map_acquired = (sqmean_map_acquired - (mean_map_acquired ** 2)) / vc

    interpolated = im * (1 - pattern)
    mean_map_interpolated = cv.filter2D(interpolated, win_mean) * (1 - pattern)
    sqmean_map_interpolated = cv.filter2D(interpolated ** 2, win_mean) * (1 - pattern)
    var_map_interpolated = (sqmean_map_interpolated - (mean_map_interpolated ** 2)) / vc

    var_map = var_map_acquired + var_map_interpolated

    return var_map


def gaussian_window():
    n_window = 7
    sigma = 1

    h = np.meshgrid(-(np.ceil(sigma * 2)), 4 * sigma / (n_window - 1), np.ceil(sigma * 2))

    win = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-0.5 * (h[0] ** 2 + h[1] ** 2) / sigma ** 2)

    return win


def getFeature(mmp, bayer, nb):
    pattern = np.kron(np.ones(nb / 2, nb / 2), bayer)

    func = np.prod()

    statistics = np.block(mmp, [nb, nb], func)  ########

    return statistics


def MoGEstimationZM(statistics):
    tol = 1e-3
    max_iter = 500

    '''
    statistics(isnan(statistics)) = 1
    '''
    data = np.log(statistics)
    '''
    data = data(not(isinf(data)|isnan(data)))
    '''
    em = EMGaussianZM(data, tol, max_iter)

    mu = [[em[2], 0]]

    sigma = np.sqrt([[em[1], em[3]]])

    mix_perc = [[1 - em[0], em[0]]]

    return mu, sigma, mix_perc


def EMGaussianZM(x, tol, max_iter):
    alpha = 0.5
    mu2 = np.mean(x)
    v2 = np.var(x)
    v1 = v2 / 10

    alpha_old = 1
    k = 1
    while np.abs(alpha - alpha_old) > tol & k < max_iter:
        alpha_old = alpha
        k = k + 1
        f1 = alpha * np.exp(-x ** 2 / 2 / v1) / np.sqrt(v1)
        f2 = (1 - alpha) * np.exp(-(x - mu2) ** 2 / 2 / v2) / np.sqrt(v2)
        alpha1 = f1 / (f1 + f2)
        alpha2 = f2 / (f1 + f2)

        alpha = np.mean(alpha1)
        v1 = np.sum(alpha1 * (x ** 2)) / np.sum(alpha1)
        mu2 = np.sum(alpha2 * x) / np.sum(alpha2)
        v2 = np.sum(alpha2 * ((x - mu2) ** 2)) / np.sum(alpha2)

    if np.abs(alpha - alpha_old) > tol:
        print('warning')

    return alpha, v1, mu2, v2


def loglikelihood(statistics, mu, sigma):
    min = 1e-320
    max = 1e304

    '''
    statistics(isnan(statistics)) = 1
    statistics(isinf(statistics))=max;
    statistics(statistics == 0) = min;
    '''
    mu1 = mu[1]
    mu2 = mu[0]

    sigma1 = sigma[1]
    sigma2 = sigma[0]
    L = np.log(sigma1) - np.log(sigma2) - 0.5 * (((np.log(statistics) - mu2) ** 2) / sigma2 ** 2) - (
            ((np.log(statistics) - mu1) ** 2) / sigma1 ** 2)

    return L


def func(x):
    y = np.sum(x)
    return y


def segmented_process(M, blk_size=(16, 16), fun=None):
    rows = []
    for i in range(0, M.shape[0], blk_size[0]):
        cols = []
        for j in range(0, M.shape[1], blk_size[1]):
            cols.append(fun(M[i:i + blk_size[0], j:j + blk_size[1]]))
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


def getMap(L, ns, nm):
    log_L_cum = segmented_process(L, [ns, ns], func(L))

    mp = cv.medianBlur(L, [ns, ns], 'symmetric')

    return mp


def CFAloc(img, bayer, nb, ns):
    nm = 5
    im = img[:, :, 2]
    dim = im.shape

    pred_error = prediction(im)

    var_map = getVarianceMap(pred_error, bayer, dim)

    stat = getFeature(var_map, bayer, nb)

    g = MoGEstimationZM(stat)

    loglikelihood_map = loglikelihood(stat, g[0], g[1])

    mp = getMap(loglikelihood_map, ns, nm)

    return mp, stat


img = cv.imread('1.jpg')

mp, star = CFAloc(img, bayer, nb, ns)

h, w = img.shape()

print(h, w)

import numpy as np
import cv2 as cv


def access_pixel(image):
    """
    访问图像所有像素
    :param image:
    :return:
    """
    print(image.shape)

    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print('height: %s, width: %s, channels: %s' % (height, width, channels))

    # for row in range(height):
    #     for col in range(width):
    #         for c in range(channels):
    #             pv = image[row, col, c]
    #             image[row, col, c] = 255 - pv
    image = cv.bitwise_not(image)
    cv.imshow('demo', image)


def create_image():
    """
    创建新图像
    :return:
    """
    img = np.zeros([400, 400, 3], np.uint8)
    img[:, :, 0] = np.ones([400, 400]) * 255
    img[:, :, 2] = np.ones([400, 400]) * 255
    cv.imshow('new image', img)

    img = np.zeros([400, 400, 1], np.uint8)
    img = img * 127
    cv.imshow('new image', img)
    cv.imwrite('127img.png', img)

    img = np.ones([3, 3], np.uint8)
    img.fill(1000.22)
    print(img)
    # 变换为一维数组
    img = img.reshape([1, 9])
    print(img)


img = cv.imread('1.jpg')
# 获取cpu当前时钟总数
t1 = cv.getTickCount()
access_pixel(img)
t2 = cv.getTickCount()
# 计算处理像素花费的时间
time = ((t2 - t1) / cv.getTickFrequency())
print('time: %s s' % time)
create_image()
cv.waitKey()
cv.destroyAllWindows()

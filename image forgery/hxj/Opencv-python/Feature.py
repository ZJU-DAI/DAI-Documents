import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = cv.rectangle(img, (200, 0), (300, 403), (0, 255, 0), 2)
cv.imshow('local_pixel', img)


# 提取图片像素到矩阵
pixel_data = np.array(gray)
# 提取目标区域
box_data = pixel_data[:, 0:403]
# 矩阵行求和
pixel_sum = np.sum(box_data, axis=1)

x = range(403)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
plt.figure(1)
ax1.bar(x, pixel_sum)


plt.show()

key = cv.waitKey(0) & 0xff
if key == ord('q'):
    cv.destroyAllWindows()
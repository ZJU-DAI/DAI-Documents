import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('1.jpg')
cv.imshow('img', img)

hist = cv.calcHist([img], [0], None, [256], [0, 255])
plt.plot(hist)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
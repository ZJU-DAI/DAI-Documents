import cv2 as cv

img = cv.imread('1.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HLS)
cv.imshow('hsv', hsv)

yuv = cv.cvtColor(img, cv.COLOR_RGB2YUV)
cv.imshow('yuv', yuv)

ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
cv.imshow('ycrcb', ycrcb)

cv.waitKey(0)
cv.destroyAllWindows()
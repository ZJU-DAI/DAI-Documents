import cv2 as cv
import numpy as np

img = cv.imread('1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv.cornerHarris(gray, 2, 3, 0.01)
dst = cv.dilate(dst, None)

ret, dst = cv.threshold(dst, 0.01*dst.max(), 255, 0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

criteria = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001

corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

res = np.hstack((centroids, corners))
res = np.int0(res)


img[res[:, 1], res[:, 0]] = [0, 255, 0]
img[res[:, 3], res[:, 2]] = [0, 0, 255]

cv.imshow('dst', img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

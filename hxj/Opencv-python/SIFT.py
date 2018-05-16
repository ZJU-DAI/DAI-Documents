import cv2 as cv


img = cv.imread('1.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('origin', img)

detector = cv.xfeatures2d.SIFT_create()
keypoints = detector.detect(gray, None)
img = cv.drawKeypoints(gray, keypoints, img)

cv.imshow('test', img)
cv.waitKey(0)
cv.destroyAllWindows()
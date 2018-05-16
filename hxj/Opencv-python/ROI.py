import cv2 as cv


img = cv.imread('1.jpg')
img = cv.rectangle(img, (200, 0), (300, 403), (0, 255, 0), 2)

# b = img[0:300, 35:100]
#
# img[100:400, 100:165] = b


cv.imshow('original', img)
cv.waitKey(0)
cv.destroyAllWindows()
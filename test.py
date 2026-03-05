import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

cv.imshow('Image Display', img)

cv.waitKey()
cv.destroyAllWindows()

print(type(img))
print(img.shape)
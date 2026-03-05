import cv2 as cv
import sys

#np.hstack을 만들기 위해 numpy라이브러리 불러오기
import numpy as np

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

img_small=cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)
#--------------grayscale 변환--------------
gray = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)


cv.imwrite('soccer_gray.jpg', gray)
cv.imwrite('soccer_small.jpg', img_small)


#-------hstack을 위해 다시 3채널로 변환------
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

#-------원본 + grayscale 가로 결합-------
combined = np.hstack((img_small, gray_bgr))

cv.imshow('Combined Image', combined)

cv.waitKey()
cv.destroyAllWindows()

print(type(img))
print(img.shape)

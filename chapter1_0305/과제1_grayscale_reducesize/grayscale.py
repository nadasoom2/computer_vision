#opencv 불러오기
import cv2 as cv
#인터픠터 제어 sys라이브러리 불러오기
import sys
#np.hstack을 위해 numpy라이브러리 불러오기
import numpy as np

#이미지 불러오기
img = cv.imread('chapter1_0305/soccer.jpg')

#예외처리(이미지가 없다면 print문 출력)
if img is None:
    sys.exit('파일이 존재하지 않습니다.')

#기본 이미지 줄이기
img_small=cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)

#----------img_small을 grayscale로 변환-----------
gray = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)

#----------hstack을 위해 다시 3채널로 변환-----------
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

#-----------원본 + grayscale 가로 결합-----------
combined = np.hstack((img_small, gray_bgr))

# ------------최종 이미지 저장------------
cv.imwrite('Combined_Image.jpg', combined)
cv.imshow('Combined Image', combined)

#아무 키나 누르면 창이 닫히도록
cv.waitKey()
cv.destroyAllWindows()

print(type(img))
print(img.shape)

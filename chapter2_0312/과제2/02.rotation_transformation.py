import cv2
import numpy as np
import glob
import sys

# 이미지 불러오기
img = cv2.imread('chapter2_0312/rose.png')

# 예외처리(이미지가 없다면 print문 출력)
if img is None:
    sys.exit('파일이 존재하지 않습니다.')

# ------------------------------
# 1. 회전 변환 행렬 계산
# ------------------------------
# 이미지의 가로, 세로 크기 추출
h, w = img.shape[:2]

# 회전 중심점을 이미지 중앙으로 설정
center = (w / 2, h / 2)

# 회전 행렬 생성 (중심: 이미지 중앙, 각도: +30도, 크기: 0.8배)
M = cv2.getRotationMatrix2D(center, -30, 0.8)

# 평행이동 적용: 회전 행렬의 마지막 열에 tx, ty 값을 더함
# x축 방향 +80px, y축 방향 -40px 이동
M[0, 2] += 80
M[1, 2] += -40

# 회전 + 크기 조절 + 평행이동을 한 번에 적용
result = cv2.warpAffine(img, M, (w, h))

# 결과 이미지 저장
cv2.imwrite('rotation_transformation.jpg', result)
# 원본 이미지 출력
cv2.imshow('Original', img)
# 변환된 이미지 출력
cv2.imshow('Transformed', result)

cv2.waitKey()
cv2.destroyAllWindows()
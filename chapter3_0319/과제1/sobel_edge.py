# OpenCV 라이브러리 불러오기 (이미지 처리 핵심 모듈)
import cv2 as cv
# NumPy 라이브러리 불러오기 (수치 연산용)
import numpy as np
# Matplotlib 불러오기 (이미지 시각화용)
import matplotlib.pyplot as plt


# ── 1. 이미지 불러오기 ───

# 원본 이미지를 BGR 컬러로 읽어오기 (파일명을 실제 이미지 경로로 변경하세요)
img = cv.imread('chapter3_0319/edgeDetectionImage.jpg')

# 이미지 로드 실패 여부 확인
if img is None:
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")


# ── 2. 그레이스케일 변환 ───

# BGR 컬러 이미지를 그레이스케일(흑백)로 변환
# Sobel 필터는 단채널(흑백) 이미지에 적용해야 함
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# ── 3. Sobel 필터로 에지 검출 ─────

# X 방향 Sobel 필터 적용
# - 첫 번째 인자: 입력 이미지 (그레이스케일)
# - cv.CV_64F: 출력 데이터 타입 64비트 부동소수점 (음수 기울기 보존)
# - dx=1, dy=0: x 방향으로만 1차 미분하여 수직 에지 검출
# - ksize=3: 3×3 커널 사용 (5로 변경 시 더 넓은 에지 검출)
sobelX = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)

# Y 방향 Sobel 필터 적용
# - dx=0, dy=1: y 방향으로만 1차 미분하여 수평 에지 검출
# - 나머지 인자는 sobelX와 동일
sobelY = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)


# ── 4. 에지 강도(Magnitude) 계산 ───

# 에지 강도 = sqrt(sobelX² + sobelY²)
# x, y 방향 에지를 합쳐 전체 에지 강도를 구함
magnitude = cv.magnitude(sobelX, sobelY)


# ── 5. uint8 타입으로 변환 ───

# 부동소수점(CV_64F) 값을 0~255 범위의 uint8로 변환
# 절댓값을 취한 뒤 스케일링하여 음수·오버플로우 방지
magnitude_uint8 = cv.convertScaleAbs(magnitude)


# ── 6. 결과 이미지 저장 ──

# 에지 강도 이미지를 파일로 저장 (저장 경로·파일명은 필요에 따라 변경 가능)
cv.imwrite('sobel_result.jpg', magnitude_uint8)

# 저장 완료 메시지 출력
print("결과 이미지가 'sobel_result.jpg'로 저장되었습니다.")


# ── 7. 원본 이미지 BGR → RGB 변환 (Matplotlib 시각화용) ──

# OpenCV는 BGR 순서로 이미지를 읽지만 Matplotlib은 RGB 순서를 사용하므로 변환 필요
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)


# ── 8. 시각화 ───

# 전체 figure 크기 설정 (가로 12인치, 세로 5인치)
plt.figure(figsize=(12, 5))

# 1행 2열 배치에서 첫 번째(왼쪽) subplot 선택
plt.subplot(1, 2, 1)
# 원본 컬러 이미지 표시
plt.imshow(img_rgb)
# subplot 제목 설정
plt.title('Original Image')
# 축(눈금) 숨기기
plt.axis('off')

# 1행 2열 배치에서 두 번째(오른쪽) subplot 선택
plt.subplot(1, 2, 2)
# 에지 강도 이미지를 흑백(gray) 컬러맵으로 표시
plt.imshow(magnitude_uint8, cmap='gray')
# subplot 제목 설정
plt.title('Sobel Edge Detection')
# 축(눈금) 숨기기
plt.axis('off')

# subplot 간 간격을 자동으로 조정하여 겹침 방지
plt.tight_layout()
# 완성된 figure를 화면에 출력
plt.show()
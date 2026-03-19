# OpenCV 라이브러리 불러오기 (이미지 처리 핵심 모듈)
import cv2 as cv
# NumPy 라이브러리 불러오기 (수치 연산용)
import numpy as np
# Matplotlib 불러오기 (이미지 시각화용)
import matplotlib.pyplot as plt


# ── 1. 이미지 불러오기 ──────────────────────────────────────────────────────────

# 원본 이미지를 BGR 컬러로 읽어오기 (파일명을 실제 이미지 경로로 변경하세요)
img = cv.imread('chapter3_0319/coffee cup.JPG')

# 이미지 로드 실패 여부 확인
if img is None:
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")


# ── 2. GrabCut 초기 설정 ────────────────────────────────────────────────────────

# GrabCut 알고리즘이 사용할 마스크 초기화 (이미지와 동일한 크기, 값은 모두 0)
# 마스크 픽셀 값의 의미:
#   cv.GC_BGD(0)    → 확실한 배경
#   cv.GC_FGD(1)    → 확실한 전경(객체)
#   cv.GC_PR_BGD(2) → 배경으로 추정
#   cv.GC_PR_FGD(3) → 전경으로 추정
mask = np.zeros(img.shape[:2], np.uint8)

# 배경 모델 초기화 (GrabCut 내부 가우시안 혼합 모델용, 수정 금지)
bgdModel = np.zeros((1, 65), np.float64)

# 전경 모델 초기화 (GrabCut 내부 가우시안 혼합 모델용, 수정 금지)
fgdModel = np.zeros((1, 65), np.float64)


# ── 3. 관심 영역(사각형) 설정 ───────────────────────────────────────────────────

# 객체를 포함하는 초기 사각형 영역 설정 (x, y, width, height)
# x=200, y=100: 사각형 왼쪽 상단 꼭짓점 좌표 (이미지 중앙에 컵이 위치)
# width=880, height=760: 커피컵 전체가 포함되도록 조정된 크기
rect = (70, 50, 1100, 850)


# ── 4. GrabCut 알고리즘 실행 ────────────────────────────────────────────────────

# GrabCut으로 전경(객체)과 배경을 분리
# - img: 입력 원본 이미지
# - mask: 분할 결과가 저장될 마스크 (입출력 겸용)
# - rect: 초기 사각형 영역 (전경이 포함된 범위)
# - bgdModel, fgdModel: 내부 모델 (알고리즘이 자동으로 업데이트)
# - iterCount=5: 반복 횟수 (많을수록 정밀하지만 느려짐)
# - mode=cv.GC_INIT_WITH_RECT: 사각형 기반으로 초기화하여 실행
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

# 이미지의 세로(h), 가로(w) 크기 가져오기
h, w = img.shape[:2]

# 1차 실행에서 배경으로 오인된 중앙 커피 영역을 GC_FGD(1)로 강제 덮어쓰기
# h//3 ~ h*2//3: 세로 기준 이미지 중앙 1/3 범위
# w//3 ~ w*2//3: 가로 기준 이미지 중앙 1/3 범위
mask[h//3 : h*2//3, w//3 : w*2//3] = cv.GC_FGD

# 강제 지정된 마스크를 힌트로 삼아 GrabCut 재실행
# - mode=cv.GC_INIT_WITH_MASK: rect를 무시하고 현재 마스크 상태 기반으로 재분석
# - 커피 영역(GC_FGD)을 참고하여 주변 픽셀의 전경/배경을 더 정확하게 재판단
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_MASK)


# ── 5. 마스크 후처리 ────────────────────────────────────────────────────────────

# 확실한 전경(GC_FGD=1)과 전경으로 추정(GC_PR_FGD=3)인 픽셀을 1로,
# 나머지(배경, 배경 추정)는 0으로 변환하여 이진 마스크 생성
mask2 = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype(np.uint8)


# ── 6. 배경 제거 이미지 생성 ────────────────────────────────────────────────────

# 원본 이미지에 이진 마스크를 곱하여 배경 픽셀을 0(검정)으로 만들기
# mask2[:, :, np.newaxis]: (H, W) → (H, W, 1) 로 차원 확장하여 3채널 이미지에 브로드캐스팅 가능하게 함
result = img * mask2[:, :, np.newaxis]


# ── 7. 결과 이미지 저장 ─────────────────────────────────────────────────────────

# 배경이 제거된 결과 이미지를 파일로 저장
cv.imwrite('grabcut_result.jpg', result)

# 저장 완료 메시지 출력
print("결과 이미지가 'grabcut_result.jpg'로 저장되었습니다.")


# ── 8. 시각화를 위한 BGR → RGB 변환 ────────────────────────────────────────────

# OpenCV BGR 순서를 Matplotlib용 RGB 순서로 변환 (원본 이미지)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# OpenCV BGR 순서를 Matplotlib용 RGB 순서로 변환 (배경 제거 결과 이미지)
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)


# ── 9. 시각화 ───────────────────────────────────────────────────────────────────

# 전체 figure 크기 설정 (가로 18인치, 세로 5인치 / 이미지 3개 나란히 배치)
plt.figure(figsize=(18, 5))

# 1행 3열 배치에서 첫 번째(왼쪽) subplot 선택
plt.subplot(1, 3, 1)
# 원본 컬러 이미지 표시
plt.imshow(img_rgb)
# subplot 제목 설정
plt.title('Original Image')
# 축(눈금) 숨기기
plt.axis('off')

# 1행 3열 배치에서 두 번째(가운데) subplot 선택
plt.subplot(1, 3, 2)
# 이진 마스크 이미지를 흑백(gray) 컬러맵으로 표시
# 흰색(1): 전경(객체), 검정(0): 배경
plt.imshow(mask2, cmap='gray')
# subplot 제목 설정
plt.title('GrabCut Mask')
# 축(눈금) 숨기기
plt.axis('off')

# 1행 3열 배치에서 세 번째(오른쪽) subplot 선택
plt.subplot(1, 3, 3)
# 배경이 제거된 결과 이미지 표시
plt.imshow(result_rgb)
# subplot 제목 설정
plt.title('Background Removed')
# 축(눈금) 숨기기
plt.axis('off')

# subplot 간 간격을 자동으로 조정하여 겹침 방지
plt.tight_layout()
# 완성된 figure를 화면에 출력
plt.show()
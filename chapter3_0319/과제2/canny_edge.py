# OpenCV 라이브러리 불러오기 (이미지 처리 핵심 모듈)
import cv2 as cv
# NumPy 라이브러리 불러오기 (수치 연산용)
import numpy as np
# Matplotlib 불러오기 (이미지 시각화용)
import matplotlib.pyplot as plt


# ── 1. 이미지 불러오기 ──

# 원본 이미지를 BGR 컬러로 읽어오기 (파일명을 실제 이미지 경로로 변경하세요)
img = cv.imread('chapter3_0319/dabo.jpg')

# 이미지 로드 실패 여부 확인
if img is None:
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")


# ── 2. 그레이스케일 변환 ──

# BGR 컬러 이미지를 그레이스케일(흑백)로 변환
# 캐니 에지 검출은 단채널(흑백) 이미지에 적용해야 함
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# ── 3. 캐니(Canny) 에지 검출 ──

# 캐니 알고리즘으로 에지 맵 생성
# - 첫 번째 인자: 입력 이미지 (그레이스케일)
# - threshold1=100: 하위 임계값 (이 값 미만의 기울기는 에지에서 제외)
# - threshold2=200: 상위 임계값 (이 값 이상의 기울기는 확실한 에지로 판정)
edges = cv.Canny(gray, threshold1=100, threshold2=200)


# ── 4. 허프 변환(HoughLinesP)으로 직선 검출 ──

# 확률적 허프 변환으로 이미지에서 직선 성분 검출
# - edges: 에지 맵 이미지 (캐니 결과)
# - rho=1: 거리 해상도 (픽셀 단위, 1이면 1픽셀 간격으로 탐색)
# - theta=np.pi/180: 각도 해상도 (라디안 단위, 1도 간격으로 탐색)
# - threshold=100: 직선으로 인정하는 최소 투표 수 (값이 클수록 더 뚜렷한 직선만 검출)
# - minLineLength=50: 직선으로 인정하는 최소 길이(픽셀) (이보다 짧은 선분은 무시)
# - maxLineGap=10: 같은 직선으로 연결할 수 있는 최대 끊김 간격(픽셀)
lines = cv.HoughLinesP(edges,
                       rho=1,
                       theta=np.pi / 180,
                       threshold=200,
                       minLineLength=100,
                       maxLineGap=7)


# ── 5. 검출된 직선을 원본 이미지에 그리기 ──

# 원본 이미지를 복사하여 직선을 덮어 그릴 캔버스 준비 (원본 보존)
result = img.copy()

# 검출된 직선이 하나 이상 있을 경우에만 그리기 수행
if lines is not None:
    # 검출된 각 직선 정보를 순서대로 처리
    for line in lines:
        # 각 직선은 [x1, y1, x2, y2] 형태의 배열로 반환됨
        x1, y1, x2, y2 = line[0]
        # 두 점(x1,y1)~(x2,y2)를 잇는 직선을 빨간색(BGR: 0,0,255), 두께 2로 그리기
        cv.line(result, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)


# ── 6. 결과 이미지 저장 ──

# 직선이 그려진 결과 이미지를 파일로 저장
cv.imwrite('dabo_result.jpg', result)

# 저장 완료 메시지 출력
print("결과 이미지가 'dabo_result.jpg'로 저장되었습니다.")


# ── 7. 시각화를 위한 BGR → RGB 변환 ──

# OpenCV BGR 순서를 Matplotlib용 RGB 순서로 변환 (원본 이미지)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# OpenCV BGR 순서를 Matplotlib용 RGB 순서로 변환 (직선 검출 결과 이미지)
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)


# ── 8. 시각화 ──

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
# 직선이 표시된 결과 이미지 출력
plt.imshow(result_rgb)
# subplot 제목 설정
plt.title('Hough Line Detection')
# 축(눈금) 숨기기
plt.axis('off')

# subplot 간 간격을 자동으로 조정하여 겹침 방지
plt.tight_layout()
# 완성된 figure를 화면에 출력
plt.show()
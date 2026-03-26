# 필요한 라이브러리 임포트
import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 파일을 그레이스케일로 읽기 (SIFT는 그레이스케일 이미지에서 동작)
img_gray = cv.imread('chapter4_0326/mot_color70.jpg', cv.IMREAD_GRAYSCALE)

# 이미지 파일을 컬러로 읽기 (시각화용 원본 이미지)
img_color = cv.imread('chapter4_0326/mot_color70.jpg')

# OpenCV는 BGR 순서로 이미지를 읽으므로, matplotlib 출력을 위해 RGB로 변환
img_color_rgb = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)

# SIFT 객체 생성
# nfeatures=500 : 검출할 최대 특징점 수를 500개로 제한 (너무 많으면 줄일 수 있음)
# nOctaveLayers=3 : 각 옥타브에서 사용할 레이어 수 (기본값 3)
# contrastThreshold=0.04 : 낮은 대비 영역의 특징점 필터링 임계값 (높일수록 특징점 감소)
# edgeThreshold=10 : 엣지 형태의 특징점 필터링 임계값 (높일수록 특징점 증가)
# sigma=1.6 : 가우시안 블러의 시그마 값 (기본값 1.6)
sift = cv.SIFT_create(nfeatures=800, nOctaveLayers=3,
                      contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

# 그레이스케일 이미지에서 특징점(keypoints)과 기술자(descriptors) 검출
# keypoints : 특징점 위치, 크기, 방향 등의 정보를 담은 리스트
# descriptors : 각 특징점의 128차원 특징 벡터
keypoints, descriptors = sift.detectAndCompute(img_gray, None)

# 검출된 특징점 수 출력
print(f'검출된 특징점 수: {len(keypoints)}')

# 원본 컬러 이미지에 특징점 시각화
# img_color : 특징점을 그릴 원본 이미지
# keypoints : 검출된 특징점 리스트
# None : 출력 이미지 (None이면 새 이미지 생성)
# (0, 255, 0) : 특징점 색상 (초록색, BGR 기준)
# flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS :
#   특징점의 크기(원의 반지름)와 방향(선)까지 함께 시각화
img_keypoints = cv.drawKeypoints(
    img_color,
    keypoints,
    None,
    color=(0, 255, 0),
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# drawKeypoints 결과도 BGR이므로 matplotlib 출력을 위해 RGB로 변환
img_keypoints_rgb = cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB)

# matplotlib으로 원본 이미지와 특징점 시각화 이미지를 나란히 출력
# figsize=(14, 6) : 그림 전체 크기 설정
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 첫 번째 subplot에 원본 이미지 출력
axes[0].imshow(img_color_rgb)
# 첫 번째 subplot 제목 설정
axes[0].set_title('Original Image', fontsize=14)
# 첫 번째 subplot의 축 눈금 제거
axes[0].axis('off')

# 두 번째 subplot에 특징점이 시각화된 이미지 출력
axes[1].imshow(img_keypoints_rgb)
# 두 번째 subplot 제목에 검출된 특징점 수 포함하여 설정
axes[1].set_title(f'SIFT Keypoints (n={len(keypoints)})', fontsize=14)
# 두 번째 subplot의 축 눈금 제거
axes[1].axis('off')


# 전체 그림의 제목 설정
plt.suptitle('SIFT Feature Detection', fontsize=16, fontweight='bold')

# subplot 간격 자동 조정
plt.tight_layout()

# 이미지를 파일로 저장
plt.savefig('sift_result.png', dpi=150, bbox_inches='tight')

# 화면에 그래프 출력
plt.show()


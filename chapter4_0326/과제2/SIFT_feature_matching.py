import cv2 as cv
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. 이미지 로드
# ─────────────────────────────────────────

# 두 이미지를 그레이스케일로 로드 (SIFT 연산용)
img1_gray = cv.imread('chapter4_0326/mot_color70.jpg', cv.IMREAD_GRAYSCALE)
img2_gray = cv.imread('chapter4_0326/mot_color83.jpg', cv.IMREAD_GRAYSCALE)

# 두 이미지를 컬러로 로드 (시각화용)
img1_color = cv.imread('chapter4_0326/mot_color70.jpg')
img2_color = cv.imread('chapter4_0326/mot_color83.jpg')

# 파일 존재 여부 확인
if img1_gray is None or img2_gray is None:
    print('이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.')
    exit()


# ─────────────────────────────────────────
# 2. SIFT 특징점 추출
# ─────────────────────────────────────────

# SIFT 객체 생성 (nfeatures=0 → 특징점 수 제한 없음)
sift = cv.SIFT_create(nfeatures=300)

# 두 이미지에서 각각 특징점(keypoints)과 기술자(descriptors) 추출
# - keypoints  : 특징점의 위치, 크기, 방향 정보
# - descriptors: 각 특징점을 표현하는 128차원 벡터
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

print(f'img1 특징점 수: {len(kp1)}')
print(f'img2 특징점 수: {len(kp2)}')


# ─────────────────────────────────────────
# 3. 특징점 매칭 (FLANN 기반)
# ─────────────────────────────────────────

# FLANN 매칭을 위한 인덱스 파라미터 설정
# - algorithm=1 : KD-Tree 알고리즘 사용 (float 기술자에 적합)
# - trees=5     : 탐색에 사용할 트리 수 (높을수록 정확하지만 느림)
index_params = dict(algorithm=1, trees=5)

# 탐색 파라미터 설정
# - checks=50 : 탐색 시 확인할 후보 수 (높을수록 정확하지만 느림)
search_params = dict(checks=50)

# FLANN 기반 매처 생성
flann = cv.FlannBasedMatcher(index_params, search_params)

# knnMatch로 각 특징점에 대해 가장 유사한 상위 2개 매칭 결과 반환
matches = flann.knnMatch(des1, des2, k=2)


# ─────────────────────────────────────────
# 4. Lowe's Ratio Test로 좋은 매칭만 필터링
# ─────────────────────────────────────────

good_matches = []

for m, n in matches:
    # m : 가장 가까운 매칭 / n : 두 번째로 가까운 매칭
    # m의 거리가 n의 거리의 70% 미만일 때만 신뢰할 수 있는 매칭으로 간주
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f'필터링 후 좋은 매칭 수: {len(good_matches)}')


# ─────────────────────────────────────────
# 5. 매칭 결과 시각화
# ─────────────────────────────────────────

# 두 이미지를 가로로 이어 붙이고 매칭 선을 그려 시각화
img_matches = cv.drawMatches(
    img1_color, kp1,
    img2_color, kp2,
    good_matches, None,
    matchColor=(0, 255, 0),       # 매칭 선 색상 (초록)
    singlePointColor=(255, 0, 0), # 매칭 안 된 특징점 색상 (파랑)
    flags=2
)

# BGR → RGB 변환 (matplotlib 출력용)
img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

# 매칭 결과 이미지 크기 기준으로 figsize 자동 계산
h, w = img_matches_rgb.shape[:2]
dpi = 100

fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

ax.imshow(img_matches_rgb)
ax.set_title(f'SIFT Feature Matching  |  Good Matches: {len(good_matches)}', fontsize=13)
ax.axis('off')

plt.tight_layout()
plt.savefig('sift_matching_result.png', dpi=dpi, bbox_inches='tight')
plt.show()
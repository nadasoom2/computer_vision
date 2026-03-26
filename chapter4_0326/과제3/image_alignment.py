import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────
# 1. 이미지 로드
# ─────────────────────────────────────────

# 두 이미지를 그레이스케일로 로드 (SIFT 연산용)
img1_gray = cv.imread('chapter4_0326/img1.jpg', cv.IMREAD_GRAYSCALE)
img2_gray = cv.imread('chapter4_0326/img2.jpg', cv.IMREAD_GRAYSCALE)

# 두 이미지를 컬러로 로드 (시각화용)
img1_color = cv.imread('chapter4_0326/img1.jpg')
img2_color = cv.imread('chapter4_0326/img2.jpg')

# 파일 존재 여부 확인
if img1_gray is None or img2_gray is None:
    print('이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.')
    exit()


# ─────────────────────────────────────────
# 2. SIFT 특징점 추출
# ─────────────────────────────────────────

# SIFT 객체 생성
sift = cv.SIFT_create()

# 두 이미지에서 각각 특징점(keypoints)과 기술자(descriptors) 추출
# - keypoints  : 특징점의 위치, 크기, 방향 정보
# - descriptors: 각 특징점을 표현하는 128차원 벡터
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

print(f'img1 특징점 수: {len(kp1)}')
print(f'img2 특징점 수: {len(kp2)}')


# ─────────────────────────────────────────
# 3. BFMatcher로 특징점 매칭
# ─────────────────────────────────────────

# BFMatcher 생성
# - cv.NORM_L2  : SIFT 기술자 간 유클리드 거리로 유사도 측정
# - crossCheck=False : knnMatch 사용을 위해 False로 설정
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

# knnMatch로 각 특징점에 대해 가장 유사한 상위 2개 매칭 결과 반환
matches = bf.knnMatch(des1, des2, k=2)


# ─────────────────────────────────────────
# 4. Lowe's Ratio Test로 좋은 매칭만 필터링
# ─────────────────────────────────────────

good_matches = []

for m, n in matches:
    # m : 가장 가까운 매칭 / n : 두 번째로 가까운 매칭
    # m의 거리가 n의 거리의 70% 미만일 때만 신뢰할 수 있는 매칭으로 간주
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

good_matches = sorted(good_matches, key=lambda x: x.distance)
good_matches = good_matches[:50]  # 원하는 개수로 변경

print(f'필터링 후 좋은 매칭 수: {len(good_matches)}')

# 호모그래피 계산을 위해 최소 4개 이상의 매칭점 필요
if len(good_matches) < 4:
    print('매칭점이 너무 적습니다. 이미지를 확인하세요.')
    exit()


# ─────────────────────────────────────────
# 5. 호모그래피 계산  ← 수정된 부분
# ─────────────────────────────────────────

src_pts = np.float32([kp2[m.trainIdx].pt  for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.queryIdx].pt  for m in good_matches]).reshape(-1, 1, 2)

# img2 → img1 방향으로 호모그래피 계산
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

inlier_count = int(mask.sum())
print(f'RANSAC 인라이어 매칭 수: {inlier_count}')


# ─────────────────────────────────────────
# 6. 원근 변환 (Warping)  ← 수정된 부분
# ─────────────────────────────────────────

h1, w1 = img1_color.shape[:2]
h2, w2 = img2_color.shape[:2]

out_w = w1 + w2
out_h = max(h1, h2)

# img2를 H로 변환 (img1 기준으로 정렬)
warped_img2 = cv.warpPerspective(img2_color, H, (out_w, out_h))

# warped_img2 위에 img1을 덮어 씌워 정렬 결과 완성
result = warped_img2.copy()
result[0:h1, 0:w1] = img1_color


# ─────────────────────────────────────────
# 7. 매칭 결과 시각화
# ─────────────────────────────────────────

# RANSAC 인라이어 매칭만 시각화하기 위해 마스크 적용
matches_mask = mask.ravel().tolist()

# 인라이어 매칭선만 그려서 시각화
img_matches = cv.drawMatches(
    img1_color, kp1,
    img2_color, kp2,
    good_matches, None,
    matchColor=(0, 255, 0),         # 인라이어 매칭선 색상 (초록)
    singlePointColor=(255, 0, 0),   # 매칭 안 된 특징점 색상 (파랑)
    matchesMask=matches_mask,       # RANSAC 인라이어만 표시
    flags=2
)


# ─────────────────────────────────────────
# 8. 결과 출력
# ─────────────────────────────────────────

# BGR → RGB 변환 (matplotlib 출력용)
result_rgb      = cv.cvtColor(result,      cv.COLOR_BGR2RGB)
img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

# 두 결과 이미지를 나란히 출력
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 왼쪽 : 변환된 이미지 (Warped + 정렬 결과)
axes[0].imshow(result_rgb)
axes[0].set_title('Warped Image (Homography Result)', fontsize=13)
axes[0].axis('off')

# 오른쪽 : 특징점 매칭 결과
axes[1].imshow(img_matches_rgb)
axes[1].set_title(f'Matching Result  |  Inliers: {inlier_count}', fontsize=13)
axes[1].axis('off')

plt.suptitle('SIFT Homography', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('homography_result.png', dpi=100, bbox_inches='tight')
plt.show()
import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성 (없으면 자동 생성)
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("chapter2_0312/left.png")
right_color = cv2.imread("chapter2_0312/right.png")

# 이미지 파일이 없을 경우 예외 발생
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# 초점 거리 (픽셀 단위)
f = 700.0
# 베이스라인 (두 카메라 사이의 거리, 단위: m)
B = 0.12

# ROI 설정 (x, y, width, height)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환 (StereoBM은 그레이스케일 입력 필요)
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# StereoBM 객체 생성 (numDisparities: 탐색할 최대 disparity 범위, blockSize: 매칭 블록 크기)
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)

# 좌/우 그레이스케일 이미지로 disparity map 계산 (정수형, 16배 스케일)
disparity_raw = stereo.compute(left_gray, right_gray)

# 16배 스케일된 정수값을 실수형으로 변환 후 16으로 나눠 실제 disparity 값으로 복원
disparity = disparity_raw.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
# disparity > 0인 픽셀만 유효한 값으로 처리 (0 이하는 계산 불가)
valid_mask = disparity > 0

# depth map 배열 초기화 (유효하지 않은 픽셀은 0으로 유지)
depth_map = np.zeros_like(disparity, dtype=np.float32)

# 유효한 픽셀에 대해서만 Z = f * B / d 공식으로 depth 계산
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # ROI 영역의 disparity 값 추출
    roi_disp = disparity[y:y+h, x:x+w]
    # ROI 영역의 depth 값 추출
    roi_depth = depth_map[y:y+h, x:x+w]

    # ROI 내 유효한 픽셀만 필터링 (disparity > 0)
    valid_disp_vals = roi_disp[roi_disp > 0]
    valid_depth_vals = roi_depth[roi_depth > 0]

    # 유효한 픽셀이 있을 경우 평균 계산, 없으면 0으로 처리
    avg_disp = np.mean(valid_disp_vals) if len(valid_disp_vals) > 0 else 0
    avg_depth = np.mean(valid_depth_vals) if len(valid_depth_vals) > 0 else 0

    # 결과 딕셔너리에 저장
    results[name] = {"avg_disp": avg_disp, "avg_depth": avg_depth}

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("=" * 45)
print(f"{'ROI':<12} {'평균 Disparity':>15} {'평균 Depth(m)':>15}")
print("=" * 45)
for name, val in results.items():
    print(f"{name:<12} {val['avg_disp']:>15.2f} {val['avg_depth']:>15.4f}")
print("=" * 45)

# depth 기준으로 가장 가까운(작은) ROI와 가장 먼(큰) ROI 탐색
closest = min(results, key=lambda k: results[k]['avg_depth'] if results[k]['avg_depth'] > 0 else float('inf'))
farthest = max(results, key=lambda k: results[k]['avg_depth'])

# 가장 가깝고 먼 ROI 출력
print(f"\n가장 가까운 ROI: {closest} (depth={results[closest]['avg_depth']:.4f}m)")
print(f"가장 먼 ROI    : {farthest} (depth={results[farthest]['avg_depth']:.4f}m)")

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
# disparity 배열 복사 (원본 보존)
disp_tmp = disparity.copy()
# 유효하지 않은 픽셀(0 이하)을 NaN으로 대체
disp_tmp[disp_tmp <= 0] = np.nan

# 유효한 값이 하나도 없으면 오류 발생
if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

# 하위 5% 값을 최솟값으로 설정 (극단값 제거)
d_min = np.nanpercentile(disp_tmp, 5)
# 상위 95% 값을 최댓값으로 설정 (극단값 제거)
d_max = np.nanpercentile(disp_tmp, 95)

# 최솟값과 최댓값이 같으면 나눗셈 오류 방지를 위해 미세값 추가
if d_max <= d_min:
    d_max = d_min + 1e-6

# 0~1 범위로 정규화
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
# 0~1 범위를 벗어난 값 클리핑
disp_scaled = np.clip(disp_scaled, 0, 1)

# 시각화용 uint8 배열 초기화
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
# NaN이 아닌 유효한 픽셀 위치 마스크 생성
valid_disp = ~np.isnan(disp_tmp)
# 유효한 픽셀만 0~255 범위로 변환하여 저장
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# JET 컬러맵 적용 (높은 값=빨강=가까움, 낮은 값=파랑=멂)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
# 시각화용 uint8 배열 초기화
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

# 유효한 픽셀이 하나라도 있을 때만 시각화 수행
if np.any(valid_mask):
    # 유효한 depth 값만 추출
    depth_valid = depth_map[valid_mask]

    # 하위 5% 값을 최솟값으로 설정 (극단값 제거)
    z_min = np.percentile(depth_valid, 5)
    # 상위 95% 값을 최댓값으로 설정 (극단값 제거)
    z_max = np.percentile(depth_valid, 95)

    # 최솟값과 최댓값이 같으면 나눗셈 오류 방지를 위해 미세값 추가
    if z_max <= z_min:
        z_max = z_min + 1e-6

    # 0~1 범위로 정규화
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    # 0~1 범위를 벗어난 값 클리핑
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 값이 클수록 멀기 때문에 반전 (가까울수록 1 → 빨강)
    depth_scaled = 1.0 - depth_scaled
    # 유효한 픽셀만 0~255 범위로 변환하여 저장
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

# JET 컬러맵 적용 (높은 값=빨강=가까움, 낮은 값=파랑=멂)
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
# 원본 이미지를 복사하여 시각화용 이미지 생성 (원본 보존)
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    # 좌측 이미지에 ROI 영역을 초록색 사각형으로 표시
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 좌측 이미지에 ROI 이름을 사각형 위에 텍스트로 표시
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 우측 이미지에 ROI 영역을 초록색 사각형으로 표시
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 우측 이미지에 ROI 이름을 사각형 위에 텍스트로 표시
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
# ROI가 표시된 좌/우 이미지 저장
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)
cv2.imwrite(str(output_dir / "right_roi.png"), right_vis)
# disparity 컬러맵 이미지 저장
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)
# depth 컬러맵 이미지 저장
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)

# -----------------------------
# 9. 출력
# -----------------------------
# ROI가 표시된 좌/우 이미지 출력
cv2.imshow("Left with ROI", left_vis)
cv2.imshow("Right with ROI", right_vis)
# disparity 컬러맵 출력
cv2.imshow("Disparity Map", disparity_color)
# depth 컬러맵 출력
cv2.imshow("Depth Map", depth_color)

# 아무 키나 누르면 모든 창 종료
cv2.waitKey(0)
cv2.destroyAllWindows()
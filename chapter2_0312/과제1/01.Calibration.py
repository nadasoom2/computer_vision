import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수 (가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 크기 (단위: mm)
square_size = 25.0

# 코너 정밀화 종료 조건 (최대 30회 반복 or 오차 0.001 이하)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D 실제 좌표 배열 초기화 (z=0 평면 가정)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
# x, y 좌표를 격자 구조로 채우고 실제 크기(mm) 적용
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 여러 이미지에서 수집한 3D 실제 좌표 리스트
objpoints = []
# 여러 이미지에서 수집한 2D 이미지 좌표 리스트
imgpoints = []

# 캘리브레이션에 사용할 이미지 파일 목록 불러오기
images = glob.glob("chapter2_0312/calibration_images/left*.jpg")

# 이미지 크기 저장 변수 초기화
img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    # 이미지 파일 읽기
    img = cv2.imread(fname)
    # 코너 검출을 위해 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 그레이스케일 이미지에서 체크보드 코너 검출
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너 검출에 성공한 경우에만 처리
    if ret:
        # 해당 이미지의 3D 실제 좌표 추가
        objpoints.append(objp)

        # 검출된 코너를 서브픽셀 단위로 정밀화 (정확도 향상)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 정밀화된 2D 이미지 좌표 추가
        imgpoints.append(corners_refined)

        # 이미지 크기 저장 (width, height) → 캘리브레이션에 사용
        img_size = (gray.shape[1], gray.shape[0])

# 캘리브레이션에 사용된 이미지 수 출력
print(f"\n총 {len(objpoints)}장 이미지 사용됨\n")

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 3D-2D 좌표 대응 관계로부터 카메라 파라미터 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,   # 3D 실제 좌표
    imgpoints,   # 2D 이미지 좌표
    img_size,    # 이미지 크기 (width, height)
    None,        # cameraMatrix (None → 자동 초기화)
    None         # distCoeffs  (None → 자동 초기화)
)

# 카메라 내부 파라미터 행렬 출력 (fx, fy: 초점거리 / cx, cy: 주점)
print("Camera Matrix K:")
print(K)

# 왜곡 계수 출력
print("\nDistortion Coefficients:")
print(dist)

# 각 왜곡 계수를 항목별로 구분하여 출력
print(f"  k1={dist[0][0]:.6f}, k2={dist[0][1]:.6f}, "
      f"p1={dist[0][2]:.6f}, p2={dist[0][3]:.6f}, k3={dist[0][4]:.6f}")

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
# 첫 번째 이미지를 대표 샘플로 읽기
sample_img = cv2.imread(images[0])

# 카메라 행렬과 왜곡 계수를 이용해 왜곡 보정 수행
undistorted = cv2.undistort(sample_img, K, dist, None, K)

# 왜곡 보정된 이미지를 파일로 저장
cv2.imwrite('Distortion_Correction.jpg', undistorted)
# 원본 이미지 창 출력
cv2.imshow("Original", sample_img)
# 왜곡 보정된 이미지 창 출력
cv2.imshow("Distortion Correction", undistorted)
# 키 입력 대기 (아무 키나 누르면 종료)
cv2.waitKey(0)
# 모든 창 닫기
cv2.destroyAllWindows()
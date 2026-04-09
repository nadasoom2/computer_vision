import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# 얼굴 랜드마크 검출에 필요한 모델 파일 경로 설정
# (MediaPipe C 라이브러리가 한글 경로를 지원하지 않으므로 chapter6_0409 폴더에 저장)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face_landmarker.task")

# 모델 파일이 없을 경우 Google 공식 저장소에서 자동 다운로드
if not os.path.exists(MODEL_PATH):
    print("모델 파일을 다운로드 중입니다...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("다운로드 완료.")

# 모델 파일 경로를 기반으로 기본 옵션 설정
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

# FaceLandmarker 옵션 설정
# - num_faces=1 : 최대 검출 얼굴 수
# - min_face_detection_confidence=0.5 : 얼굴 검출 최소 신뢰도
# - min_face_presence_confidence=0.5  : 얼굴 존재 최소 신뢰도
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5
)

# FaceLandmarker 검출기 초기화
face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

# 이미지 파일 경로 설정 (chapter6_0409 폴더 내 namyoung.png 사용)
image_path = "chapter6_0409/namyoung.png"

# MediaPipe Image 형식으로 이미지 로드
mp_image = mp.Image.create_from_file(image_path)

# OpenCV로도 이미지 읽기 (랜드마크 시각화 및 화면 출력에 사용)
image = cv2.imread(image_path)

# 이미지 로드 실패 시 오류 메시지 출력 후 종료
if image is None:
    print(f"이미지를 불러올 수 없습니다: {image_path}")
    exit()

# 이미지의 높이와 너비 추출 (정규화 좌표 → 픽셀 좌표 변환에 사용)
img_h, img_w = image.shape[:2]

# FaceLandmarker로 얼굴 랜드마크 검출 수행
results = face_landmarker.detect(mp_image)

# 검출된 얼굴이 있을 경우 랜드마크 시각화 진행
if results.face_landmarks:

    # 검출된 각 얼굴에 대해 반복 (num_faces=1 이므로 최대 1회)
    for face_landmarks in results.face_landmarks:

        # 468개의 각 랜드마크에 대해 반복
        for landmark in face_landmarks:

            # 정규화된 랜드마크 x 좌표를 실제 픽셀 좌표로 변환
            x = int(landmark.x * img_w)

            # 정규화된 랜드마크 y 좌표를 실제 픽셀 좌표로 변환
            y = int(landmark.y * img_h)

            # 해당 랜드마크 위치에 초록색 점(반지름 1) 그리기
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# 결과 이미지 크기 조정
image = cv2.resize(image, None, fx=1.5, fy=1.5)
# 결과 이미지를 화면에 출력
cv2.imshow("Face Landmark Detection", image)

# 키 입력 대기 루프 (ESC 키(27)를 누르면 종료)
while True:
    # 키 입력 대기 (1ms 간격)
    key = cv2.waitKey(1) & 0xFF

    # ESC 키 입력 시 루프 종료
    if key == 27:
        break

# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()

# FaceLandmarker 리소스 해제
face_landmarker.close()

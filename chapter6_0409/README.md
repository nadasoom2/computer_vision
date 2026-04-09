# 과제1

## 1. 과제(문제)설명
이 과제는 YOLOv3와 SORT(Simple Online and Realtime Tracking) 알고리즘을 이용해 영상 속 여러 객체를 동시에 검출하고 추적하는 프로그램을 구현하는 것이다.

영상의 각 프레임에서 객체를 검출한 뒤, 칼만 필터로 객체의 다음 위치를 예측하고, 헝가리안 알고리즘으로 검출 결과와 기존 추적기를 매칭하여 동일한 객체에 일관된 ID를 부여한다.

## 2. 주요기능
- YOLOv3 기반 객체 검출
- NMS(Non-Maximum Suppression)를 이용한 중복 박스 제거
- 칼만 필터를 이용한 객체 위치 예측
- 헝가리안 알고리즘을 이용한 검출 결과와 추적기 매칭
- 객체별 고유 ID 부여 및 색상 구분 시각화
- 영상 또는 카메라 입력에 대한 실시간 다중 객체 추적

## 3. 핵심코드설명

### 3-1. IoU 계산과 바운딩 박스 변환
검출 박스와 추적 박스를 비교하기 위해 IoU를 계산하고, 칼만 필터 계산에 맞게 좌표 형식을 변환한다.

```python
def xywh_to_xyxy(bbox):
    """[cx, cy, w, h] → [x1, y1, x2, y2] 변환"""
    cx, cy, w, h = bbox
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)

def xyxy_to_xywh(bbox):
    """[x1, y1, x2, y2] → [cx, cy, w, h] 변환"""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)

def iou(bb_test, bb_gt):
    """두 바운딩 박스 간 IoU(Intersection over Union) 계산"""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)

    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union = area_test + area_gt - intersection

    return intersection / union if union > 0 else 0.0
```

### 3-2. 칼만 필터 기반 추적기
`KalmanBoxTracker` 클래스는 하나의 객체를 담당하며, 상태 벡터를 이용해 위치와 속도를 함께 추정한다. 검출이 잠시 끊겨도 이전 상태를 바탕으로 다음 위치를 예측할 수 있다.

```python
class KalmanBoxTracker:
    """
    단일 객체를 칼만 필터로 추적하는 클래스.
    상태 벡터: [cx, cy, w, h, vx, vy, vw, vh]
    관측 벡터: [cx, cy, w, h]
    """

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return xywh_to_xyxy(self.kf.x[:4].flatten())

    def update(self, bbox_xyxy):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(xyxy_to_xywh(bbox_xyxy).reshape(4, 1))
```

### 3-3. SORT 추적과 데이터 연관
`Sort` 클래스는 각 프레임마다 추적기의 예측값과 YOLO 검출 결과를 비교한 뒤, IoU가 높은 쌍을 헝가리안 알고리즘으로 매칭한다. 매칭되지 않은 검출은 새 추적기로 생성되고, 오래 관측되지 않은 추적기는 제거된다.

```python
def update(self, detections):
    self.frame_count += 1

    for trk in self.trackers:
        trk.predict()

    matched, unmatched_dets, _ = self._match_detections(detections)

    for t_idx, d_idx in matched:
        self.trackers[t_idx].update(detections[d_idx])

    for d_idx in unmatched_dets:
        new_trk = KalmanBoxTracker(detections[d_idx])
        self.trackers.append(new_trk)
```

### 3-4. YOLOv3 검출과 시각화
`YOLOv3Detector` 클래스는 OpenCV DNN을 사용해 프레임에서 객체를 검출한다. 이후 `draw_tracks()` 함수가 추적 박스와 ID를 화면에 표시한다.

```python
def draw_tracks(frame, tracks):
    """추적 결과를 프레임에 그립니다: 바운딩 박스 + ID 레이블"""
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        color = id_to_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        label = f"ID: {track_id}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), color, -1)
        cv2.putText(frame, label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame
```

## 4. 실행방법
1. 다음 파일들이 같은 폴더 또는 코드에서 지정한 경로에 있어야 한다.
   - `multi_object_tracking.py`
   - `slow_traffic_small.mp4`
   - `yolov3.cfg`
   - `yolov3.weights`

2. 필요한 라이브러리를 설치한다.

```python
pip install opencv-python numpy scipy filterpy
```

3. `chapter6_0409/과제1/multi_object_tracking.py`를 실행한다.

```python
python multi_object_tracking.py
```

4. 실행 후 영상 창이 열리며, 객체의 추적 박스와 ID가 표시된다.
5. 종료하려면 `q` 키를 누른다.

## 5. 실험결과
아래는 실행 결과를 캡처한 이미지이다.

![실험 결과 1](스크린샷%202026-04-09%20152356.png)

---

# 과제2

## 1. 과제(문제)설명
이 과제는 MediaPipe Face Landmarker를 이용해 이미지 속 얼굴의 468개 랜드마크를 검출하고, 검출된 랜드마크를 원본 이미지 위에 시각화하는 프로그램을 구현하는 것이다.

입력 이미지에서 얼굴을 찾은 뒤, 각 랜드마크의 정규화 좌표를 픽셀 좌표로 변환하여 점으로 표시한다. 모델 파일이 없을 경우에는 공식 배포 모델을 자동으로 내려받아 실행할 수 있도록 구성했다.

## 2. 주요기능
- MediaPipe Face Landmarker 기반 얼굴 랜드마크 검출
- 최대 1개의 얼굴에 대해 468개 랜드마크 추출
- 정규화 좌표를 픽셀 좌표로 변환 후 시각화
- 검출 결과를 OpenCV 창에 출력
- 모델 파일이 없을 때 자동 다운로드 처리

## 3. 핵심코드설명

### 3-1. 모델 경로 설정 및 자동 다운로드
모델 파일이 없으면 공식 저장소에서 `face_landmarker.task`를 내려받아 사용한다. 이 방식으로 실행 환경에 모델이 없어도 바로 동작할 수 있다.

```python
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("모델 파일을 다운로드 중입니다...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("다운로드 완료.")
```

### 3-2. FaceLandmarker 옵션 설정
얼굴 검출 수와 신뢰도 기준을 설정하고, 해당 옵션으로 FaceLandmarker 객체를 생성한다.

```python
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5
)

face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
```

### 3-3. 랜드마크 검출 및 좌표 변환
검출된 랜드마크의 좌표는 0~1 범위의 정규화 값이므로, 이미지 크기를 곱해 실제 픽셀 좌표로 바꾼 뒤 점을 찍는다.

```python
results = face_landmarker.detect(mp_image)

if results.face_landmarks:
    for face_landmarks in results.face_landmarks:
        for landmark in face_landmarks:
            x = int(landmark.x * img_w)
            y = int(landmark.y * img_h)
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
```

### 3-4. 결과 출력과 종료 처리
시각화된 이미지를 OpenCV 창에 출력하고, ESC 키를 누르면 프로그램이 종료되도록 구성했다.

```python
image = cv2.resize(image, None, fx=1.5, fy=1.5)
cv2.imshow("Face Landmark Detection", image)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
face_landmarker.close()
```

## 4. 실행방법
1. 다음 파일들이 같은 폴더에 있어야 한다.
   - `face_landmark_detection.py`
   - `namyoung.png`

2. 필요한 라이브러리를 설치한다.

```python
pip install opencv-python mediapipe
```

3. `chapter6_0409/과제2/face_landmark_detection.py`를 실행한다.

```python
python face_landmark_detection.py
```

4. 처음 실행 시 모델 파일이 없으면 자동으로 다운로드된다.
5. 실행 후 얼굴 랜드마크가 초록색 점으로 표시된 결과 이미지가 출력된다.
6. 종료하려면 `ESC` 키를 누른다.

## 5. 실험결과
아래는 실행 결과를 캡처한 이미지이다.

![실험 결과 2](과제2/스크린샷%202026-04-09%20154726.png)


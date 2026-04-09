"""
multi_tracking.py
SORT(Simple Online and Realtime Tracking) 알고리즘을 이용한 다중 객체 추적 프로그램
- YOLOv3로 객체를 검출하고, 칼만 필터 + 헝가리안 알고리즘으로 추적을 수행합니다.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# ─────────────────────────────────────────────
# 유틸 함수: 바운딩 박스 변환
# ─────────────────────────────────────────────

def xywh_to_xyxy(bbox):
    """[cx, cy, w, h] → [x1, y1, x2, y2] 변환"""
    cx, cy, w, h = bbox
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)

def xyxy_to_xywh(bbox):
    """[x1, y1, x2, y2] → [cx, cy, w, h] 변환"""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)

def iou(bb_test, bb_gt):
    """두 바운딩 박스 간 IoU(Intersection over Union) 계산 (xyxy 형식)"""
    # 교집합 영역의 좌상단·우하단 좌표 계산
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    # 교집합 너비·높이가 0 이하이면 겹침 없음
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)

    # IoU = 교집합 / 합집합
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union = area_test + area_gt - intersection

    return intersection / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# 개별 객체 추적기: 칼만 필터 기반
# ─────────────────────────────────────────────

class KalmanBoxTracker:
    """
    단일 객체를 칼만 필터로 추적하는 클래스.
    상태 벡터: [cx, cy, w, h, vx, vy, vw, vh]  (위치 + 속도)
    관측 벡터: [cx, cy, w, h]
    """

    # 전체 추적기에 걸쳐 고유 ID를 부여하기 위한 클래스 변수
    count = 0

    def __init__(self, bbox_xyxy):
        """바운딩 박스 [x1,y1,x2,y2]로 칼만 필터 초기화"""

        # ── 칼만 필터 설정 ──────────────────────────
        # dim_x=8: 상태 변수 수, dim_z=4: 관측 변수 수
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # 상태 전이 행렬 F: 등속 운동 모델 (위치 += 속도)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # 관측 행렬 H: 상태 벡터에서 위치(cx,cy,w,h)만 관측
        self.kf.H = np.eye(4, 8, dtype=float)

        # 관측 노이즈 공분산 행렬 R
        self.kf.R[2:, 2:] *= 10.0

        # 초기 추정 오차 공분산 행렬 P (속도 성분의 불확실성을 크게 설정)
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # 프로세스 노이즈 공분산 행렬 Q
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # 초기 상태를 바운딩 박스로 설정 (xyxy → xywh 변환)
        self.kf.x[:4] = xyxy_to_xywh(bbox_xyxy).reshape(4, 1)

        # ── 추적기 메타 정보 ─────────────────────────
        # 전역 고유 ID 부여
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # 마지막으로 관측된 프레임으로부터 경과한 프레임 수
        self.time_since_update = 0

        # 이 추적기가 등장한 총 프레임 수 (안정성 판단에 사용)
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        """칼만 필터 예측 단계: 다음 프레임의 위치를 추정하고 반환"""
        self.kf.predict()
        self.age += 1

        # 업데이트가 없으면 연속 감지 횟수 초기화
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # 예측된 상태를 xyxy 형식으로 변환하여 반환
        return xywh_to_xyxy(self.kf.x[:4].flatten())

    def update(self, bbox_xyxy):
        """칼만 필터 업데이트 단계: 실제 검출 결과로 상태를 보정"""
        self.time_since_update = 0
        self.hit_streak += 1

        # 검출된 바운딩 박스를 관측값으로 변환 후 필터 업데이트
        self.kf.update(xyxy_to_xywh(bbox_xyxy).reshape(4, 1))

    def get_state(self):
        """현재 추정된 바운딩 박스를 xyxy 형식으로 반환"""
        return xywh_to_xyxy(self.kf.x[:4].flatten())


# ─────────────────────────────────────────────
# SORT 추적기: 헝가리안 알고리즘으로 데이터 연관
# ─────────────────────────────────────────────

class Sort:
    """
    SORT(Simple Online and Realtime Tracking) 추적기.
    - 칼만 필터로 각 객체의 위치를 예측
    - 헝가리안 알고리즘으로 검출 결과와 기존 추적기를 매칭
    """

    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        """
        max_age      : 검출이 없어도 추적기를 유지할 최대 프레임 수
        min_hits     : 유효 추적으로 출력하기 위한 최소 연속 감지 횟수
        iou_threshold: 매칭에 사용할 IoU 임계값
        """
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold

        # 현재 살아있는 추적기 목록
        self.trackers = []
        # 처리한 총 프레임 수
        self.frame_count = 0

    def _match_detections(self, detections):
        """
        헝가리안 알고리즘으로 검출 박스와 기존 추적기를 매칭.
        반환: (matched, unmatched_dets, unmatched_trks)
        """
        # 기존 추적기가 없으면 모두 미매칭 검출로 처리
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []

        # IoU 비용 행렬 생성 (행=추적기, 열=검출)
        # predict()는 update()에서 이미 호출됨 → get_state()로 현재 예측 위치만 가져옴
        iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=float)
        for t, trk in enumerate(self.trackers):
            trk_box = trk.get_state()
            for d, det in enumerate(detections):
                iou_matrix[t, d] = iou(trk_box, det)

        # 헝가리안 알고리즘으로 최적 매칭 수행 (비용 최소화 → IoU 최대화)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched      = []
        unmatched_d  = list(range(len(detections)))
        unmatched_t  = list(range(len(self.trackers)))

        for r, c in zip(row_ind, col_ind):
            # IoU가 임계값 미만이면 매칭 실패로 처리
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                # 매칭된 항목을 미매칭 목록에서 제거
                if c in unmatched_d: unmatched_d.remove(c)
                if r in unmatched_t: unmatched_t.remove(r)

        return matched, unmatched_d, unmatched_t

    def update(self, detections):
        """
        한 프레임의 검출 결과를 받아 추적기를 업데이트하고
        확정된 추적 결과 [x1, y1, x2, y2, id]를 반환합니다.
        detections: Nx4 배열 [x1, y1, x2, y2]
        """
        self.frame_count += 1

        # ── 기존 추적기 예측 ───────────────────────────
        # 칼만 필터로 각 추적기의 다음 위치 예측
        for trk in self.trackers:
            trk.predict()

        # ── 데이터 연관 ───────────────────────────────
        # unmatched_trks는 time_since_update 증가로 자동 관리되므로 별도 처리 불필요
        matched, unmatched_dets, _ = self._match_detections(detections)

        # 매칭된 추적기 업데이트
        for t_idx, d_idx in matched:
            self.trackers[t_idx].update(detections[d_idx])

        # 새 검출에 대해 새로운 추적기 생성
        for d_idx in unmatched_dets:
            new_trk = KalmanBoxTracker(detections[d_idx])
            self.trackers.append(new_trk)

        # ── 결과 수집 및 오래된 추적기 제거 ───────────
        results  = []
        survivors = []

        for trk in self.trackers:
            # 충분히 연속으로 감지된 추적기만 출력 (초기 노이즈 필터링)
            if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                box = trk.get_state()
                results.append(np.append(box, trk.id + 1))  # ID는 1부터 시작

            # max_age 이내인 추적기만 유지
            if trk.time_since_update <= self.max_age:
                survivors.append(trk)

        self.trackers = survivors

        return np.array(results) if results else np.empty((0, 5))


# ─────────────────────────────────────────────
# YOLOv3 객체 검출기
# ─────────────────────────────────────────────

class YOLOv3Detector:
    """OpenCV DNN 모듈로 YOLOv3 모델을 로드하고 객체를 검출하는 클래스"""

    def __init__(self, weights_path, cfg_path,
                 conf_threshold=0.5, nms_threshold=0.4, input_size=416):
        """
        weights_path   : yolov3.weights 파일 경로
        cfg_path       : yolov3.cfg 파일 경로
        conf_threshold : 검출 신뢰도 임계값
        nms_threshold  : NMS(비최대 억제) 임계값
        input_size     : YOLO 입력 이미지 크기 (정사각형)
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold
        self.input_size     = input_size

        # YOLOv3 모델 로드 (Darknet 형식)
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

        # 추론에 GPU(CUDA) 사용 가능 시 활성화, 없으면 CPU 사용
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 출력 레이어 이름 가져오기 (YOLO의 3가지 스케일 출력)
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        self.output_layers = [layer_names[i - 1] for i in unconnected.flatten()]

    def detect(self, frame):
        """
        한 프레임에서 객체를 검출합니다.
        반환: Nx4 배열 [x1, y1, x2, y2] (픽셀 좌표)
        """
        H, W = frame.shape[:2]

        # 프레임을 YOLO 입력 형식(blob)으로 변환
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0,
            (self.input_size, self.input_size),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)

        # 순전파(Forward Pass)로 각 스케일의 출력 획득
        layer_outputs = self.net.forward(self.output_layers)

        boxes      = []  # [x, y, w, h] 목록
        confidences = []  # 신뢰도 목록
        class_ids  = []  # 클래스 ID 목록

        for output in layer_outputs:
            for detection in output:
                # 처음 4개 값은 박스 정보, 나머지는 클래스별 확률
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                # 신뢰도 임계값 이하의 검출은 무시
                if confidence < self.conf_threshold:
                    continue

                # 중심 좌표·너비·높이를 픽셀 단위로 변환
                cx = int(detection[0] * W)
                cy = int(detection[1] * H)
                w  = int(detection[2] * W)
                h  = int(detection[3] * H)

                boxes.append([cx - w // 2, cy - h // 2, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        # NMS(비최대 억제)로 중복 박스 제거
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )

        detections_xyxy = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # [x1, y1, x2, y2] 형식으로 변환
                detections_xyxy.append([x, y, x + w, y + h])

        return np.array(detections_xyxy, dtype=float) if detections_xyxy else np.empty((0, 4))


# ─────────────────────────────────────────────
# 시각화 유틸
# ─────────────────────────────────────────────

# 추적 ID별 색상을 일관되게 유지하기 위한 캐시 딕셔너리
_color_cache = {}

def id_to_color(track_id):
    """추적 ID에 대응하는 고유 BGR 색상 반환 (ID마다 동일한 색상 유지)"""
    if track_id not in _color_cache:
        # ID 값을 시드로 하여 재현 가능한 랜덤 색상 생성
        rng = np.random.default_rng(seed=int(track_id) * 37)
        _color_cache[track_id] = tuple(int(c) for c in rng.integers(80, 230, size=3))
    return _color_cache[track_id]

def draw_tracks(frame, tracks):
    """추적 결과를 프레임에 그립니다: 바운딩 박스 + ID 레이블"""
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        color = id_to_color(track_id)

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # ID 레이블 배경 사각형 (가독성을 위해)
        label     = f"ID: {track_id}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), color, -1)

        # ID 레이블 텍스트 출력
        cv2.putText(frame, label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def run(video_source=0,
        weights="yolov3.weights",
        cfg="yolov3.cfg",
        conf_thresh=0.5,
        nms_thresh=0.4,
        max_age=3,
        min_hits=3,
        iou_thresh=0.3):
    """
    다중 객체 추적 메인 루프.
    video_source : 카메라 인덱스(0,1,...) 또는 동영상 파일 경로
    """

    # ── 초기화 ─────────────────────────────────
    # YOLOv3 검출기 초기화
    detector = YOLOv3Detector(weights, cfg, conf_thresh, nms_thresh)

    # SORT 추적기 초기화
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_thresh)

    # 칼만 필터 추적기 ID 초기화 (영상 전환 시 ID 재사용 방지)
    KalmanBoxTracker.count = 0

    # 비디오 소스 열기
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] 비디오 소스를 열 수 없습니다: {video_source}")
        return

    print("[INFO] 추적 시작. 종료하려면 'q'를 누르세요.")

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 비디오 스트림이 종료되었습니다.")
            break

        # ── 객체 검출 ───────────────────────────
        # YOLOv3로 현재 프레임에서 객체 검출
        detections = detector.detect(frame)

        # ── 객체 추적 ───────────────────────────
        # SORT 추적기에 검출 결과를 전달하여 ID가 부여된 추적 결과 획득
        tracks = tracker.update(detections)

        # ── 결과 시각화 ─────────────────────────
        # 추적 박스와 ID를 프레임에 그리기
        frame = draw_tracks(frame, tracks)

        # 현재 추적 중인 객체 수를 화면 좌상단에 표시
        cv2.putText(frame, f"Tracking: {len(tracks)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 결과 프레임 출력
        cv2.imshow("SORT Multi-Object Tracking", frame)

        # 'q' 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 사용자가 종료했습니다.")
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# 진입점 — F5로 바로 실행 가능하도록 경로 고정
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # ── 파일 경로 설정 ────────────────────────────
    BASE_DIR = r"c:\smkim\chapter6_0409"

    # 추적할 동영상 파일 경로
    VIDEO_PATH   = os.path.join(BASE_DIR, "slow_traffic_small.mp4")

    # YOLOv3 모델 관련 파일 경로
    WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov3.weights")
    CFG_PATH     = os.path.join(BASE_DIR, "yolov3.cfg")

    # ── 추적 파라미터 설정 ────────────────────────
    CONF_THRESH = 0.5   # 객체 검출 신뢰도 임계값
    NMS_THRESH  = 0.4   # NMS 임계값
    MAX_AGE     = 3     # 검출 없어도 추적 유지할 최대 프레임 수
    MIN_HITS    = 3     # 추적 ID 확정까지 필요한 최소 연속 감지 횟수
    IOU_THRESH  = 0.3   # 매칭 IoU 임계값

    # 필수 파일 존재 여부 사전 확인 (오류 메시지를 명확하게 출력)
    for path, desc in [
        (VIDEO_PATH,   "동영상"),
        (WEIGHTS_PATH, "YOLOv3 가중치(.weights)"),
        (CFG_PATH,     "YOLOv3 설정(.cfg)"),
    ]:
        if not os.path.exists(path):
            print(f"[ERROR] {desc} 파일을 찾을 수 없습니다: {os.path.abspath(path)}")
            exit(1)

    # 다중 객체 추적 실행
    run(
        video_source=VIDEO_PATH,
        weights=WEIGHTS_PATH,
        cfg=CFG_PATH,
        conf_thresh=CONF_THRESH,
        nms_thresh=NMS_THRESH,
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        iou_thresh=IOU_THRESH,
    )

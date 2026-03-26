# 📌 과제 1 : SIFT 특징점 검출 및 시각화

## 1. 문제 정의
- 주어진 이미지(`mot_color70.jpg`)에서 SIFT 알고리즘을 사용하여 특징점을 검출하고 시각화
- 원본 이미지와 특징점이 표시된 이미지를 나란히 비교 출력

---

## 2. 주요 기능
- SIFT 객체 생성 및 파라미터 조절
- `detectAndCompute()`로 특징점 및 기술자 추출
- `drawKeypoints()`로 특징점의 위치 · 크기 · 방향 시각화
- 원본 이미지와 결과 이미지를 나란히 출력 (matplotlib)

---

## 3. 핵심 코드 설명

### SIFT 객체 생성
```python
sift = cv.SIFT_create(
    nfeatures=500,          # 검출할 최대 특징점 수 (줄이면 특징점 감소)
    nOctaveLayers=3,        # 각 옥타브의 레이어 수
    contrastThreshold=0.04, # 낮은 대비 영역 필터링 (높이면 특징점 감소)
    edgeThreshold=10,       # 엣지 형태 특징점 필터링
    sigma=1.6               # 가우시안 블러 시그마 값
)
```

### 특징점 검출
```python
# keypoints  : 특징점의 위치, 크기, 방향 정보
# descriptors: 각 특징점을 표현하는 128차원 벡터
keypoints, descriptors = sift.detectAndCompute(img_gray, None)
```

### 특징점 시각화
```python
img_keypoints = cv.drawKeypoints(
    img_color,
    keypoints,
    None,
    color=(0, 255, 0),
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # 크기 · 방향 함께 표시
)
```

### 이미지 실제 크기로 출력
```python
h, w = img_color.shape[:2]
dpi = 100
fig, axes = plt.subplots(1, 2, figsize=(w * 2 / dpi, h / dpi), dpi=dpi)
```

---

## 4. 실행 방법
- `mot_color70.jpg`를 스크립트와 **같은 폴더**에 위치시킴
- 스크립트 실행
```bash
python SIFT_feature_detection.py
```

---

## 5. 실행 결과

<img width="2085" height="774" alt="sift_detection_result" src="https://github.com/user-attachments/assets/5fe8766d-5856-46f5-9bc5-b6fd6eb53c25" />


---
---

# 📌 과제 2 : SIFT 특징점 매칭

## 1. 문제 정의
- 두 이미지(`mot_color70.jpg`, `mot_color80.jpg`) 간 SIFT 특징점을 매칭
- 신뢰도 높은 매칭점만 선별하여 매칭 결과 시각화

---

## 2. 주요 기능
- FLANN 기반 매처로 빠른 특징점 매칭
- `knnMatch()` + Lowe's Ratio Test로 정확한 매칭점 필터링
- `drawMatches()`로 두 이미지 간 매칭선 시각화

---

## 3. 핵심 코드 설명

### FLANN 매처 생성
```python
index_params  = dict(algorithm=1, trees=5)  # KD-Tree 알고리즘 사용
search_params = dict(checks=50)             # 탐색 후보 수
flann = cv.FlannBasedMatcher(index_params, search_params)
```

### knnMatch로 상위 2개 매칭 추출
```python
# k=2 : 각 특징점에 대해 가장 유사한 2개의 매칭 반환
matches = flann.knnMatch(des1, des2, k=2)
```

### Lowe's Ratio Test로 좋은 매칭 필터링
```python
good_matches = []
for m, n in matches:
    # 1등 매칭 거리가 2등의 70% 미만일 때만 신뢰할 수 있는 매칭으로 선별
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```

### 매칭 결과 시각화
```python
img_matches = cv.drawMatches(
    img1_color, kp1,
    img2_color, kp2,
    good_matches, None,
    matchColor=(0, 255, 0),       # 매칭선 색상 (초록)
    singlePointColor=(255, 0, 0), # 매칭 안 된 특징점 색상 (파랑)
    flags=2
)
```

---

## 4. 실행 방법
- `mot_color70.jpg`, `mot_color80.jpg`를 스크립트와 **같은 폴더**에 위치시킴
- 스크립트 실행
```bash
python SIFT_matching.py
```

---

## 5. 실행 결과

<img width="1753" height="530" alt="sift_matching_result" src="https://github.com/user-attachments/assets/0caa24b7-e6bf-45b4-85d1-22b78aafc218" />

---
---

# 📌 과제 3 : SIFT 호모그래피 (이미지 정렬)

## 1. 문제 정의
- 두 이미지(`img1.jpg`, `img2.jpg`) 간 SIFT 특징점 대응점을 찾아 호모그래피 행렬 계산
- 한 이미지를 변환하여 다른 이미지와 정렬 (파노라마 형태)
- 변환된 이미지(Warped Image)와 매칭 결과를 나란히 출력

---

## 2. 주요 기능
- BFMatcher + `knnMatch()` + Lowe's Ratio Test로 정확한 매칭점 선별
- `findHomography()` + RANSAC으로 이상점(Outlier) 제거 및 호모그래피 행렬 계산
- `warpPerspective()`로 이미지 원근 변환 및 정렬
- 빈 영역(검정) 자동 크롭

---

## 3. 핵심 코드 설명

### BFMatcher 생성 및 매칭
```python
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)  # 유클리드 거리 기반 매처
matches = bf.knnMatch(des1, des2, k=2)            # 상위 2개 매칭 추출
```

### Lowe's Ratio Test + 상위 N개 선택
```python
good_matches = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:  # 임계값 낮출수록 매칭 수 감소
        good_matches.append(m)

# distance 기준 정렬 후 상위 N개만 선택
good_matches = sorted(good_matches, key=lambda x: x.distance)
good_matches = good_matches[:50]
```

### 호모그래피 행렬 계산
```python
src_pts = np.float32([kp2[m.trainIdx].pt  for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.queryIdx].pt  for m in good_matches]).reshape(-1, 1, 2)

# RANSAC으로 이상점 제거하며 호모그래피 계산 (허용 오차 5.0 픽셀)
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
```

### 원근 변환 및 정렬
```python
# 출력 크기 : 두 이미지를 합친 파노라마 크기
out_w = w1 + w2
out_h = max(h1, h2)

# img2를 H로 변환하여 img1 기준으로 정렬
warped_img2 = cv.warpPerspective(img2_color, H, (out_w, out_h))
result = warped_img2.copy()
result[0:h1, 0:w1] = img1_color  # img1 고정
```


## 4. 실행 방법
- `img1.jpg`, `img2.jpg`를 스크립트와 **같은 폴더**에 위치시킴
- 스크립트 실행
```bash
python SIFT_homography.py
```

---

## 5. 실행 결과

<img width="1790" height="471" alt="homography_result" src="https://github.com/user-attachments/assets/cc85e824-1129-4207-9178-0a9aadb574ea" />

---

# 📎 공통 참고사항

| 항목 | 내용 |
|------|------|
| **Python 버전** | Python 3.13.7 |
| **필요 라이브러리** | `opencv-python`, `matplotlib`, `numpy` |
| **이미지 위치** | 스크립트와 동일한 폴더에 위치 |
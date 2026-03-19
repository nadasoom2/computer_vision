# 과제 1 · Sobel 에지 검출 (`sobel_edge.py`)

## 📌 문제 정의
- 입력 이미지를 그레이스케일로 변환한 뒤 Sobel 필터를 적용하여 X·Y 방향의 에지를 검출
- 두 방향의 에지를 합산한 **에지 강도 이미지**를 생성하고 원본과 나란히 시각화

---

## ⚙️ 주요 기능
- BGR 이미지 → 그레이스케일 변환
- Sobel 필터로 X축(수직 에지), Y축(수평 에지) 개별 검출
- `cv.magnitude()`로 전체 에지 강도 계산
- 결과 이미지 저장 및 Matplotlib 시각화

---

## 🔍 핵심 코드 설명

**① Sobel 필터 적용**
- `CV_64F` 타입을 사용하여 음수 기울기(어두운→밝은 방향)도 손실 없이 보존
```python
sobelX = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # X 방향 (수직 에지)
sobelY = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # Y 방향 (수평 에지)
```

**② 에지 강도 계산 및 uint8 변환**
- 두 방향의 에지를 합쳐 전체 강도를 구한 뒤 0~255 범위로 변환
```python
magnitude = cv.magnitude(sobelX, sobelY)        # sqrt(X² + Y²)
magnitude_uint8 = cv.convertScaleAbs(magnitude) # float64 → uint8
```

**③ 결과 저장 및 시각화**
```python
cv.imwrite('sobel_result.jpg', magnitude_uint8)
plt.imshow(magnitude_uint8, cmap='gray')
```

---

## ▶️ 실행 방법
1. `edgeDetectionImage.jpg` 파일을 스크립트와 동일한 경로에 위치
2. 아래 명령어 실행
```bash
python sobel_edge.py
```
3. `sobel_result.jpg` 저장 및 시각화 창 출력 확인

---

## 실행 결과


---

# 과제 2 · 캐니 에지 검출 + 허프 직선 변환 (`canny_edge.py`)

## 📌 문제 정의
- 다보탑 이미지에 캐니 알고리즘으로 에지 맵을 생성
- 허프 변환으로 직선 성분을 검출하고 원본 이미지에 **빨간색 선**으로 표시

---

## ⚙️ 주요 기능
- `cv.Canny()`로 이중 임계값 기반 에지 맵 생성
- `cv.HoughLinesP()`로 확률적 허프 변환 직선 검출
- `cv.line()`으로 검출된 직선을 빨간색으로 시각화
- 결과 이미지 저장 및 Matplotlib 시각화

---

## 🔍 핵심 코드 설명

**① 캐니 에지 검출**
- `threshold1`: 이 값 미만의 픽셀은 에지에서 완전히 제외
- `threshold2`: 이 값 이상의 픽셀은 확실한 에지로 확정
- 두 값 사이의 픽셀은 확실한 에지와 연결된 경우에만 포함
```python
edges = cv.Canny(gray, threshold1=100, threshold2=200)
```

**② 허프 변환 직선 검출**
- `threshold`: 직선으로 인정할 최소 투표 수 (클수록 뚜렷한 직선만 검출)
- `minLineLength`: 직선으로 인정할 최소 픽셀 길이
- `maxLineGap`: 같은 직선으로 연결할 수 있는 최대 끊김 간격
```python
lines = cv.HoughLinesP(edges,
                       rho=1,
                       theta=np.pi / 180,
                       threshold=200,
                       minLineLength=100,
                       maxLineGap=7)
```

**③ 검출된 직선 그리기**
- 원본 이미지를 복사하여 직선을 덮어쓰고 원본은 보존
```python
result = img.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(result, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
```

---

## ▶️ 실행 방법
1. `dabo.jpg` 파일을 스크립트와 동일한 경로에 위치
2. 아래 명령어 실행
```bash
python canny_edge.py
```
3. `dabo_result.jpg` 저장 및 시각화 창 출력 확인

---

## 실행 결과



---

# 과제 3 · GrabCut 객체 추출 (`grabcut.py`)

## 📌 문제 정의
- 커피컵 이미지에서 사용자가 지정한 사각형 영역을 기반으로 GrabCut 알고리즘을 적용
- 전경(커피컵)과 배경을 분리하여 **배경이 제거된 이미지** 출력
- 어두운 커피 영역이 배경으로 오인되는 문제를 마스크 강제 지정으로 해결

---

## ⚙️ 주요 기능
- `cv.grabCut()`으로 대화식 전경/배경 분리 수행
- 커피 영역 오인식 문제 해결을 위한 **2단계 GrabCut 실행**
- `np.where()`로 이진 마스크 생성 후 배경 픽셀 제거
- 원본 / 마스크 / 배경 제거 이미지 3개를 Matplotlib으로 시각화

---

## 🔍 핵심 코드 설명

**① GrabCut 초기 설정**
- 마스크와 내부 모델을 반드시 아래 형식으로 초기화 (임의 수정 금지)
```python
mask      = np.zeros(img.shape[:2], np.uint8)
bgdModel  = np.zeros((1, 65), np.float64)  # 배경 GMM 모델 (수정 금지)
fgdModel  = np.zeros((1, 65), np.float64)  # 전경 GMM 모델 (수정 금지)
```

**② 관심 영역 설정**
- `(x, y, width, height)` 형식으로 커피컵 전체가 포함되도록 지정
- `x + width <= 이미지 가로`, `y + height <= 이미지 세로` 조건을 반드시 만족해야 함
```python
rect = (70, 50, 1100, 850)
```

**③ 2단계 GrabCut 실행**
- 1차: `GC_INIT_WITH_RECT` — 사각형 기반으로 전경/배경 1차 추정
- 중간: 어두운 커피 영역을 `GC_FGD`로 강제 지정하여 오인식 보정
- 2차: `GC_INIT_WITH_MASK` — 수정된 마스크를 힌트로 삼아 재분석
```python
# 1차 실행
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

# 커피 영역 강제 전경 지정
h, w = img.shape[:2]
mask[h//3 : h*2//3, w//3 : w*2//3] = cv.GC_FGD

# 2차 실행
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_MASK)
```

**④ 마스크 후처리 및 배경 제거**
- 전경(`GC_FGD=1`)과 전경 추정(`GC_PR_FGD=3`) 픽셀만 1로 변환
- 이진 마스크를 원본 이미지에 곱하여 배경을 검정(0)으로 제거
```python
mask2  = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype(np.uint8)
result = img * mask2[:, :, np.newaxis]  # (H,W) → (H,W,1) 차원 확장 후 브로드캐스팅
```

---

## ▶️ 실행 방법
1. `coffee cup.JPG` 파일을 스크립트와 동일한 경로에 위치
2. 아래 명령어 실행
```bash
python grabcut.py
```
3. `grabcut_result.jpg` 저장 및 시각화 창(원본 / 마스크 / 배경 제거) 출력 확인

---

## 실행 결과



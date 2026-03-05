# ROI Selection - 이미지 영역 선택 프로그램

## 프로그램 개요

`roi_selection.py`는 OpenCV를 사용하여 이미지에서 마우스를 이용해 관심 영역(ROI, Region of Interest)을 선택하고 저장하는 프로그램입니다. 사용자가 마우스로 드래그 하여 원하는 영역을 선택하면, 해당 영역을 추출하거나 파일로 저장할 수 있습니다.

## 주요 기능

- **마우스 기반 ROI 선택**: 마우스를 드래그하여 직사각형 영역 선택
- **실시간 선택 영역 표시**: 드래그 중 선택 영역을 실시간으로 화면에 표시
- **ROI 추출**: numpy 슬라이싱을 이용한 효율적인 영역 추출
- **이미지 저장**: 선택된 ROI를 새로운 이미지 파일로 저장
- **리셋 기능**: 선택 초기화 및 원본 이미지 복원

## 필수 요구사항

- Python 3.x
- OpenCV (cv2): `pip install opencv-python`
- NumPy (opencv-python과 함께 설치됨)

## 사용 방법

### 1. 단축키

| 단축키 | 기능 | 설명 |
|--------|------|------|
| 마우스 드래그 | ROI 선택 | 이미지에서 드래그하여 영역 선택 |
| **r** | 초기화 | ROI 선택을 초기화하고 원본 이미지 복원 |
| **s** | 저장 | 선택된 ROI를 'roi_image.jpg'로 저장 |
| **q** | 종료 | 프로그램 종료 |

## 핵심 코드 설명


### 1. 전역 변수 설정
```python
start_x, start_y = -1, -1    # ROI 시작 좌표
end_x, end_y = -1, -1        # ROI 끝 좌표
drawing = False               # 드래그 상태 저장
roi = None                    # 추출된 ROI 이미지
```

### 2. 마우스 이벤트 콜백 함수: `select_roi()`

#### 마우스 왼쪽 버튼 클릭 (EVENT_LBUTTONDOWN)
```python
if event == cv.EVENT_LBUTTONDOWN:
    drawing = True
    start_x, start_y = x, y
```
- 드래그 시작 시 ROI의 시작점 저장
- `drawing` 플래그를 True로 설정

#### 마우스 이동 (EVENT_MOUSEMOVE)
```python
elif event == cv.EVENT_MOUSEMOVE:
    if drawing:
        temp = img_original.copy()
        cv.rectangle(temp, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv.imshow('Image Display', temp)
```
- 드래그 중일 때만 실행
- 원본 이미지 복사본에 선택 중인 사각형을 실시간으로 표시
- 녹색(0, 255, 0) 사각형 테두리 표시 (두께: 2픽셀)

#### 마우스 왼쪽 버튼 떼기 (EVENT_LBUTTONUP)
```python
elif event == cv.EVENT_LBUTTONUP:
    drawing = False
    end_x, end_y = x, y
    
    cv.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    roi = img_original[min(start_y,end_y):max(start_y,end_y),
                       min(start_x,end_x):max(start_x,end_x)]
```
- ROI 끝점 저장
- 선택된 영역에 초록색 사각형 그리기
- **numpy 슬라이싱**: 이미지 배열에서 선택된 영역만 추출
  - `[y축:y축, x축:x축]` 순서 (이미지 배열은 행-열 순서)
  - `min/max` 함수: 사용자가 우상향 또는 좌하향 드래그해도 올바르게 추출

### 4. 마우스 이벤트 처리 등록
```python
cv.namedWindow('Image Display')
cv.setMouseCallback('Image Display', select_roi)
```
- 'Image Display' 창에서 발생하는 마우스 이벤트를 `select_roi()` 함수에 전달

### 5. 메인 루프 및 키보드 입력 처리
```python
while True:
    cv.imshow('Image Display', img)
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('r'):        # 'r' 키 - 초기화
        img = img_original.copy()
        roi = None
    
    elif key == ord('s'):      # 's' 키 - 저장
        if roi is not None:
            cv.imwrite('roi_image.jpg', roi)
            print("ROI 이미지가 저장되었습니다.")
    
    elif key == ord('q'):      # 'q' 키 - 종료
        break
```
- `cv.waitKey(1)`: 1밀리초 동안 키 입력 대기
- `& 0xFF`: 32비트 정수에서 실제 키 값만 추출
- `ord()`: 문자를 ASCII 코드로 변환


## 주요 개념

### ROI (Region of Interest)
- 이미지에서 분석 또는 처리 대상인 특정 영역
- 컴퓨터 비전에서 관심 있는 객체가 있는 영역을 선택하여 처리

### NumPy 슬라이싱 (배열 인덱싱)
- 이미지는 NumPy 3D 배열로 표현됨: `(높이, 너비, 채너(BGR))`
- `img[y1:y2, x1:x2]`: y1부터 y2까지, x1부터 x2까지의 영역 추출
- 효율적이고 빠른 영역 추출 방식

## 주의사항

1. **이미지 파일 경로**: 코드의 `cv.imread('soccer.jpg')`에서 'soccer.jpg' 파일이 프로그램과 같은 디렉토리에 있어야 합니다.
2. **좌표 순서**: 이미지 배열 인덱싱은 `[y, x]` 순서입니다 (행-열)
3. **극단점 처리**: `min/max` 함수를 사용하여 역방향 드래그(우상향, 좌하향)도 올바르게 처리합니다.

## 출력 파일

- **roi_image.jpg**: 's' 키를 눌러 저장한 ROI 이미지

![roi_image](https://github.com/user-attachments/assets/7633ae37-76c1-4dc9-ad78-d9f91bab09d5)



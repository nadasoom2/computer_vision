# grayscale.py – Grayscale & Resize 처리 스크립트

이 README는 `grayscale.py` 파일의 동작과 핵심 코드에 대한 설명을 제공합니다. OpenCV로 이미지를 읽어 축소하고, 그레이스케일 변환 후 원본과 결합하는 예제로, 컴퓨터 비전 수업에서 기초적인 이미지 처리 과정을 다루고 있습니다.

## 주요 기능

1. **이미지 로드 및 예외 처리**: `cv.imread`로 이미지를 불러오고, 파일이 존재하지 않으면 프로그램을 종료시킵니다.
2. **이미지 축소**: `cv.resize`를 이용해 가로/세로 크기를 각각 50%로 줄입니다.
3. **그레이스케일 변환**: 축소된 이미지를 `cv.cvtColor`로 흑백으로 변환합니다.
4. **파일 저장**: 축소된 원본, 그레이스케일 이미지, 결합된 이미지를 각각 JPEG 형식으로 저장합니다.
5. **채널 변환 및 결합**: 그레이스케일 이미지를 다시 3채널로 변환한 뒤 `numpy.hstack`을 이용해 원본과 나란히 이어붙입니다.
6. **결과 표시**: `cv.imshow`를 사용해 결합 이미지를 화면에 출력하고 키 입력을 기다린 뒤 종료합니다.

## 핵심 코드 설명

- `img = cv.imread('soccer.jpg')` : 작업 대상이 되는 원본 이미지를 읽어옵니다.
- `img_small = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5)` : 이미지 축소 비율을 설정하여 크기를 줄입니다.
- `gray = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)` : 컬러 이미지를 그레이스케일로 변환합니다.
- `gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)` : hstack을 위해 그레이스케일 이미지를 3채널로 확장합니다.
- `combined = np.hstack((img_small, gray_bgr))` : 두 이미지를 가로 방향으로 결합합니다.
- 저장 및 표시 관련 함수: `cv.imwrite`, `cv.imshow`, `cv.waitKey`, `cv.destroyAllWindows()`.

## 실행 방법

1. OpenCV와 NumPy가 설치된 Python 환경에서 실행합니다.
2. 동일한 디렉터리에 `soccer.jpg` 파일을 위치시킵니다.
3. `python grayscale.py` 명령으로 스크립트를 실행하면 결과 이미지들이 생성되고 창이 표시됩니다.

---

이 스크립트는 이미지 전처리 및 시각화의 기초를 배우는 데 유용하며, 추가적인 변환이나 필터를 실험하는 출발점으로 활용할 수 있습니다.


<img width="1431" height="501" alt="스크린샷 2026-03-05 160035" src="https://github.com/user-attachments/assets/bd7ea99a-aeff-4adf-8f7f-a9ee565df3f2" />

# adjust_pensize.py

이 프로젝트에는 OpenCV를 사용하여 이미지 위에 마우스로 드로잉하고 붓 크기를 조절할 수 있는 간단한 그래픽 프로그램이 포함되어 있습니다. `adjust_pensize.py` 파일을 실행하면 `soccer.jpg` 이미지를 불러와, 마우스 왼쪽/오른쪽 버튼으로 파란색/빨간색 선을 그리고 `+`/`-` 키로 붓 크기를 변경할 수 있습니다. `q` 키를 누르면 현재 작업 중인 이미지를 저장하고 프로그램을 종료합니다.

## 주요 기능

1. **마우스 드로잉**: `draw()` 콜백이 호출되어 클릭/드래그 시 선을 그림. 왼쪽 버튼은 파란색, 오른쪽 버튼은 빨간색.
2. **붓 크기 조절**: `+`/`-` 키로 붓 반경을 1부터 15까지 증감.
3. **저장 및 종료**: `q` 키를 누르면 현재 이미지를 `adjusted_pensize.jpg`로 저장하고 프로그램 종료.
4. **디버그 출력**: 종료 직후 콘솔에 이미지 객체 유형과 크기를 표시.

## 핵심 코드 설명

```python
def draw(event, x, y, flags, param):
    global drawing, color, brush_size, img

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        color = (255, 0, 0)
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = True
        color = (0, 0, 255)
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        cv.circle(img, (x, y), brush_size, color, -1)
    elif event in (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP):
        drawing = False
```
마우스 버튼에 따라 색을 정하고, 드래그 중에 원으로 선을 그립니다.

```python
cv.namedWindow('Image Display')
cv.setMouseCallback('Image Display', draw)

while True:
    cv.imshow('Image Display', img)
    key = cv.waitKey(1) & 0xFF
    if key == ord('+'):
        brush_size = min(15, brush_size + 1)
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)
    elif key == ord('q'):
        cv.imwrite('adjusted_pensize.jpg', img)
        break

cv.destroyAllWindows()
```
메인 루프에서는 창을 띄우고 키 입력으로 붓 크기 변경과 종료/저장 동작을 처리합니다.

> 다른 변수 초기화와 디버그 출력은 보조적인 부분이므로 생략했습니다.


---

> ⚠️ `soccer.jpg` 파일이 동일 디렉토리에 있어야 프로그램이 정상 동작합니다.

이 README는 코드의 목적과 주요 동작을 설명하며, 학습 및 수정 시 참고 자료로 사용할 수 있습니다.



![adjusted_pensize](https://github.com/user-attachments/assets/0d770f18-1100-4a12-8051-beda56832a43)


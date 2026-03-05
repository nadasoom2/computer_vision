# OpenCV 라이브러리를 불러옴 (이미지 처리에 사용)
import cv2 as cv

# 프로그램을 종료하기 위한 sys 모듈 불러오기
import sys


# soccer.jpg 이미지를 읽어서 img 변수에 저장
img = cv.imread('soccer.jpg')

# 이미지 파일이 존재하지 않을 경우 프로그램 종료
if img is None:
    sys.exit('파일이 존재하지 않습니다.')


# 초기 붓 크기를 5로 설정
brush_size = 5

# 현재 그림을 그리고 있는 상태인지 저장하는 변수 (처음에는 False)
drawing = False

# 기본 붓 색상 설정 (BGR 기준 파란색)
color = (255, 0, 0)


# 마우스 이벤트가 발생할 때 실행되는 함수 정의
def draw(event, x, y, flags, param):

    # 함수 밖에 있는 전역 변수들을 사용하기 위해 선언
    global drawing, color, brush_size, img

    # 마우스 왼쪽 버튼을 눌렀을 때
    if event == cv.EVENT_LBUTTONDOWN:
        # 그리기 시작 상태로 변경
        drawing = True
        # 붓 색상을 파란색으로 설정
        color = (255, 0, 0)

    # 마우스 오른쪽 버튼을 눌렀을 때
    elif event == cv.EVENT_RBUTTONDOWN:
        # 그리기 시작 상태로 변경
        drawing = True
        # 붓 색상을 빨간색으로 설정
        color = (0, 0, 255)

    # 마우스를 움직일 때
    elif event == cv.EVENT_MOUSEMOVE:
        # 현재 그리는 상태라면
        if drawing:
            # 현재 마우스 위치에 붓 크기만큼 원을 그려 붓질 효과 생성
            cv.circle(img, (x, y), brush_size, color, -1)

    # 마우스 왼쪽 또는 오른쪽 버튼을 뗐을 때
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        # 그리기 상태 종료
        drawing = False


# 이미지 창 생성
cv.namedWindow('Image Display')

# 해당 창에서 발생하는 마우스 이벤트를 draw 함수로 처리
cv.setMouseCallback('Image Display', draw)


# 프로그램이 계속 실행되도록 무한 반복
while True:

    # 현재 이미지를 화면에 출력
    cv.imshow('Image Display', img)

    # 1ms 동안 키 입력을 기다리고 입력된 키 값을 저장
    key = cv.waitKey(1) & 0xFF


    # + 키를 누르면 붓 크기를 1 증가 (최대 15)
    if key == ord('+'):
        brush_size = min(15, brush_size + 1)

    # - 키를 누르면 붓 크기를 1 감소 (최소 1)
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)

    # q 키를 누르면 반복문 종료
    elif key == ord('q'):
        # 현재 이미지를 파일로 저장
        cv.imwrite('adjusted_pensize.jpg', img)  
        break


# 모든 OpenCV 창을 닫음
cv.destroyAllWindows()


# img 변수의 데이터 타입 출력
print(type(img))

# img 이미지의 크기(높이, 너비, 채널) 출력
print(img.shape)
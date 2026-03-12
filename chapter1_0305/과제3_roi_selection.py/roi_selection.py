# OpenCV 라이브러리 불러오기
import cv2 as cv

# 프로그램 종료를 위한 sys 모듈 불러오기
import sys


# soccer.jpg 이미지를 읽어서 img 변수에 저장
img = cv.imread('chapter1_0305/soccer.jpg')

# 이미지가 존재하지 않을 경우 프로그램 종료
if img is None:
    sys.exit('파일이 존재하지 않습니다.')


# 원본 이미지를 따로 복사 (ROI 리셋할 때 사용)
img_original = img.copy()

# ROI 시작 좌표
start_x, start_y = -1, -1

# ROI 끝 좌표
end_x, end_y = -1, -1

# 현재 드래그 중인지 확인하는 변수
drawing = False

# 잘라낸 ROI 이미지를 저장할 변수
roi = None


# 마우스 이벤트 처리 함수
def select_roi(event, x, y, flags, param):

    # 전역 변수 사용 선언
    global start_x, start_y, end_x, end_y, drawing, img, roi

    # 마우스 왼쪽 버튼을 눌렀을 때 (ROI 시작)
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    # 마우스를 움직이는 동안 (드래그 중)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            # 원본 이미지를 복사하여 사각형을 계속 업데이트
            temp = img_original.copy()
            cv.rectangle(temp, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv.imshow('Image Display', temp)

    # 마우스 버튼을 놓았을 때 (ROI 선택 완료)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

        # 사각형 표시
        cv.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # numpy 슬라이싱을 이용해 ROI 추출
        roi = img_original[min(start_y,end_y):max(start_y,end_y),
                           min(start_x,end_x):max(start_x,end_x)]

        # ROI가 정상적으로 존재하면 새로운 창에 출력
        if roi.size != 0:
            cv.imshow('ROI', roi)


# 이미지 창 생성
cv.namedWindow('Image Display')

# 해당 창에서 발생하는 마우스 이벤트를 select_roi 함수로 처리
cv.setMouseCallback('Image Display', select_roi)


# 무한 루프를 통해 키 입력을 지속적으로 확인
while True:

    # 현재 이미지를 화면에 출력
    cv.imshow('Image Display', img)

    # 1ms 동안 키 입력 대기
    key = cv.waitKey(1) & 0xFF


    # r 키를 누르면 ROI 선택을 초기화
    if key == ord('r'):
        img = img_original.copy()
        roi = None


    # s 키를 누르면 선택된 ROI 이미지를 파일로 저장
    elif key == ord('s'):
        if roi is not None:
            cv.imwrite('roi_image.jpg', roi)
            print("ROI 이미지가 저장되었습니다.")


    # q 키를 누르면 프로그램 종료
    elif key == ord('q'):
        break


# 모든 OpenCV 창 닫기
cv.destroyAllWindows()


# img 변수의 데이터 타입 출력
print(type(img))

# img 이미지의 크기(높이, 너비, 채널) 출력
print(img.shape)
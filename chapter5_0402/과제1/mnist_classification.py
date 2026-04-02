# MNIST 데이터셋, 모델 구성, 레이어 관련 모듈을 불러옴
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# MNIST 데이터셋을 로드하여 훈련 세트와 테스트 세트로 자동 분할
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 픽셀 값을 0~255에서 0.0~1.0 사이로 정규화하여 학습 효율을 높임
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# Sequential 모델 생성 (레이어를 순서대로 쌓는 구조)
model = Sequential([
    # 28x28 2D 이미지를 784개의 1D 벡터로 펼침 (입력층 역할)
    Flatten(input_shape=(28, 28)),
    # 128개의 뉴런을 가진 은닉층, 활성화 함수로 ReLU 사용
    Dense(128, activation='relu'),
    # 64개의 뉴런을 가진 은닉층, 활성화 함수로 ReLU 사용
    Dense(64, activation='relu'),
    # 출력층: 0~9 숫자 10개 클래스에 대한 확률을 출력하는 Softmax 사용
    Dense(10, activation='softmax'),
])

# 모델 컴파일: 옵티마이저, 손실 함수, 평가 지표 설정
model.compile(
    # Adam 옵티마이저로 가중치를 효율적으로 업데이트
    optimizer='adam',
    # 정수 레이블에 맞는 다중 분류 손실 함수 사용
    loss='sparse_categorical_crossentropy',
    # 훈련 및 평가 시 정확도를 측정 지표로 사용
    metrics=['accuracy']
)

# 모델 구조 요약 출력
model.summary()

# 훈련 세트로 모델을 5 에포크 동안 학습, 검증 데이터로 테스트 세트 사용
history = model.fit(
    x_train, y_train,
    # 한 번에 처리할 샘플 수 (배치 크기)
    batch_size=32,
    # 전체 훈련 데이터를 반복 학습할 횟수
    epochs=5,
    # 각 에포크마다 검증 데이터로 성능 확인
    validation_data=(x_test, y_test)
)

# 테스트 세트로 최종 손실값과 정확도를 평가
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# 최종 테스트 정확도를 출력
print(f"\n테스트 정확도: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

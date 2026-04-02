import os
# TensorFlow 로그 레벨 설정 (불필요한 경고 메시지 숨김)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
# TensorFlow 및 Keras 임포트
import tensorflow as tf
# TensorFlow 2.16+ 에서는 keras를 독립 패키지로 직접 임포트
from keras import layers, models
from keras.datasets import cifar10
# 이미지 로드 및 전처리를 위한 라이브러리
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# ── GPU 설정 ────────────────────────────────────────────────
# 사용 가능한 GPU 목록 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리를 필요한 만큼만 동적으로 할당 (메모리 낭비 방지)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU 사용 가능: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    # GPU가 없을 경우 CPU로 대체 실행
    print("GPU를 찾을 수 없습니다. CPU로 실행합니다.")

# ── 데이터 로드 ──────────────────────────────────────────────
# CIFAR-10 데이터셋을 훈련용(50,000장)과 테스트용(10,000장)으로 나눠 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# CIFAR-10의 10개 클래스 이름 정의
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"훈련 데이터 shape: {x_train.shape}")   # (50000, 32, 32, 3)
print(f"테스트 데이터 shape: {x_test.shape}")  # (10000, 32, 32, 3)

# ── 데이터 전처리 ────────────────────────────────────────────
# 픽셀 값을 0~255에서 0.0~1.0 범위로 정규화 → 학습 수렴 속도 향상
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# ── CNN 모델 설계 ────────────────────────────────────────────
model = models.Sequential([

    # ── 첫 번째 합성곱 블록 ──────────────────────────────────
    # 32개의 3×3 필터로 특징 맵 추출, 입력 shape=(32, 32, 3)
    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                  input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # ── 두 번째 합성곱 블록 ──────────────────────────────────
    # 64개의 3×3 필터로 중간 수준 특징 추출
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # ── 완전 연결(분류) 블록 ─────────────────────────────────
    layers.Flatten(),
    # 분류기 크기를 줄여 과적합과 학습 편차를 완화
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax'),
])

# ── 모델 컴파일 ──────────────────────────────────────────────
model.compile(
    # Adam 옵티마이저: 학습률 자동 조정으로 빠른 수렴
    optimizer='adam',
    # 정수형 레이블이므로 sparse_categorical_crossentropy 사용
    loss='sparse_categorical_crossentropy',
    # 정확도를 평가 지표로 사용
    metrics=['accuracy']
)

# ── 콜백 설정 ────────────────────────────────────────────────
# val_loss가 5 에폭 동안 개선되지 않으면 학습 조기 종료 (최적 가중치 복원)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
# val_loss가 3 에폭 동안 개선되지 않으면 학습률을 0.5배로 줄임
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
)

# ── 모델 훈련 ────────────────────────────────────────────────
history = model.fit(
    x_train, y_train,
    # 한 번에 64개 샘플씩 처리
    batch_size=64,
    # 최대 30 에폭 훈련 (early stopping으로 조기 종료 가능)
    epochs=20,
    # 훈련 데이터의 20%를 검증용으로 사용
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ── 테스트셋 성능 평가 ───────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss:     {test_loss:.4f}")

# ── 학습 곡선 시각화 ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 정확도 그래프
ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# 손실 그래프
ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
# 학습 곡선 이미지를 파일로 저장
plt.savefig('training_history.png', dpi=150)
plt.show()

# ── dog.jpg 예측 ─────────────────────────────────────────────
# 예측할 이미지 파일 경로 설정
img_path = 'chapter5_0402/dog.jpg'

if os.path.exists(img_path):
    # 이미지를 CIFAR-10 입력 크기(32×32)로 리사이즈하여 로드
    img = load_img(img_path, target_size=(32, 32))
    # PIL Image를 NumPy 배열로 변환 → shape: (32, 32, 3)
    img_array = img_to_array(img)
    # 훈련 데이터와 동일하게 0~1 범위로 정규화
    img_array = img_array / 255.0
    # 배치 차원 추가 → shape: (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 모델로 클래스별 확률 예측
    predictions = model.predict(img_array)
    # 가장 높은 확률의 클래스 인덱스 선택
    predicted_index = np.argmax(predictions[0])
    # 인덱스를 클래스 이름으로 변환
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    print(f"\n[dog.jpg Prediction Result]")
    print(f"  Predicted Class: {predicted_class}")
    print(f"  Confidence:      {confidence:.2f}%")

    # 예측 결과와 원본 이미지를 함께 시각화
    plt.figure(figsize=(4, 4))
    plt.imshow(load_img(img_path, target_size=(128, 128)))
    plt.title(f"Prediction: {predicted_class} ({confidence:.1f}%)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150)
    plt.show()
else:
    # dog.jpg 파일이 없을 경우 테스트셋에서 임의로 개(dog) 이미지 사용
    print(f"\n'{img_path}' not found. Using a dog image from the test set instead.")
    # CIFAR-10에서 dog 클래스 인덱스는 5
    dog_indices = np.where(y_test.flatten() == 5)[0]
    sample_idx  = dog_indices[0]

    sample_img   = x_test[sample_idx]
    sample_label = class_names[y_test[sample_idx][0]]

    # 배치 차원 추가 후 예측
    pred = model.predict(np.expand_dims(sample_img, axis=0))
    pred_class  = class_names[np.argmax(pred[0])]
    pred_conf   = np.max(pred[0]) * 100

    print(f"  Actual Class:    {sample_label}")
    print(f"  Predicted Class: {pred_class}")
    print(f"  Confidence:      {pred_conf:.2f}%")

    plt.figure(figsize=(5, 5))
    # 저해상도(32×32) 이미지를 보기 좋게 확대하여 출력
    plt.imshow(sample_img, interpolation='bicubic')
    plt.title(f"Actual: {sample_label}\nPredicted: {pred_class}  ({pred_conf:.1f}%)",
              fontsize=13, fontweight='bold', pad=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150)
    plt.show()

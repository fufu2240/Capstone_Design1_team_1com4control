from ultralytics import YOLO
import os
import shutil


# 설정: 학습 파라미터 정의
DATA_YAML_PATH = "my_dataset/dataset.yaml"  # 데이터셋 yaml 경로
PRETRAINED_MODEL_PATH = "fire_smoke_yolov8n.pt"  # 사전학습된 모델 (또는 yolov8n.pt)
SAVE_RUN_NAME = "fire_smoke_transfer"  # 결과 저장 폴더 이름
IMAGE_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
DEVICE = "cuda"  # "cpu" 도 가능

# 학습 결과 경로 초기화 (선택)

SAVE_DIR = f"runs/detect/{SAVE_RUN_NAME}"
if os.path.exists(SAVE_DIR):
    print(f"[Info] 기존 결과 삭제: {SAVE_DIR}")
    shutil.rmtree(SAVE_DIR)

# 모델 로딩
model = YOLO(PRETRAINED_MODEL_PATH)
print("[Info] 모델 로드 완료")

# 전이학습 실행
model.train(
    data=DATA_YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    name=SAVE_RUN_NAME,
    project="runs/detect",
    pretrained=True,
    single_cls=False,       # 클래스가 하나뿐이면 True
    verbose=True,
    exist_ok=True,
    patience=20,
    optimizer="SGD",        # Adam도 가능
    lr0=0.01,               # 초기 학습률
    lrf=0.01,               # 최종 학습률
    warmup_epochs=3,
    weight_decay=0.001,     # 과적합 방지
    mosaic=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # 데이터 증강 설정
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0
)

# 최종 모델 경로 출력
print("[Done] 학습 완료. 최종 모델 경로:")
print(f"{SAVE_DIR}/weights/best.pt")